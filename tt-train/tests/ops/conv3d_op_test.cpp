// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ops/conv3d_op.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/binary_ops.hpp"

namespace {

using Dims3 = ttml::ops::Conv3dDims;

struct Conv3dCase {
    uint32_t N = 1;
    Dims3 in_size{4, 5, 6};  // D, H, W
    uint32_t C_in = 32;
    uint32_t C_out = 32;
    Dims3 kernel{3, 3, 3};
    Dims3 stride{1, 1, 1};
    Dims3 padding{0, 0, 0};
    Dims3 dilation{1, 1, 1};
    uint32_t groups = 1;
    bool with_bias = true;
    bool bias_rank1 = false;  // pass the bias as [C_out] instead of [1, 1, 1, C_out]
    bool input_requires_grad = true;
    bool weight_requires_grad = true;
    bool bias_requires_grad = true;
    ttnn::Layout layout = ttnn::Layout::ROW_MAJOR;

    [[nodiscard]] Dims3 out_size() const {
        Dims3 out{};
        for (size_t i = 0; i < 3; ++i) {
            const uint32_t effective_kernel = dilation[i] * (kernel[i] - 1) + 1;
            out[i] = (in_size[i] + 2 * padding[i] - effective_kernel) / stride[i] + 1;
        }
        return out;
    }
};

// Plain-float CPU reference for channels-last conv3d and its gradients.
struct Conv3dReference {
    std::vector<float> output;       // [N, D_out, H_out, W_out, C_out]
    std::vector<float> grad_input;   // [N, D, H, W, C_in]
    std::vector<float> grad_weight;  // [C_out, C_in / groups, kD, kH, kW]
    std::vector<float> grad_bias;    // [C_out]
};

Conv3dReference compute_reference(
    const Conv3dCase& c,
    const std::vector<float>& input,
    const std::vector<float>& weight,
    const std::vector<float>& bias,
    const std::vector<float>& grad_output) {
    const auto out = c.out_size();
    const auto [D, H, W] = c.in_size;
    const auto [kD, kH, kW] = c.kernel;

    Conv3dReference ref;
    ref.output.assign(static_cast<size_t>(c.N) * out[0] * out[1] * out[2] * c.C_out, 0.F);
    ref.grad_input.assign(input.size(), 0.F);
    ref.grad_weight.assign(weight.size(), 0.F);
    ref.grad_bias.assign(c.C_out, 0.F);

    auto input_index = [&](uint32_t n, uint32_t d, uint32_t h, uint32_t w, uint32_t ci) {
        return (((static_cast<size_t>(n) * D + d) * H + h) * W + w) * c.C_in + ci;
    };
    auto output_index = [&](uint32_t n, uint32_t d, uint32_t h, uint32_t w, uint32_t co) {
        return (((static_cast<size_t>(n) * out[0] + d) * out[1] + h) * out[2] + w) * c.C_out + co;
    };
    const uint32_t C_in_per_group = c.C_in / c.groups;
    const uint32_t C_out_per_group = c.C_out / c.groups;
    // ci_local indexes the group's own input channels.
    auto weight_index = [&](uint32_t co, uint32_t ci_local, uint32_t kd, uint32_t kh, uint32_t kw) {
        return (((static_cast<size_t>(co) * C_in_per_group + ci_local) * kD + kd) * kH + kh) * kW + kw;
    };

    for (uint32_t n = 0; n < c.N; ++n) {
        for (uint32_t od = 0; od < out[0]; ++od) {
            for (uint32_t oh = 0; oh < out[1]; ++oh) {
                for (uint32_t ow = 0; ow < out[2]; ++ow) {
                    for (uint32_t co = 0; co < c.C_out; ++co) {
                        const size_t o_idx = output_index(n, od, oh, ow, co);
                        const float dy = grad_output[o_idx];
                        const uint32_t ci_begin = (co / C_out_per_group) * C_in_per_group;
                        float acc = c.with_bias ? bias[co] : 0.F;
                        if (c.with_bias) {
                            ref.grad_bias[co] += dy;
                        }
                        for (uint32_t kd = 0; kd < kD; ++kd) {
                            const int64_t id = static_cast<int64_t>(od) * c.stride[0] + kd * c.dilation[0] -
                                               static_cast<int64_t>(c.padding[0]);
                            if (id < 0 || id >= static_cast<int64_t>(D)) {
                                continue;
                            }
                            for (uint32_t kh = 0; kh < kH; ++kh) {
                                const int64_t ih = static_cast<int64_t>(oh) * c.stride[1] + kh * c.dilation[1] -
                                                   static_cast<int64_t>(c.padding[1]);
                                if (ih < 0 || ih >= static_cast<int64_t>(H)) {
                                    continue;
                                }
                                for (uint32_t kw = 0; kw < kW; ++kw) {
                                    const int64_t iw = static_cast<int64_t>(ow) * c.stride[2] + kw * c.dilation[2] -
                                                       static_cast<int64_t>(c.padding[2]);
                                    if (iw < 0 || iw >= static_cast<int64_t>(W)) {
                                        continue;
                                    }
                                    for (uint32_t ci_local = 0; ci_local < C_in_per_group; ++ci_local) {
                                        const size_t i_idx = input_index(
                                            n,
                                            static_cast<uint32_t>(id),
                                            static_cast<uint32_t>(ih),
                                            static_cast<uint32_t>(iw),
                                            ci_begin + ci_local);
                                        const size_t w_idx = weight_index(co, ci_local, kd, kh, kw);
                                        acc += input[i_idx] * weight[w_idx];
                                        ref.grad_input[i_idx] += dy * weight[w_idx];
                                        ref.grad_weight[w_idx] += dy * input[i_idx];
                                    }
                                }
                            }
                        }
                        ref.output[o_idx] = acc;
                    }
                }
            }
        }
    }
    return ref;
}

std::vector<float> uniform_vector(size_t size, float lo, float hi, uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> data(size);
    std::generate(data.begin(), data.end(), [&]() { return dist(gen); });
    return data;
}

ttnn::Tensor make_device_tensor(const std::vector<float>& data, const ttnn::Shape& shape, ttnn::Layout layout) {
    auto* device = &ttml::autograd::ctx().get_device();
    return ttml::core::from_vector<float, ttnn::DataType::BFLOAT16>(data, shape, device, layout);
}

// Pearson correlation between two tensors, the structural check that a max-abs bound alone cannot give.
double pcc(const std::vector<float>& a, const std::vector<float>& b) {
    double mean_a = 0.0;
    double mean_b = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        mean_a += a[i];
        mean_b += b[i];
    }
    mean_a /= static_cast<double>(a.size());
    mean_b /= static_cast<double>(a.size());
    double cov = 0.0;
    double var_a = 0.0;
    double var_b = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double da = a[i] - mean_a;
        const double db = b[i] - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }
    return cov / std::sqrt(var_a * var_b);
}

// Every element must be within abs_floor + rel_of_max * max|expected| of the reference (bf16 rounding of the result
// is relative to the tensor's largest magnitude, not to each element), and the two tensors must correlate above
// 0.999 so small-magnitude entries cannot hide behind the bound.
void expect_close(
    const std::vector<float>& actual,
    const std::vector<float>& expected,
    float rel_of_max,
    float abs_floor,
    const char* what) {
    ASSERT_EQ(actual.size(), expected.size()) << what;
    float max_abs_err = 0.F;
    float max_ref = 0.F;
    size_t worst = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const float err = std::abs(actual[i] - expected[i]);
        max_ref = std::max(max_ref, std::abs(expected[i]));
        if (err > max_abs_err) {
            max_abs_err = err;
            worst = i;
        }
    }
    const float tolerance = abs_floor + rel_of_max * max_ref;
    EXPECT_LE(max_abs_err, tolerance) << what << ": worst index " << worst << " actual " << actual[worst]
                                      << " expected " << expected[worst] << " (max |ref| " << max_ref << ")";
    EXPECT_GT(pcc(actual, expected), 0.999) << what;
}

class Conv3dOpTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

void run_case(const Conv3dCase& c) {
    const auto out = c.out_size();
    const auto [D, H, W] = c.in_size;
    const auto [kD, kH, kW] = c.kernel;

    const ttnn::Shape input_shape({c.N, D, H, W, c.C_in});
    const ttnn::Shape weight_shape({c.C_out, c.C_in / c.groups, kD, kH, kW});
    const ttnn::Shape bias_shape = c.bias_rank1 ? ttnn::Shape({c.C_out}) : ttnn::Shape({1U, 1U, 1U, c.C_out});
    const ttnn::Shape output_shape({c.N, out[0], out[1], out[2], c.C_out});

    auto input_data = uniform_vector(input_shape.volume(), -1.F, 1.F, /*seed=*/1);
    auto weight_data = uniform_vector(weight_shape.volume(), -0.5F, 0.5F, /*seed=*/2);
    auto bias_data = uniform_vector(c.C_out, -1.F, 1.F, /*seed=*/3);
    auto grad_output_data = uniform_vector(output_shape.volume(), -1.F, 1.F, /*seed=*/4);

    auto input =
        ttml::autograd::create_tensor(make_device_tensor(input_data, input_shape, c.layout), c.input_requires_grad);
    auto weight =
        ttml::autograd::create_tensor(make_device_tensor(weight_data, weight_shape, c.layout), c.weight_requires_grad);
    ttml::autograd::TensorPtr bias;
    if (c.with_bias) {
        // A rank-1 bias cannot be tilized on its own; the op accepts it in ROW_MAJOR.
        const auto bias_layout = c.bias_rank1 ? ttnn::Layout::ROW_MAJOR : c.layout;
        bias =
            ttml::autograd::create_tensor(make_device_tensor(bias_data, bias_shape, bias_layout), c.bias_requires_grad);
    }
    auto grad_output = make_device_tensor(grad_output_data, output_shape, c.layout);

    // Build the reference from the bf16-rounded values that actually live on device.
    input_data = ttml::core::to_vector(input->get_value());
    weight_data = ttml::core::to_vector(weight->get_value());
    if (c.with_bias) {
        bias_data = ttml::core::to_vector(bias->get_value());
    }
    grad_output_data = ttml::core::to_vector(grad_output);
    const auto ref = compute_reference(c, input_data, weight_data, bias_data, grad_output_data);

    auto result = ttml::ops::conv3d(input, weight, bias, c.stride, c.padding, c.dilation, c.groups);

    ASSERT_EQ(result->get_value().logical_shape(), output_shape);
    EXPECT_EQ(result->get_value().layout(), c.layout);
    expect_close(ttml::core::to_vector(result->get_value()), ref.output, 1e-2F, 2e-2F, "forward");

    const bool any_grad = c.input_requires_grad || c.weight_requires_grad || (c.with_bias && c.bias_requires_grad);
    if (!any_grad) {
        EXPECT_FALSE(result->get_requires_grad());
        return;
    }
    result->set_grad(grad_output);
    result->backward();

    if (c.input_requires_grad) {
        ASSERT_TRUE(input->is_grad_initialized());
        EXPECT_EQ(input->get_grad().logical_shape(), input_shape);
        EXPECT_EQ(input->get_grad().layout(), c.layout);
        expect_close(ttml::core::to_vector(input->get_grad()), ref.grad_input, 1e-2F, 2e-2F, "grad_input");
    } else {
        EXPECT_FALSE(input->is_grad_initialized()) << "frozen input received a gradient";
    }
    if (c.weight_requires_grad) {
        ASSERT_TRUE(weight->is_grad_initialized());
        EXPECT_EQ(weight->get_grad().logical_shape(), weight_shape);
        EXPECT_EQ(weight->get_grad().layout(), c.layout);
        expect_close(ttml::core::to_vector(weight->get_grad()), ref.grad_weight, 1e-2F, 2e-2F, "grad_weight");
    } else {
        EXPECT_FALSE(weight->is_grad_initialized()) << "frozen weight received a gradient";
    }
    if (c.with_bias) {
        if (c.bias_requires_grad) {
            ASSERT_TRUE(bias->is_grad_initialized());
            EXPECT_EQ(bias->get_grad().logical_shape(), bias_shape);
            EXPECT_EQ(bias->get_grad().layout(), bias->get_value().layout());
            expect_close(ttml::core::to_vector(bias->get_grad()), ref.grad_bias, 1e-2F, 2e-2F, "grad_bias");
        } else {
            EXPECT_FALSE(bias->is_grad_initialized()) << "frozen bias received a gradient";
        }
    }
}

}  // namespace

TEST_F(Conv3dOpTest, ForwardBackwardWithBias) {
    Conv3dCase c;
    c.padding = {1, 1, 1};
    run_case(c);
}

TEST_F(Conv3dOpTest, ForwardBackwardNoBias) {
    Conv3dCase c;
    c.with_bias = false;
    run_case(c);
}

TEST_F(Conv3dOpTest, BatchStrideAndPadding) {
    Conv3dCase c;
    c.N = 2;
    c.in_size = {5, 6, 7};
    c.stride = {2, 2, 2};
    c.padding = {1, 1, 1};
    run_case(c);
}

TEST_F(Conv3dOpTest, AsymmetricStrideAndPadding) {
    Conv3dCase c;
    c.in_size = {6, 7, 8};
    c.kernel = {3, 3, 3};
    c.stride = {1, 2, 3};
    c.padding = {0, 1, 2};
    run_case(c);
}

TEST_F(Conv3dOpTest, Dilation) {
    Conv3dCase c;
    c.in_size = {6, 7, 8};
    c.dilation = {2, 2, 2};
    c.padding = {2, 2, 2};
    run_case(c);
}

TEST_F(Conv3dOpTest, PaddingLargerThanKernelSpanPointwise) {
    Conv3dCase c;
    c.kernel = {1, 1, 1};
    c.padding = {1, 1, 1};
    run_case(c);
}

// kernel span 2 (3x3x3, dilation 1) with padding 3, 4 and 3: exercises the dX crop offset for p > span, which the
// pointwise case above cannot (its span is 0).
TEST_F(Conv3dOpTest, PaddingLargerThanKernelSpan) {
    Conv3dCase c;
    c.padding = {3, 4, 3};
    run_case(c);
}

TEST_F(Conv3dOpTest, PaddingLargerThanKernelSpanWithStride) {
    Conv3dCase c;
    c.N = 2;
    c.stride = {2, 2, 2};
    c.padding = {3, 3, 3};
    run_case(c);
}

TEST_F(Conv3dOpTest, OutputChannelsNotTileAligned) {
    Conv3dCase c;
    c.C_out = 40;
    c.padding = {1, 1, 1};
    run_case(c);
}

TEST_F(Conv3dOpTest, InputChannelsNotTileAligned) {
    Conv3dCase c;
    c.C_in = 12;
    c.padding = {1, 1, 1};
    run_case(c);
}

TEST_F(Conv3dOpTest, InputAndOutputChannelsNotTileAligned) {
    Conv3dCase c;
    c.C_in = 3;
    c.C_out = 40;
    c.N = 2;
    c.stride = {2, 1, 2};
    c.padding = {1, 1, 1};
    run_case(c);
}

TEST_F(Conv3dOpTest, GroupsWithUnalignedChannels) {
    Conv3dCase c;
    c.C_in = 24;
    c.C_out = 16;
    c.groups = 2;
    c.padding = {1, 1, 1};
    run_case(c);
}

TEST_F(Conv3dOpTest, PointwiseKernelWithStride) {
    Conv3dCase c;
    c.kernel = {1, 1, 1};
    c.stride = {2, 2, 2};
    c.C_in = 64;
    run_case(c);
}

// Even kernel volumes lower the conv3d channel block from 32 to 16, so the padded channel count spans two
// blocks and the weight re-blocking permute is exercised; asymmetric kernels expose any kD/kH/kW axis mix-up.
TEST_F(Conv3dOpTest, PatchEmbedKernel) {
    Conv3dCase c;
    c.C_in = 16;
    c.C_out = 40;
    c.kernel = {1, 2, 2};
    c.stride = {1, 2, 2};
    run_case(c);
}

TEST_F(Conv3dOpTest, AsymmetricOddKernel) {
    Conv3dCase c;
    c.kernel = {1, 3, 3};
    c.padding = {0, 1, 1};
    run_case(c);
}

TEST_F(Conv3dOpTest, EvenCubicKernelWithStride) {
    Conv3dCase c;
    c.kernel = {2, 2, 2};
    c.stride = {2, 2, 2};
    run_case(c);
}

TEST_F(Conv3dOpTest, Groups) {
    Conv3dCase c;
    c.C_in = 64;
    c.C_out = 32;
    c.groups = 2;
    c.padding = {1, 1, 1};
    run_case(c);
}

TEST_F(Conv3dOpTest, GroupsWithStrideAndBatch) {
    Conv3dCase c;
    c.N = 2;
    c.C_in = 64;
    c.C_out = 64;
    c.groups = 4;
    c.stride = {2, 2, 2};
    c.padding = {1, 1, 1};
    c.in_size = {5, 6, 7};
    run_case(c);
}

TEST_F(Conv3dOpTest, TileLayoutTensors) {
    Conv3dCase c;
    c.layout = ttnn::Layout::TILE;
    c.padding = {1, 1, 1};
    run_case(c);
}

TEST_F(Conv3dOpTest, OneDimensionalBias) {
    Conv3dCase c;
    c.bias_rank1 = true;
    c.padding = {1, 1, 1};
    run_case(c);
}

TEST_F(Conv3dOpTest, LargerChannelCounts) {
    Conv3dCase c;
    c.C_in = 128;
    c.C_out = 128;
    c.padding = {1, 1, 1};
    run_case(c);
}

TEST_F(Conv3dOpTest, TileLayoutUnalignedGroupedBatch) {
    Conv3dCase c;
    c.layout = ttnn::Layout::TILE;
    c.N = 2;
    c.C_in = 24;
    c.C_out = 40;
    c.groups = 2;
    c.padding = {1, 1, 1};
    run_case(c);
}

// LoRA-style: frozen base weight, only the input gradient flows.
TEST_F(Conv3dOpTest, FrozenWeightAndBias) {
    Conv3dCase c;
    c.weight_requires_grad = false;
    c.bias_requires_grad = false;
    c.padding = {1, 1, 1};
    run_case(c);
}

// Latents as input: no input gradient, only dW and dB.
TEST_F(Conv3dOpTest, FrozenInput) {
    Conv3dCase c;
    c.input_requires_grad = false;
    c.padding = {1, 1, 1};
    run_case(c);
}

TEST_F(Conv3dOpTest, FrozenWeightTrainableBias) {
    Conv3dCase c;
    c.input_requires_grad = false;
    c.weight_requires_grad = false;
    run_case(c);
}

TEST_F(Conv3dOpTest, NoGradientsRequested) {
    Conv3dCase c;
    c.input_requires_grad = false;
    c.weight_requires_grad = false;
    c.bias_requires_grad = false;
    run_case(c);
}

TEST_F(Conv3dOpTest, PreparedWeightsMatchOnTheFlyPreparation) {
    const Conv3dCase c{.C_in = 64, .C_out = 40, .padding = {1, 1, 1}, .groups = 2};
    const auto [D, H, W] = c.in_size;
    const auto [kD, kH, kW] = c.kernel;
    const ttnn::Shape input_shape({c.N, D, H, W, c.C_in});
    const ttnn::Shape weight_shape({c.C_out, c.C_in / c.groups, kD, kH, kW});
    const ttnn::Shape output_shape({c.N, 4U, 5U, 6U, c.C_out});

    auto input = ttml::autograd::create_tensor(
        make_device_tensor(uniform_vector(input_shape.volume(), -1.F, 1.F, 21), input_shape, c.layout), true);
    auto weight = ttml::autograd::create_tensor(
        make_device_tensor(uniform_vector(weight_shape.volume(), -0.5F, 0.5F, 22), weight_shape, c.layout), true);
    auto grad_output = make_device_tensor(uniform_vector(output_shape.volume(), -1.F, 1.F, 23), output_shape, c.layout);

    auto reference = ttml::ops::conv3d(input, weight, nullptr, c.stride, c.padding, c.dilation, c.groups);
    reference->set_grad(grad_output);
    reference->backward();
    const auto ref_out = ttml::core::to_vector(reference->get_value());
    const auto ref_dx = ttml::core::to_vector(input->get_grad());
    const auto ref_dw = ttml::core::to_vector(weight->get_grad());
    input->set_grad(ttnn::Tensor());
    weight->set_grad(ttnn::Tensor());
    ttml::autograd::ctx().reset_graph();

    const auto prepared = ttml::ops::prepare_conv3d_weight(weight->get_value(), c.groups);
    EXPECT_EQ(prepared.forward.size(), c.groups);
    EXPECT_EQ(prepared.transposed.size(), c.groups);
    auto result = ttml::ops::conv3d(input, weight, nullptr, prepared, c.stride, c.padding, c.dilation, c.groups);
    result->set_grad(grad_output);
    result->backward();

    // Same kernels on the same data: bit-identical is the expectation, the tolerance only absorbs nothing.
    expect_close(ttml::core::to_vector(result->get_value()), ref_out, 0.F, 1e-6F, "prepared forward");
    expect_close(ttml::core::to_vector(input->get_grad()), ref_dx, 0.F, 1e-6F, "prepared grad_input");
    expect_close(ttml::core::to_vector(weight->get_grad()), ref_dw, 0.F, 1e-6F, "prepared grad_weight");

    // A prepared weight built for a different shape or group count is rejected.
    const auto other = ttml::ops::prepare_conv3d_weight(weight->get_value(), /*groups=*/1);
    EXPECT_THROW(
        ttml::ops::conv3d(input, weight, nullptr, other, c.stride, c.padding, c.dilation, c.groups),
        std::invalid_argument);
}

// Every training step must hit the program cache: a miss in any of the composed ttnn ops would recompile per step.
TEST_F(Conv3dOpTest, ProgramCacheStableAcrossSteps) {
    auto* device = &ttml::autograd::ctx().get_device();
    Conv3dCase c;
    c.N = 2;
    c.C_out = 40;
    c.stride = {1, 2, 2};
    c.padding = {1, 1, 1};
    const auto out = c.out_size();
    const auto [D, H, W] = c.in_size;
    const auto [kD, kH, kW] = c.kernel;
    const ttnn::Shape input_shape({c.N, D, H, W, c.C_in});
    const ttnn::Shape weight_shape({c.C_out, c.C_in, kD, kH, kW});
    const ttnn::Shape bias_shape({1U, 1U, 1U, c.C_out});
    const ttnn::Shape output_shape({c.N, out[0], out[1], out[2], c.C_out});

    auto step = [&](uint32_t seed) {
        auto input = ttml::autograd::create_tensor(
            make_device_tensor(uniform_vector(input_shape.volume(), -1.F, 1.F, seed), input_shape, c.layout), true);
        auto weight = ttml::autograd::create_tensor(
            make_device_tensor(uniform_vector(weight_shape.volume(), -0.5F, 0.5F, seed + 1), weight_shape, c.layout),
            true);
        auto bias = ttml::autograd::create_tensor(
            make_device_tensor(uniform_vector(c.C_out, -1.F, 1.F, seed + 2), bias_shape, c.layout), true);
        auto result = ttml::ops::conv3d(input, weight, bias, c.stride, c.padding, c.dilation, c.groups);
        result->set_grad(
            make_device_tensor(uniform_vector(output_shape.volume(), -1.F, 1.F, seed + 3), output_shape, c.layout));
        result->backward();
        ttml::autograd::ctx().reset_graph();
    };

    step(31);
    const auto entries_after_first = device->num_program_cache_entries();
    // Guard against a vacuous pass with the cache disabled.
    EXPECT_GT(entries_after_first, 0U);
    step(41);
    EXPECT_EQ(device->num_program_cache_entries(), entries_after_first)
        << "second identical step compiled new programs";
}

TEST_F(Conv3dOpTest, GradientsAccumulateAcrossTwoUses) {
    const Conv3dCase c{.layout = ttnn::Layout::TILE};
    const auto [D, H, W] = c.in_size;
    const auto [kD, kH, kW] = c.kernel;
    const ttnn::Shape input_shape({c.N, D, H, W, c.C_in});
    const ttnn::Shape weight_shape({c.C_out, c.C_in, kD, kH, kW});

    auto input = ttml::autograd::create_tensor(
        make_device_tensor(uniform_vector(input_shape.volume(), -1.F, 1.F, 11), input_shape, c.layout),
        /*requires_grad=*/true);
    auto weight = ttml::autograd::create_tensor(
        make_device_tensor(uniform_vector(weight_shape.volume(), -0.5F, 0.5F, 12), weight_shape, c.layout),
        /*requires_grad=*/true);

    auto single = ttml::ops::conv3d(input, weight);
    single->backward();
    const auto single_input_grad = ttml::core::to_vector(input->get_grad());
    const auto single_weight_grad = ttml::core::to_vector(weight->get_grad());

    input->set_grad(ttnn::Tensor());
    weight->set_grad(ttnn::Tensor());
    ttml::autograd::ctx().reset_graph();

    auto sum = ttml::ops::conv3d(input, weight) + ttml::ops::conv3d(input, weight);
    sum->backward();
    auto doubled_input_grad = ttml::core::to_vector(input->get_grad());
    auto doubled_weight_grad = ttml::core::to_vector(weight->get_grad());

    std::vector<float> expected_input_grad(single_input_grad.size());
    std::vector<float> expected_weight_grad(single_weight_grad.size());
    std::transform(single_input_grad.begin(), single_input_grad.end(), expected_input_grad.begin(), [](float v) {
        return 2.F * v;
    });
    std::transform(single_weight_grad.begin(), single_weight_grad.end(), expected_weight_grad.begin(), [](float v) {
        return 2.F * v;
    });
    expect_close(doubled_input_grad, expected_input_grad, 1e-2F, 2e-2F, "accumulated grad_input");
    expect_close(doubled_weight_grad, expected_weight_grad, 1e-2F, 2e-2F, "accumulated grad_weight");
}

TEST_F(Conv3dOpTest, RejectsUnsupportedArguments) {
    const Conv3dCase c;
    const auto [D, H, W] = c.in_size;
    const auto [kD, kH, kW] = c.kernel;
    const ttnn::Shape input_shape({c.N, D, H, W, c.C_in});
    const ttnn::Shape weight_shape({c.C_out, c.C_in, kD, kH, kW});
    const auto zeros = [](const ttnn::Shape& shape) {
        return ttml::autograd::create_tensor(
            make_device_tensor(std::vector<float>(shape.volume(), 0.F), shape, ttnn::Layout::ROW_MAJOR));
    };
    auto input = zeros(input_shape);
    auto weight = zeros(weight_shape);

    const Dims3 unit{1, 1, 1};
    const Dims3 none{0, 0, 0};

    // groups: zero, in_channels mismatch (weight has C_in/groups == 32 columns so groups=2 needs 64 input
    // channels), and out_channels not divisible by groups
    EXPECT_THROW(
        ttml::ops::conv3d(input, weight, nullptr, /*stride=*/unit, /*padding=*/none, /*dilation=*/unit, /*groups=*/0),
        std::invalid_argument);
    EXPECT_THROW(
        ttml::ops::conv3d(input, weight, nullptr, /*stride=*/unit, /*padding=*/none, /*dilation=*/unit, /*groups=*/2),
        std::invalid_argument);
    EXPECT_THROW(
        ttml::ops::conv3d(
            zeros(ttnn::Shape({c.N, D, H, W, 96U})),
            zeros(weight_shape),
            nullptr,
            /*stride=*/unit,
            /*padding=*/none,
            /*dilation=*/unit,
            /*groups=*/3),
        std::invalid_argument);
    // padding mode
    EXPECT_THROW(
        ttml::ops::conv3d(
            input, weight, nullptr, /*stride=*/unit, /*padding=*/none, /*dilation=*/unit, /*groups=*/1, "replicate"),
        std::invalid_argument);
    // zero stride / dilation
    EXPECT_THROW(ttml::ops::conv3d(input, weight, nullptr, /*stride=*/{0, 1, 1}), std::invalid_argument);
    EXPECT_THROW(
        ttml::ops::conv3d(input, weight, nullptr, /*stride=*/unit, /*padding=*/none, /*dilation=*/{1, 0, 1}),
        std::invalid_argument);
    // dilation 3 makes the effective kernel 7 > D = 4 with no padding
    EXPECT_THROW(
        ttml::ops::conv3d(input, weight, nullptr, /*stride=*/unit, /*padding=*/none, /*dilation=*/{3, 1, 1}),
        std::invalid_argument);
    // rank
    EXPECT_THROW(ttml::ops::conv3d(zeros(ttnn::Shape({D, H, W, c.C_in})), weight), std::invalid_argument);
    EXPECT_THROW(ttml::ops::conv3d(input, zeros(ttnn::Shape({c.C_out, c.C_in, kD, kH}))), std::invalid_argument);
    // channel mismatch
    EXPECT_THROW(ttml::ops::conv3d(input, zeros(ttnn::Shape({c.C_out, 64U, kD, kH, kW}))), std::invalid_argument);
    // dtype mismatch. Float tensors cannot mismatch: autograd reads every float parameter at bf16 (HALF) precision,
    // so a float32 weight is silently cast. A uint32 tensor bypasses autocast and reaches the check.
    {
        auto* device = &ttml::autograd::ctx().get_device();
        auto weight_u32 = ttml::autograd::create_tensor(ttml::core::from_vector<uint32_t, ttnn::DataType::UINT32>(
            std::vector<uint32_t>(weight_shape.volume(), 0U), weight_shape, device, ttnn::Layout::ROW_MAJOR));
        EXPECT_THROW(ttml::ops::conv3d(input, weight_u32), std::invalid_argument);
    }
    // bias shape
    EXPECT_THROW(ttml::ops::conv3d(input, weight, zeros(ttnn::Shape({1U, c.C_out + 1U}))), std::invalid_argument);
    EXPECT_THROW(ttml::ops::conv3d(input, weight, zeros(ttnn::Shape({c.C_out, 1U}))), std::invalid_argument);
}
