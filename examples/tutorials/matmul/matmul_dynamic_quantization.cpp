/*******************************************************************************
* Copyright 2026 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/// @example matmul_dynamic_quantization.cpp
/// > Annotated version: @ref matmul_dynamic_quantization_cpp

/// @page matmul_dynamic_quantization_cpp_brief
/// @brief C++ API example demonstrating MatMul dynamic quantization with
/// run-time source and weights quantization parameters.

/// @page matmul_dynamic_quantization_cpp MatMul Tutorial: Dynamic Quantization
/// \copybrief matmul_dynamic_quantization_cpp_brief
///
/// Concepts:
/// - Dynamic quantization parameters provided at execution time
///   - Source grouped scales:
///     dnnl::primitive_attr::set_scales(DNNL_ARG_SRC, ...)
///   - Weights grouped scales:
///     dnnl::primitive_attr::set_scales(DNNL_ARG_WEIGHTS, ...)
/// - Signed/unsigned quantization and zero-point relationship
///   - General affine quantization:
///     q = clamp(round(x / s) + zp, qmin, qmax)
///   - General dequantization:
///     x_hat = (q - zp) * s
///   - Signed integer types (s8/s4): usually symmetric around 0, so zp = 0
///   - Unsigned integer types (u8/u4): usually affine, zp often in [qmin, qmax]
///     to map real zero into integer domain (q = zp when x = 0)
/// - Create primitive once, execute multiple times with different parameters
/// - Weights pre-packing via memory::format_tag::any
///
/// Computation in this example:
/// \f[
/// C_{dst} = (A_{s8} - zp_A[:]) * scale_A[:] * (B_{wq} - zp_B[:]) * scale_B[:]
/// \f]
///
/// where source and weights quantization parameters are grouped over the K
/// dimension, and weights are additionally per-output-channel (N). This file
/// iterates over 4 cases: s8:s8:f16, s8:s4:f16, s8:u8:f16, and s8:u4:f32.
///
/// Note: this tutorial's current 4 cases run with zp_A = 0 and zp_B = 0.
/// For signed tensors this is the common symmetric setup; for unsigned tensors
/// this means we intentionally quantize non-negative values in this tutorial.
///
/// @include matmul_dynamic_quantization.cpp

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <stdexcept>
#include <vector>

#include "oneapi/dnnl/dnnl.hpp"

#include "example_utils.hpp"

using namespace dnnl;

namespace {

int number_of_runs = 1;

struct case_config_t {
    std::string name;
    memory::data_type wei_dt;
    memory::data_type dst_dt;
    bool src_common_scale;
};

int qmin(memory::data_type dt) {
    switch (dt) {
        case memory::data_type::s8: return -128;
        case memory::data_type::s4: return -8;
        case memory::data_type::u8: return 0;
        case memory::data_type::u4: return 0;
        default: throw std::runtime_error("Unsupported weight data type");
    }
}

int qmax(memory::data_type dt) {
    switch (dt) {
        case memory::data_type::s8: return 127;
        case memory::data_type::s4: return 7;
        case memory::data_type::u8: return 255;
        case memory::data_type::u4: return 15;
        default: throw std::runtime_error("Unsupported weight data type");
    }
}

bool is_unsigned(memory::data_type dt) {
    return dt == memory::data_type::u8 || dt == memory::data_type::u4;
}

void init_f32(std::vector<float> &v, float lo, float hi) {
    std::mt19937 gen;
    std::uniform_real_distribution<float> u(lo, hi);
    for (auto &e : v)
        e = u(gen);
}

template <typename T>
T clamp_to_int(float x) {
    const float lo = static_cast<float>(std::numeric_limits<T>::lowest());
    const float hi = static_cast<float>(std::numeric_limits<T>::max());
    x = std::min(std::max(x, lo), hi);
    return static_cast<T>(std::nearbyint(x));
}

int clamp_int(int x, int lo, int hi) {
    return std::min(std::max(x, lo), hi);
}

void reference_grouped_qdq_matmul(int64_t M, int64_t N, int64_t K,
        int64_t src_group_size, int64_t wei_group_size, bool src_common_scale,
        const std::vector<int8_t> &src_s8, const std::vector<int32_t> &wei_q,
        const std::vector<float> &src_scales, const std::vector<float> &wei_scales,
        std::vector<float> &C) {
    const int64_t wei_num_groups = (K + wei_group_size - 1) / wei_group_size;
    const int64_t src_num_groups = (K + src_group_size - 1) / src_group_size;
    (void)wei_num_groups;

    C.assign(M * N, 0.0f);
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                const int64_t src_g = k / src_group_size;
                const int64_t wei_g = k / wei_group_size;
                const int64_t wei_idx = wei_g * N + n;
                const int64_t src_idx
                        = src_common_scale ? 0 : (m * src_num_groups + src_g);

                // General dequantization form:
                // a_fp32 = (a_q - zp_a) * s_src
                // b_fp32 = (b_q - zp_b) * s_wei
                // Signed q commonly uses zp = 0 (symmetric).
                // Unsigned q commonly uses affine zp to represent x = 0.
                // This tutorial currently sets zp_a = zp_b = 0.
                const float a = static_cast<float>(src_s8[m * K + k])
                        * src_scales[src_idx];
                const float b = static_cast<float>(wei_q[k * N + n])
                        * wei_scales[wei_idx];
                acc += a * b;
            }
            C[m * N + n] = acc;
        }
    }
}

void quantize_src_grouped_s8(const std::vector<float> &src_f32, int64_t M,
        int64_t K, int64_t group_size, bool src_common_scale,
        std::vector<float> &src_scales, std::vector<int8_t> &src_s8) {
    const int64_t num_groups = (K + group_size - 1) / group_size;
    src_s8.resize(M * K);

    if (src_common_scale) {
        // General affine quantization form:
        // q = clamp(round(x / s) + zp, qmin, qmax)
        // For signed source (s8), symmetric quantization usually sets zp = 0.
        // This tutorial uses that setup:
        // s = max(abs(x)) / 127, q = round(x / s)
        float abs_max = 0.0f;
        for (const auto v : src_f32)
            abs_max = std::max(abs_max, std::abs(v));
        const float s = std::max(abs_max / 127.0f, 1e-8f);
        src_scales.assign(1, s);
        for (int64_t i = 0; i < M * K; ++i) {
            src_s8[i] = clamp_to_int<int8_t>(src_f32[i] / s);
        }
        return;
    }

    src_scales.assign(M * num_groups, 1.0f);

    for (int64_t m = 0; m < M; ++m) {
        for (int64_t g = 0; g < num_groups; ++g) {
            const int64_t k_begin = g * group_size;
            const int64_t k_end = std::min(k_begin + group_size, K);
            const int64_t idx = m * num_groups + g;

            std::vector<float> group_vals;
            group_vals.reserve(k_end - k_begin);
            for (int64_t k = k_begin; k < k_end; ++k) {
                group_vals.push_back(src_f32[m * K + k]);
            }

            // Group-wise affine quantization form on K chunks:
            // q(m,k) = clamp(round(x(m,k) / s(m,g(k))) + zp(m,g(k)), qmin, qmax)
            // This tutorial currently uses zp(m,g) = 0, so it becomes:
            // s(m,g) = max(abs(x(m,g,:))) / 127
            // q(m,k) = round(x(m,k) / s(m,g(k)))
            float abs_max = 0.0f;
            for (const auto v : group_vals)
                abs_max = std::max(abs_max, std::abs(v));
            src_scales[idx] = std::max(abs_max / 127.0f, 1e-8f);

            for (int64_t k = k_begin; k < k_end; ++k) {
                const float q = src_f32[m * K + k] / src_scales[idx];
                src_s8[m * K + k] = clamp_to_int<int8_t>(q);
            }
        }
    }
}

void quantize_wei_grouped(const std::vector<float> &wei_f32, int64_t K,
        int64_t N, int64_t group_size, memory::data_type wei_dt,
        std::vector<float> &wei_scales, std::vector<int32_t> &wei_q) {
    const int64_t num_groups = (K + group_size - 1) / group_size;
    wei_scales.assign(num_groups * N, 1.0f);
    wei_q.resize(K * N);

    const int q_lo = qmin(wei_dt);
    const int q_hi = qmax(wei_dt);

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t g = 0; g < num_groups; ++g) {
            const int64_t k_begin = g * group_size;
            const int64_t k_end = std::min(k_begin + group_size, K);

            std::vector<float> block;
            block.reserve(k_end - k_begin);
            for (int64_t k = k_begin; k < k_end; ++k)
                block.push_back(wei_f32[k * N + n]);

            const int64_t idx = g * N + n;

            if (is_unsigned(wei_dt)) {
                // Unsigned integer quantization (u8/u4):
                // general affine form q = clamp(round(x / s) + zp, 0, qmax).
                // In many pipelines, zp is chosen so x = 0 maps to q = zp.
                // Unsigned path in this tutorial uses zp = 0, q in [0, qmax]:
                // s = max(x) / qmax, q = round(x / s)
                float max_val = 0.0f;
                for (const auto v : block)
                    max_val = std::max(max_val, v);
                wei_scales[idx] = std::max(max_val / static_cast<float>(q_hi), 1e-8f);
            } else {
                // Signed integer quantization (s8/s4):
                // commonly symmetric with zp = 0.
                // This tutorial uses: s = max(abs(x)) / qmax, q = round(x / s)
                float abs_max = 0.0f;
                for (const auto v : block)
                    abs_max = std::max(abs_max, std::abs(v));
                wei_scales[idx]
                        = std::max(abs_max / static_cast<float>(q_hi), 1e-8f);
            }

            for (int64_t k = k_begin; k < k_end; ++k) {
                const int q = static_cast<int>(std::nearbyint(
                        wei_f32[k * N + n] / wei_scales[idx]));
                wei_q[k * N + n] = clamp_int(q, q_lo, q_hi);
            }
        }
    }
}

void write_wei_to_memory(const std::vector<int32_t> &wei_q, int64_t K, int64_t N,
        memory::data_type wei_dt, memory &wei_m) {
    if (wei_dt == memory::data_type::s8) {
        std::vector<int8_t> wei_s8(K * N);
        for (int64_t i = 0; i < K * N; ++i)
            wei_s8[i] = static_cast<int8_t>(wei_q[i]);
        write_to_dnnl_memory(wei_s8.data(), wei_m);
        return;
    }

    if (wei_dt == memory::data_type::u8) {
        std::vector<uint8_t> wei_u8(K * N);
        for (int64_t i = 0; i < K * N; ++i)
            wei_u8[i] = static_cast<uint8_t>(wei_q[i]);
        write_to_dnnl_memory(wei_u8.data(), wei_m);
        return;
    }

    const int64_t nelems = K * N;
    std::vector<uint8_t> packed((nelems + 1) / 2, 0);
    for (int64_t i = 0; i < nelems; ++i) {
        uint8_t nibble = 0;
        if (wei_dt == memory::data_type::u4) {
            nibble = static_cast<uint8_t>(wei_q[i] & 0x0F);
        } else {
            nibble = static_cast<uint8_t>(static_cast<int8_t>(wei_q[i])) & 0x0F;
        }
        if ((i & 1) == 0) {
            packed[i / 2] |= nibble;
        } else {
            packed[i / 2] |= static_cast<uint8_t>(nibble << 4);
        }
    }
    write_to_dnnl_memory(packed.data(), wei_m);
}

matmul::primitive_desc create_matmul_pd(int64_t M, int64_t N, int64_t K,
        int64_t src_group_size, int64_t wei_group_size, const case_config_t &cfg,
        const engine &eng) {
    memory::desc src_md({M, K}, memory::data_type::s8, {K, 1});
    const auto wei_tag
            = (cfg.wei_dt == memory::data_type::s4
                      || cfg.wei_dt == memory::data_type::u4)
            ? memory::format_tag::ab
            : memory::format_tag::any;
    memory::desc wei_md({K, N}, cfg.wei_dt, wei_tag);
    memory::desc dst_md({M, N}, cfg.dst_dt, {N, 1});

    primitive_attr attr;
    if (cfg.src_common_scale) {
        attr.set_scales(DNNL_ARG_SRC, /* mask */ 0, {}, memory::data_type::f32);
    } else {
        attr.set_scales(DNNL_ARG_SRC, /* mask */ 1 << 1, {1, src_group_size},
                memory::data_type::f32);
    }
    attr.set_scales(DNNL_ARG_WEIGHTS, /* mask */ (1 << 0) | (1 << 1),
            {wei_group_size, 1}, memory::data_type::f32);

    return matmul::primitive_desc(eng, src_md, wei_md, dst_md, attr);
}

void compare_outputs(const std::vector<float> &ref, memory &dst_f32_m) {
    std::vector<float> dst(ref.size());
    read_from_dnnl_memory(dst.data(), dst_f32_m);

    double ref_l2 = 0.0;
    double diff_l2 = 0.0;
    double max_abs_diff = 0.0;
    size_t non_finite_count = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const auto v = dst[i];
        if (!std::isfinite(v)) {
            non_finite_count++;
            continue;
        }
        const double r = ref[i];
        const double d = static_cast<double>(v) - r;
        ref_l2 += r * r;
        diff_l2 += d * d;
        max_abs_diff = std::max(max_abs_diff, std::abs(d));
    }

    const double rel_l2 = std::sqrt(diff_l2) / std::max(std::sqrt(ref_l2), 1e-12);
    std::cout << "Output comparison (vs grouped QDQ reference): rel_l2="
              << rel_l2 << ", max_abs_diff=" << max_abs_diff
              << ", non_finite=" << non_finite_count << std::endl;

    const size_t n_show = std::min<size_t>(10, ref.size());
    std::cout << "\nFirst " << n_show << " elements (ref vs dst):\n";
    std::cout << std::left << std::setw(8) << "index" << std::setw(18)
              << "ref" << std::setw(18) << "dst" << "abs_diff"
              << std::endl;
    for (size_t i = 0; i < n_show; ++i) {
        const double abs_diff
                = std::abs(static_cast<double>(dst[i]) - static_cast<double>(ref[i]));
        std::cout << std::left << std::setw(8) << i << std::setw(18)
                  << ref[i] << std::setw(18) << dst[i] << abs_diff << std::endl;
    }

    const size_t n_tail = std::min<size_t>(10, ref.size());
    const size_t tail_start = ref.size() - n_tail;
    std::cout << "\nLast " << n_tail << " elements (ref vs dst):\n";
    std::cout << std::left << std::setw(8) << "index" << std::setw(18)
              << "ref" << std::setw(18) << "dst" << "abs_diff"
              << std::endl;
    for (size_t i = tail_start; i < ref.size(); ++i) {
        const double abs_diff
                = std::abs(static_cast<double>(dst[i]) - static_cast<double>(ref[i]));
        std::cout << std::left << std::setw(8) << i << std::setw(18)
                  << ref[i] << std::setw(18) << dst[i] << abs_diff << std::endl;
    }
}

void run_case(const case_config_t &cfg, const engine &eng) {
    const int64_t M = 64;
    const int64_t K = 128;
    const int64_t N = 96;
    const int64_t src_group_size = 64;
    const int64_t wei_group_size = 64;
    const int64_t src_num_groups = (K + src_group_size - 1) / src_group_size;
    const int64_t wei_num_groups = (K + wei_group_size - 1) / wei_group_size;

    std::cout << "\n=== Running case: " << cfg.name << " ===" << std::endl;

    std::vector<float> src_f32(M * K);
    std::vector<float> wei_f32(K * N);
    // Use wider ranges to make quantization effects easier to observe.
    init_f32(src_f32, -8.0f, 8.0f);
    if (is_unsigned(cfg.wei_dt)) {
        init_f32(wei_f32, 0.0f, 6.0f);
    } else {
        init_f32(wei_f32, -6.0f, 6.0f);
    }

    std::vector<float> src_scales;
    std::vector<int8_t> src_s8;
    quantize_src_grouped_s8(src_f32, M, K, src_group_size, cfg.src_common_scale,
            src_scales, src_s8);

    std::vector<float> wei_scales;
    std::vector<int32_t> wei_q;
    quantize_wei_grouped(wei_f32, K, N, wei_group_size, cfg.wei_dt, wei_scales,
            wei_q);

    std::vector<float> ref_qdq_f32;
    reference_grouped_qdq_matmul(M, N, K, src_group_size, wei_group_size,
            cfg.src_common_scale, src_s8, wei_q, src_scales, wei_scales,
            ref_qdq_f32);

    auto matmul_pd = create_matmul_pd(
            M, N, K, src_group_size, wei_group_size, cfg, eng);

    memory src_s8_m({{M, K}, memory::data_type::s8, {K, 1}}, eng);
    write_to_dnnl_memory(src_s8.data(), src_s8_m);

    memory wei_plain_m({{K, N}, cfg.wei_dt, memory::format_tag::ab}, eng);
    write_wei_to_memory(wei_q, K, N, cfg.wei_dt, wei_plain_m);

    memory wei_exec_m = wei_plain_m;
    if (matmul_pd.weights_desc() != wei_plain_m.get_desc()) {
        memory wei_packed_m(matmul_pd.weights_desc(), eng);
        stream rs(eng);
        reorder(wei_plain_m, wei_packed_m).execute(rs, wei_plain_m, wei_packed_m);
        rs.wait();
        wei_exec_m = wei_packed_m;
    }

    memory src_scale_m = cfg.src_common_scale
            ? memory({{1}, memory::data_type::f32, {1}}, eng)
            : memory({{M, src_num_groups}, memory::data_type::f32,
                              {src_num_groups, 1}},
                    eng);
    write_to_dnnl_memory(src_scales.data(), src_scale_m);

    memory wei_scales_m({{N, wei_num_groups}, memory::data_type::f32, {1, N}},
            eng);
    write_to_dnnl_memory(wei_scales.data(), wei_scales_m);

    memory dst_m({{M, N}, cfg.dst_dt, {N, 1}}, eng);

    matmul matmul_p(matmul_pd);
    stream s(eng);
    for (int run = 0; run < number_of_runs; ++run) {
        matmul_p.execute(s,
                {{DNNL_ARG_SRC, src_s8_m},
                        {DNNL_ARG_WEIGHTS, wei_exec_m},
                        {DNNL_ARG_DST, dst_m},
                        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, src_scale_m},
                        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS,
                                wei_scales_m}});
    }
    s.wait();

    memory dst_f32_m({{M, N}, memory::data_type::f32, {N, 1}}, eng);
    // Keep a single f32 comparison path for all destination dtypes.
    reorder(dst_m, dst_f32_m).execute(s, dst_m, dst_f32_m);
    s.wait();

    compare_outputs(ref_qdq_f32, dst_f32_m);
}

} // namespace

void matmul_dynamic_quantization(engine::kind engine_kind) {
    engine eng(engine_kind, 0);

    const std::vector<case_config_t> cases {
            {"s8:s8:f16", memory::data_type::s8, memory::data_type::f16, true},
            {"s8:s4:f16", memory::data_type::s4, memory::data_type::f16, false},
            {"s8:u8:f16", memory::data_type::u8, memory::data_type::f16, false},
            {"s8:u4:f32", memory::data_type::u4, memory::data_type::f32, false},
    };

    for (const auto &cfg : cases) {
        run_case(cfg, eng);
    }
}

int main(int argc, char **argv) {
    engine::kind engine_kind = parse_engine_kind(argc, argv);
    // This tutorial targets CPU dynamic quantization path.
    if (engine_kind != engine::kind::cpu) return 0;
    return handle_example_errors(matmul_dynamic_quantization, engine_kind);
}
