#pragma once

#include <cstddef>
#include <string_view>

namespace ml_kernels {

struct KernelSpec {
    std::string_view name;
    std::size_t tile_m;
    std::size_t tile_n;
    std::size_t tile_k;
};

inline constexpr KernelSpec kSmokeSpec{
    "smoke_gemm_like",
    64,
    64,
    32,
};

} // namespace ml_kernels
