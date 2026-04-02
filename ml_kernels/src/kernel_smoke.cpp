#include <iostream>

#include "ml_kernels/kernel_common.h"

int main() {
    const auto &spec = ml_kernels::kSmokeSpec;
    std::cout << "ml_kernels workspace ready\n";
    std::cout << "kernel=" << spec.name
              << " tile=(" << spec.tile_m
              << "," << spec.tile_n
              << "," << spec.tile_k << ")\n";
    return 0;
}
