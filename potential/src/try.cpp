#include <array>
#include <iostream>
#include <cuda_runtime.h>
#include "kernels.h"

static void array_sum()
{
    const size_t array_size = 32;
    const size_t byte_count = 32 * sizeof(float);
    using array_t = std::array<float, array_size>;

    array_t arr1;
    for (size_t i = 0; i < array_size; ++i) {
        arr1[i] = static_cast<float>(i);
    }
    array_t arr2 = arr1;
    array_t res = {};

    float *cu_arr1;
    float *cu_arr2;
    float *cu_res;
    CUCH(cudaMalloc((void **) &cu_arr1, byte_count));
    CUCH(cudaMalloc((void **) &cu_arr2, byte_count));
    CUCH(cudaMalloc((void **) &cu_res, byte_count));

    CUCH(cudaMemcpy(cu_arr1, arr1.data(), byte_count, cudaMemcpyHostToDevice));
    CUCH(cudaMemcpy(cu_arr2, arr2.data(), byte_count, cudaMemcpyHostToDevice));

    run_array_add(cu_arr1, cu_arr2, cu_res, array_size);

    CUCH(cudaMemcpy(res.data(), cu_res, byte_count, cudaMemcpyDeviceToHost));

    CUCH(cudaFree(cu_arr1));
    CUCH(cudaFree(cu_arr2));
    CUCH(cudaFree(cu_res));

    std::cout << "Result:";
    for (const float &item : res) {
        std::cout << " " << item;
    }
    std::cout << std::endl;
}

int main()
{
    print_device_properties();
}
