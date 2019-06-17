//
// Created by pal on 30.5.19.
//
#include <iostream>
#include <omp.h>
#include "implementation.hpp"
#include "../serial/implementation.hpp"

static size_t compute_serial(const std::vector<char> &array_1, const std::vector<char> &array_2)
{
    SerialEditDistance<char, size_t, false> serial_impl;
    serial_impl.init(array_1.size(), array_2.size());
    return serial_impl.compute(array_1, array_2);
}

int main()
{
    EditDistance<char, size_t, true> implementation;
    implementation.init(4, 4);
    std::vector<char> array_1 = {'a', 'b', 'c', 'd'};
    std::vector<char> array_2 = {'e', 'f', 'g', 'h'};
    size_t res = implementation.compute(array_1, array_2);
    size_t serial_res = compute_serial(array_1, array_2);
    std::cout << "My Result: " << res << ", Serial result: " << serial_res << std::endl;
}
