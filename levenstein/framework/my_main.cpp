//
// Created by pal on 30.5.19.
//
#include <iostream>
#include <omp.h>
#include "implementation.hpp"

int main()
{
    EditDistance<char, size_t, true> implementation;
    implementation.init(4, 4);
    std::vector<char> array_1 = {'a', 'b', 'c', 'd'};
    std::vector<char> array_2 = {'e', 'f', 'g', 'h'};
    size_t res = implementation.compute(array_1, array_2);
    std::cout << res << std::endl;
}
