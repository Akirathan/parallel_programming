//
// Created by pal on 14.3.19.
//

#ifndef K_MEANS_TRY_HPP
#define K_MEANS_TRY_HPP

#include <iostream>
#include <vector>
#include <array>

#include <tbb/parallel_for.h>
#include <tbb/mutex.h>

void test_two_parrallel_fors()
{
    int data = 0;
    tbb::mutex mutex;

    for (size_t i = 0; i < 60; ++i) {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, 100), [&](const tbb::blocked_range<size_t> &range) {
            tbb::mutex::scoped_lock lock(mutex);
            data++;
        });

        tbb::parallel_for(tbb::blocked_range<size_t>(0, 100), [&](const tbb::blocked_range<size_t> &range) {
            tbb::mutex::scoped_lock lock(mutex);
            data--;
        });
    }
}

#endif //K_MEANS_TRY_HPP
