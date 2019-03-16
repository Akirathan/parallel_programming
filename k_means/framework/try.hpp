//
// Created by pal on 14.3.19.
//

#ifndef K_MEANS_TRY_HPP
#define K_MEANS_TRY_HPP

#include <cstdlib>
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

bool operator<(const point_t &i, const point_t& j)
{
    return i.x <= j.x && i.y < j.y;
}

size_t operator-(const point_t &i, const point_t& j)
{
    return static_cast<size_t>(std::abs((i.x - j.x) + (i.y - j.y)));
}

point_t operator+(const point_t &i, size_t k)
{
    int64_t newX = i.x + k;
    int64_t newY = i.y;
    return {newX, newY};
}

void test_custom_blocked_range()
{
    tbb::parallel_for(tbb::blocked_range<point_t>({0, 0}, {2, 2}), [&](const tbb::blocked_range<point_t> &range) {
        std::cout << "range.begin.x=" << range.begin().x << "range.begin.y=" << range.begin().y << std::endl;
        std::cout << "range.end.x=" << range.end().x << "range.end.y=" << range.end().y << std::endl;
    });
}

#endif //K_MEANS_TRY_HPP
