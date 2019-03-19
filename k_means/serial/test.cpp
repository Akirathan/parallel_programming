#include <iostream>
#include <vector>
#include <cassert>
#include "implementation.hpp"

static void sample_test()
{
    SerialKMeans kmeans;
    std::vector<point_t> points;
    const size_t k = 2;
    const size_t iters = 3;

    point_t a{2, 6};
    point_t b{3, 5};
    point_t c{4, 12};
    point_t d{8, 5};
    point_t e{9, 12};
    points.push_back(a);
    points.push_back(b);
    points.push_back(c);
    points.push_back(d);
    points.push_back(e);

    std::vector<point_t> centroids;
    centroids.reserve(k);

    std::vector<uint8_t> assignments;
    assignments.reserve(points.size());

    kmeans.init(points.size(), k, iters);
    kmeans.compute(points, k, iters, centroids, assignments);

    assert(assignments[0] == assignments[1] && assignments[1] == assignments[3]);
    assert(assignments[2] == assignments[4]);

    std::cout << "The end" << std::endl;
}

int main()
{
    sample_test();
}