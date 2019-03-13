#include <vector>
#include <cassert>
#include "implementation.hpp"

int main()
{
    const size_t iters = 1;
    const size_t k = 2;

    point_t a{1, 2};
    point_t b{3, 4};
    point_t c{5, 6};
    point_t d{7, 8};
    std::vector<point_t> points{a, b, c, d};

    std::vector<point_t> centroids;
    std::vector<uint8_t> assignments;
    KMeans<point_t, uint8_t, true> kMeans;

    kMeans.init(points.size(), k, iters);
    kMeans.compute(points, k, iters, centroids, assignments);

    // [a], [b,c,d]
    assert(assignments[0] != assignments[1]);
    assert(assignments[1] == assignments[2] && assignments[2] == assignments[3]);
    assert(centroids[0].x == 1 && centroids[0].y == 2);
    assert(centroids[1].x == 5 && centroids[1].y == 6);
}
