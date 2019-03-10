#include <vector>
#include "implementation.hpp"

int main()
{
    const size_t iters = 3;
    const size_t k = 2;

    point_t a{1, 2};
    point_t b{3, 4};
    point_t c{5, 6};
    point_t d{7, 8};
    std::vector<point_t> points{a, b, c, d};

    std::vector<point_t> centroids;
    std::vector<uint8_t> assignments;
    KMeans kMeans;

    kMeans.init(points.size(), k, iters);
    kMeans.compute(points, k, iters, centroids, assignments);
}
