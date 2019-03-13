#include <vector>
#include <iostream>
#include <cassert>
#include "implementation.hpp"
#include "../serial/implementation.hpp"

static void run_parallel(size_t k, size_t iters, const std::vector<point_t> &points, std::vector<point_t> &centroids,
                         std::vector<uint8_t> &assignments);

static void run_serial(size_t k, size_t iters, const std::vector<point_t> &points, std::vector<point_t> &centroids,
                       std::vector<uint8_t> &assignments);

static void run_both_and_compare(size_t k, size_t iters, const std::vector<point_t> &points);

template<typename T> void compare(const std::vector<T> &vectorA, const std::vector<T> &vectorB);

static void sample_test();
static void other_test();


int main()
{
    sample_test();
    other_test();
    std::cerr << "All tests passed" << std::endl;
}

void sample_test()
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
    run_serial(k, iters, points, centroids, assignments);

    // [a], [b,c,d]
    assert(assignments[0] != assignments[1]);
    assert(assignments[1] == assignments[2] && assignments[2] == assignments[3]);
    assert(centroids[0].x == 1 && centroids[0].y == 2);
    assert(centroids[1].x == 5 && centroids[1].y == 6);
}

void other_test()
{
    const size_t iters = 2;
    const size_t k = 2;
    point_t a{1, 2};
    point_t b{3, 4};
    point_t c{5, 6};
    point_t d{7, 8};
    std::vector<point_t> points{a, b, c, d};

    run_both_and_compare(k, iters, points);
}

void run_parallel(size_t k, size_t iters, const std::vector<point_t> &points, std::vector<point_t> &centroids,
                  std::vector<uint8_t> &assignments)
{
    KMeans<point_t, uint8_t, true> kMeans;
    kMeans.init(points.size(), k, iters);
    kMeans.compute(points, k, iters, centroids, assignments);
}

void run_serial(size_t k, size_t iters, const std::vector<point_t> &points, std::vector<point_t> &centroids,
                std::vector<uint8_t> &assignments)
{
    SerialKMeans serialKMeans;
    serialKMeans.init(points.size(), k, iters);
    serialKMeans.compute(points, k, iters, centroids, assignments);
}

void run_both_and_compare(size_t k, size_t iters, const std::vector<point_t> &points)
{
    std::vector<point_t> parallelCentroids;
    std::vector<point_t> serialCentroids;
    std::vector<uint8_t> parallelAssignments;
    std::vector<uint8_t> serialAssignments;

    run_parallel(k, iters, points, parallelCentroids, parallelAssignments);
    run_serial(k, iters, points, serialCentroids, serialAssignments);

    compare(parallelCentroids, serialCentroids);
    compare(parallelAssignments, serialAssignments);
}

bool operator==(const point_t &pointA, const point_t &pointB)
{
    return pointA.x == pointB.x && pointA.y == pointB.y;
}

template<typename T>
void compare(const std::vector<T> &vectorA, const std::vector<T> &vectorB)
{
    assert(vectorA.size() == vectorB.size());

    for (size_t i = 0; i < vectorA.size(); i++) {
        assert(vectorA[i] == vectorB[i]);
    }
}

