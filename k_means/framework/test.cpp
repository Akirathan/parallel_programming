#include <vector>
#include <iostream>
#include <functional>
#include <cassert>
#include "implementation.hpp"
#include "../serial/implementation.hpp"
#include "try.hpp"

static void run_parallel(size_t k, size_t iters, const std::vector<point_t> &points, std::vector<point_t> &centroids,
                         std::vector<uint8_t> &assignments);

static void run_serial(size_t k, size_t iters, const std::vector<point_t> &points, std::vector<point_t> &centroids,
                       std::vector<uint8_t> &assignments);

static void run_both_and_compare(size_t k, size_t iters, const std::vector<point_t> &points);

void compare_centroids(const std::vector<point_t> &centroidsA, const std::vector<point_t> &centroidsB);
void compare_assignments(const std::vector<uint8_t> &assignmentsA, const std::vector<uint8_t> &assignmentsB);

static void sample_test();
static void other_test();
static void rnd_test();

static std::vector<std::function<void(void)>> tests {
    sample_test,
    other_test,
    rnd_test
};

const static bool debugOutput = false;


int main()
{
    //test_two_parrallel_fors();

    for (size_t i = 0; i < 100; ++i) {
        for (const auto &test : tests) {
            test();
        }
    }

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
    run_parallel(k, iters, points, centroids, assignments);

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

static void rnd_test()
{
    const size_t iters = 5;
    const size_t k = 2;
    std::vector<point_t> points;
    for (size_t i = 0; i < 100; i++) {
        point_t point{static_cast<int64_t>(i), static_cast<int64_t>(i+1)};
        points.push_back(point);
    }

    run_both_and_compare(k, iters, points);
}

void run_parallel(size_t k, size_t iters, const std::vector<point_t> &points, std::vector<point_t> &centroids,
                  std::vector<uint8_t> &assignments)
{
    KMeans<point_t, uint8_t, debugOutput> kMeans;
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

    compare_centroids(parallelCentroids, serialCentroids);
    compare_assignments(parallelAssignments, serialAssignments);
}

bool operator==(const point_t &pointA, const point_t &pointB)
{
    return pointA.x == pointB.x && pointA.y == pointB.y;
}

std::ostream &operator<<(std::ostream &os, const point_t &point)
{
    os << "x: " << point.x << " y: " << point.y;
    return os;
}


void compare_centroids(const std::vector<point_t> &centroidsA, const std::vector<point_t> &centroidsB)
{
    assert(centroidsA.size() == centroidsB.size());

    for (size_t i = 0; i < centroidsA.size(); i++) {
        if (!(centroidsA[i] == centroidsB[i])) {
            std::cerr << "Centroids differ - expected:" << centroidsA[i] << " got:" << centroidsB[i] << std::endl;
            assert(false);
        }
    }
}

void compare_assignments(const std::vector<uint8_t> &assignmentsA, const std::vector<uint8_t> &assignmentsB)
{
    assert(assignmentsA.size() == assignmentsB.size());

    for (size_t i = 0; i < assignmentsA.size(); i++) {
        if (assignmentsA[i] != assignmentsB[i]) {
            std::cerr << "Assignments differ - expected:" << assignmentsA[i] << " got:" << assignmentsB[i] << std::endl;
            assert(false);
        }
    }
}

