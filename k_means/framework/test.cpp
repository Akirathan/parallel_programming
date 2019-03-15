#include <vector>
#include <iostream>
#include <functional>
#include <cassert>
#include <cstdlib>

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

static std::vector<point_t> generate_random_points(size_t count);
static void sample_test();
static void other_test();
static void rnd_test();
static void big_tests();
static void file_test();
static void _load_file(const std::string &fileName, std::vector<point_t> &res);

static std::vector<std::function<void(void)>> tests {
    sample_test,
    other_test,
    rnd_test,
    //big_tests
    file_test
};

const static bool debugOutput = false;


int main()
{
    //test_two_parrallel_fors();

    for (const auto &test : tests) {
        test();
    }

    std::cerr << "All tests passed" << std::endl;
}

std::vector<point_t> generate_random_points(size_t count)
{
    std::vector<point_t> points;
    for (size_t i = 0; i < count; i++) {
        point_t point{static_cast<int64_t>(rand()), static_cast<int64_t>(rand())};
        points.push_back(point);
    }
    return points;
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
    auto points = generate_random_points(1000);
    run_both_and_compare(5, 50, points);
}

void big_tests()
{
    const size_t iters = 100;
    const size_t k = 240;
    std::vector<point_t> points = generate_random_points(100 * 1024);

    run_both_and_compare(k, iters, points);
}

void file_test()
{
    size_t iters = 4;
    size_t k = 4;
    std::string fileName = "/home/pal/dev/parallel_programming/k_means/data/01-256k";
    std::vector<point_t> points;
    _load_file(fileName, points);

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

/*
 * \bried Load an entire file into a vector of points.
 */
static void _load_file(const std::string &fileName, std::vector<point_t> &res)
{
    // Open the file.
    std::FILE *fp = std::fopen(fileName.c_str(), "rb");
    if (fp == nullptr)
        throw (bpp::RuntimeError() << "File '" << fileName << "' cannot be opened for reading.");

    // Determine length of the file and
    std::fseek(fp, 0, SEEK_END);
    std::size_t count = (std::size_t)(std::ftell(fp) / sizeof(point_t));
    std::fseek(fp, 0, SEEK_SET);
    res.resize(count);

    // Read the entire file.
    std::size_t offset = 0;
    while (offset < count) {
        std::size_t batch = std::min<std::size_t>(count - offset, 1024*1024);
        if (std::fread(&res[offset], sizeof(point_t), batch, fp) != batch)
            throw (bpp::RuntimeError() << "Error while reading from file '" << fileName << "'.");
        offset += batch;
    }

    std::fclose(fp);
}


