#include <cassert>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <iostream>
#include "kernels.h"

// TODO: quickfix
#ifdef LOCAL
#define __global__
#endif

static const bool g_verbose = false;


static __global__ void array_sum(Point<double> *dest_array, const Point<double> *src_array, size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    assert(idx < size);
    dest_array[idx].x += src_array[idx].x;
    dest_array[idx].y += src_array[idx].y;
}

/**
 * Takes a certain point according to current thread index and computes sum of all the forces with other points.
 */
static __global__ void
compute_repulsive(Point<double> *repulsive_forces, const Point<double> *points, size_t points_size,
                  double vertexRepulsion)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t p1_idx = i;

    assert(p1_idx < points_size);

    double accumulator_x = 0;
    double accumulator_y = 0;
    for (size_t j = 0; j < points_size; ++j) {
        if (i == j)
            continue;

        size_t p2_idx = j;
        double dx = points[p1_idx].x - points[p2_idx].x;
        double dy = points[p1_idx].y - points[p2_idx].y;
        double sqLen = dx * dx + dy * dy > (double) 0.0001 ? dx * dx + dy * dy : (double) 0.0001;
        double fact = vertexRepulsion / (sqLen * (double) std::sqrt(sqLen));    // mul factor
        dx *= fact;
        dy *= fact;

        accumulator_x += dx;
        accumulator_y += dy;
    }
    repulsive_forces[p1_idx].x += accumulator_x;
    repulsive_forces[p1_idx].y += accumulator_y;
}

/**
 * Computes compulsive forces for one edge. Updates forces for both end points.
 */
static __global__ void
compute_compulsive(Point<double> *compulsive_forces, const Point<double> *points, const Edge<uint32_t> *edges,
                   size_t edges_size, const uint32_t *lengths, double edgeCompulsion)
{
    size_t edge_idx = blockIdx.x * blockDim.x + threadIdx.x;
    assert(edge_idx < edges_size);
    Edge<uint32_t> edge = edges[edge_idx];

    double dx = points[edge.p2].x - points[edge.p1].x;
    double dy = points[edge.p2].y - points[edge.p1].y;
    double sqLen = dx * dx + dy * dy;
    double fact = (double) std::sqrt(sqLen) * edgeCompulsion / (double) (lengths[edge_idx]);
    dx *= fact;
    dy *= fact;

    atomicAdd(&compulsive_forces[edge.p1].x, dx);
    atomicAdd(&compulsive_forces[edge.p1].y, dy);
    atomicAdd(&compulsive_forces[edge.p2].x, -dx);
    atomicAdd(&compulsive_forces[edge.p2].y, -dy);
}

static __global__ void update_velocities(Point<double> *velocities, const Point<double> *forces, size_t forces_size,
                                         double time_quantum, double vertex_mass, double slowdown)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    assert(i < forces_size);

    double fact = time_quantum / vertex_mass;    // v = Ft/m  => t/m is mul factor for F.
    velocities[i].x = (velocities[i].x + forces[i].x * fact) * slowdown;
    velocities[i].y = (velocities[i].y + forces[i].y * fact) * slowdown;
}

static __global__ void update_point_positions(Point<double> *points, size_t points_size,
                                              const Point<double> *velocities, double time_quantum)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    assert(i < points_size);

    points[i].x += velocities[i].x * time_quantum;
    points[i].y += velocities[i].y * time_quantum;
}

struct kernel_config_t {
    dim3 blocks;
    dim3 threads;
};

static kernel_config_t get_one_dimensional_config(size_t array_size)
{
    assert(array_size % 2 == 0);

    dim3 blocks{1, 1, 1};
    dim3 threads{(unsigned)array_size, 1, 1};
    while (threads.x > 1024) {
        blocks.x *= 2;
        threads.x /= 2;
    }
    return {blocks, threads};
}

static void print_config(const kernel_config_t &config)
{
    std::cout << "blocks_dim=(" << config.blocks.x << "," << config.blocks.y << ","
              << config.blocks.z << "), threads_dim=(" << config.threads.x << "," << config.threads.y << ","
              << config.threads.z
              << ")." << std::endl;
}

static void check_last_kernel_launch()
{
    CUCH(cudaGetLastError());
}

void run_array_sum(Point<double> *dest_array, const Point<double> *src_array, size_t size)
{
    kernel_config_t config = get_one_dimensional_config(size);
    array_sum<<<config.blocks, config.threads>>>(dest_array, src_array, size);
    check_last_kernel_launch();
}

void run_compute_repulsive(Point<double> *repulsive_forces, const Point<double> *points, size_t point_size,
                           double vertexRepulsion)
{
    kernel_config_t config = get_one_dimensional_config(point_size);
    if (g_verbose) {
        std::cout << "Running compute repulsive kernel for:";
        print_config(config);
    }
    compute_repulsive<<<config.blocks, config.threads>>>(repulsive_forces, points, point_size, vertexRepulsion);
    check_last_kernel_launch();
}

void run_compute_compulsive(Point<double> *compulsive_forces, const Point<double> *points, const Edge<uint32_t> *edges,
                            size_t edges_size, const uint32_t *lengths, double edgeCompulsion)
{
    kernel_config_t config = get_one_dimensional_config(edges_size);
    if (g_verbose) {
        std::cout << "Running compute compulsive kernel for:";
        print_config(config);
    }
    compute_compulsive<<<config.blocks, config.threads>>>
        (compulsive_forces, points, edges, edges_size, lengths, edgeCompulsion);
    check_last_kernel_launch();
}

void run_update_velocities(Point<double> *velocities, const Point<double> *forces, size_t forces_size,
                           const ModelParameters<double> &parameters)
{
    kernel_config_t config = get_one_dimensional_config(forces_size);
    if (g_verbose) {
        std::cout << "Running update velocities kernel for:";
        print_config(config);
    }
    update_velocities<<<config.blocks, config.threads>>>(velocities, forces, forces_size, parameters.timeQuantum,
            parameters.vertexMass, parameters.slowdown);
    check_last_kernel_launch();
}

void run_update_point_positions(Point<double> *points, size_t points_size,
                                const Point<double> *velocities, const ModelParameters<double> &params)
{
    kernel_config_t config = get_one_dimensional_config(points_size);
    if (g_verbose) {
        std::cout << "Running update point positions kernel for:";
        print_config(config);
    }
    update_point_positions<<<config.blocks, config.threads>>>(points, points_size, velocities, params.timeQuantum);
    check_last_kernel_launch();
}
