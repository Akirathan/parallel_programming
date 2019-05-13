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

/*
 * Sample Kernel
 */
static __global__ void my_kernel(float *src)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	src[idx] += 1.0f;
}

static __global__ void array_sum(Point<double> *dest_array, const Point<double> *src_array, size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    assert(idx < size);
    dest_array[idx].x += src_array[idx].x;
    dest_array[idx].y += src_array[idx].y;
}

static __global__ void compute_repulsive(const Point<double> *points, Point<double> *repulsive_forces,
        size_t points_size, double vertexRepulsion)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    assert(i < points_size && j < points_size);

    if (i < j) {
        std::printf("\t\tKernel computing repulsive forces for i=%d, j=%d\n", i, j);
        double dx = points[i].x - points[j].x;
        double dy = points[i].y - points[j].y;
        double sqLen = dx*dx + dy*dy > (double)0.0001 ? dx*dx + dy*dy : (double)0.0001;
        double fact = vertexRepulsion / (sqLen * (double)std::sqrt(sqLen));	// mul factor
        dx *= fact;
        dy *= fact;

        atomicAdd(&repulsive_forces[i].x, dx);
        atomicAdd(&repulsive_forces[i].y, dy);
        atomicAdd(&repulsive_forces[j].x, -dx);
        atomicAdd(&repulsive_forces[j].y, -dy);
    }
}

static __global__ void compute_compulsive(const Point<double> *points, size_t points_size,
        const Edge<uint32_t> *edges, size_t edges_size,
        const uint32_t *lengths, size_t length_size,
        Point<double> *compulsive_forces, double edgeCompulsion)
{
    size_t edge_idx = blockIdx.x * blockDim.x + threadIdx.x;
    assert(edge_idx < edges_size);
    Edge<uint32_t> edge = edges[edge_idx];
    assert(edge.p1 < points_size && edge.p2 < points_size);

    double dx = points[edge.p2].x - points[edge.p1].x;
    double dy = points[edge.p2].y - points[edge.p1].y;
    double sqLen = dx*dx + dy*dy;
    double fact = (double)std::sqrt(sqLen) * edgeCompulsion / (double)(lengths[edge_idx]);
    dx *= fact;
    dy *= fact;

    atomicAdd(&compulsive_forces[edge.p1].x, dx);
    atomicAdd(&compulsive_forces[edge.p1].y, dy);
    atomicAdd(&compulsive_forces[edge.p2].x, -dx);
    atomicAdd(&compulsive_forces[edge.p2].y, -dy);
}

static __global__ void update_velocities(Point<double> *velocities, const Point<double> *forces, size_t forces_size,
        const ModelParameters<double> &params)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    assert(i < forces_size);

    double fact = params.timeQuantum / params.vertexMass;	// v = Ft/m  => t/m is mul factor for F.
    velocities[i].x = (velocities[i].x + forces[i].x * fact) * params.slowdown;
    velocities[i].y = (velocities[i].y + forces[i].y * fact) * params.slowdown;
}

static __global__ void update_point_positions(Point<double> *points, size_t points_size,
        const Point<double> *velocities, const ModelParameters<double> &params)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    assert(i < points_size);

    points[i].x += velocities[i].x * params.timeQuantum;
    points[i].y += velocities[i].y * params.timeQuantum;
}

struct kernel_config_t {
    dim3 blocks;
    dim3 threads;
};

static kernel_config_t get_one_dimensional_config(size_t array_size)
{
    dim3 blocks{1, 1, 1};
    dim3 threads{(unsigned)array_size, 1, 1};
    while (threads.x > 1024) {
        blocks.x *= 2;
        threads.x /= 2;
    }
    return {blocks, threads};
}

/*
 * This is how a kernel call should be wrapped in a regular function call,
 * so it can be easilly used in cpp-only code.
 */
void run_my_kernel(float *src)
{
	my_kernel<<<64, 64>>>(src);
}

void run_array_sum(Point<double> *dest_array, const Point<double> *src_array, size_t size)
{
    kernel_config_t config = get_one_dimensional_config(size);
    array_sum<<<config.blocks, config.threads>>>(dest_array, src_array, size);
}

void run_compute_repulsive(const Point<double> *points, size_t point_size, Point<double> *repulsive_forces,
        double vertexRepulsion)
{
    assert(point_size % 2 == 0);

    dim3 blocks{1, 1, 1};
    dim3 threads{(unsigned)point_size, (unsigned)point_size, 1};
    while (threads.x * threads.y > 1024) {
        blocks.x *= 2;
        blocks.y *= 2;
        threads.x /= 2; // TODO: integer division?
        threads.y /= 2;
    }

    std::cout << "Running compute repulsive kernel for blocks_dim=(" << blocks.x << "," << blocks.y << ","
              << blocks.z << "), threads_dim=(" << threads.x << "," << threads.y << "," << threads.z
              << ")." << std::endl;
    compute_repulsive<<<blocks, threads>>>(points, repulsive_forces, point_size, vertexRepulsion);

    // Check if kernel was launched properly.
    CUCH(cudaGetLastError());
}

static void print_config(const kernel_config_t &config)
{
    std::cout << "blocks_dim=(" << config.blocks.x << "," << config.blocks.y << ","
              << config.blocks.z << "), threads_dim=(" << config.threads.x << "," << config.threads.y << "," << config.threads.z
              << ")." << std::endl;
}

void run_compute_compulsive(const Point<double> *points, size_t points_size,
                            const Edge<uint32_t> *edges, size_t edges_size,
                            const uint32_t *lengths, size_t lengths_size,
                            Point<double> *compulsive_forces_matrix, double edgeCompulsion)
{
    assert(edges_size % 2 == 0);

    kernel_config_t config = get_one_dimensional_config(edges_size);
    std::cout << "Running compute compulsive kernel for:"; print_config(config);
    compute_compulsive<<<config.blocks, config.threads>>>
        (points, points_size, edges, edges_size, lengths, lengths_size, compulsive_forces_matrix, edgeCompulsion);
}

void run_update_velocities(Point<double> *velocities, const Point<double> *forces, size_t forces_size,
                           const ModelParameters<double> &parameters)
{
    assert(forces_size % 2 == 0);
    kernel_config_t config = get_one_dimensional_config(forces_size);
    std::cout << "Running update velocities kernel for:"; print_config(config);
    update_velocities<<<config.blocks, config.threads>>>(velocities, forces, forces_size, parameters);
}

void run_update_point_positions(Point<double> *points, size_t points_size,
                                const Point<double> *velocities, const ModelParameters<double> &params)
{
    assert(points_size % 2 == 0);
    kernel_config_t config = get_one_dimensional_config(points_size);
    std::cout << "Running update point positions kernel for:"; print_config(config);
    update_point_positions<<<config.blocks, config.threads>>>(points, points_size, velocities, params);
}
