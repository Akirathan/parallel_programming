#include <cassert>
#include <cmath>
#include <algorithm>
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

static __global__ void array_add(const float *array_1, const float *array_2, float *dest, size_t size)
{
	size_t idx = threadIdx.x;
	assert(idx < size);
	dest[idx] = array_1[idx] + array_2[idx];
}

static __global__ void print_thread_idx(int *dest, size_t size)
{
	size_t idx = threadIdx.x;
	assert(idx < size);
	dest[idx] = 23;
}

static __global__ void compute_repulsive(const Point<double> *points, Point<double> *forces,
        size_t points_size, double vertexRepulsion)
{
    size_t i = threadIdx.x;
    size_t j = threadIdx.y;
    assert(i < points_size && j < points_size);

    if (i < j) {
        double dx = points[i].x - points[j].x;
        double dy = points[i].y - points[j].y;
        double sqLen = dx*dx + dy*dy > (double)0.0001 ? dx*dx + dy*dy : (double)0.0001;
        double fact = vertexRepulsion / (sqLen * (double)std::sqrt(sqLen));	// mul factor
        dx *= fact;
        dy *= fact;

        /*auto pointI = forces[i].load();
        pointI.x += dx;
        pointI.y += dy;
        forces[i].store(pointI);*/

        //atomicAdd(&forces[i].x, dx);

        /*auto pointJ = forces[j].load();
        pointJ.x -= dx;
        pointJ.y -= dy;
        forces[j].store(pointJ);*/
    }
}

/*
 * This is how a kernel call should be wrapped in a regular function call,
 * so it can be easilly used in cpp-only code.
 */
void run_my_kernel(float *src)
{
	my_kernel<<<64, 64>>>(src);
}

void run_array_add(const float *array_1, const float *array_2, float *dest, size_t size)
{
	assert(dest != nullptr);
	assert(size > 0 && size % 32 == 0);
	array_add<<<1, size>>>(array_1, array_2, dest, size);
}

void run_print_thread_idx(int *dest, size_t size)
{
	print_thread_idx<<<1, size>>>(dest, size);
}

void run_compute_repulsive(const Point<double> *points, size_t point_size, Point<double> *forces,
        double vertexRepulsion)
{
    compute_repulsive<<<1, dim3{point_size, point_size, 1}>>>(points, forces, point_size, vertexRepulsion);
}