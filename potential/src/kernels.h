#ifndef CUDA_POTENTIAL_IMPLEMENTATION_KERNELS_H
#define CUDA_POTENTIAL_IMPLEMENTATION_KERNELS_H

#include <cuda_runtime.h>
#include <atomic>
#include <stdexcept>
#include <sstream>
#include <cstdint>
#include <iostream>

#include "data.hpp"


/**
 * A stream exception that is base for all runtime errors.
 */
class CudaError : public std::exception
{
protected:
	std::string mMessage;	///< Internal buffer where the message is kept.
	cudaError_t mStatus;

public:
	CudaError(cudaError_t status = cudaSuccess) : std::exception(), mStatus(status) {}
	CudaError(const char *msg, cudaError_t status = cudaSuccess) : std::exception(), mMessage(msg), mStatus(status) {}
	CudaError(const std::string &msg, cudaError_t status = cudaSuccess) : std::exception(), mMessage(msg), mStatus(status) {}
	virtual ~CudaError() throw() {}

	virtual const char* what() const throw()
	{
		return mMessage.c_str();
	}

	// Overloading << operator that uses stringstream to append data to mMessage.
	template<typename T>
	CudaError& operator<<(const T &data)
	{
		std::stringstream stream;
		stream << mMessage << data;
		mMessage = stream.str();
		return *this;
	}
};


/**
 * CUDA error code check. This is internal function used by CUCH macro.
 */
inline void _cuda_check(cudaError_t status, int line, const char *srcFile, const char *errMsg = NULL)
{
	if (status != cudaSuccess) {
		throw (CudaError(status) << "CUDA Error (" << status << "): " << cudaGetErrorString(status) << "\n"
			<< "at " << srcFile << "[" << line << "]: " << errMsg);
	}
}

/**
 * Macro wrapper for CUDA calls checking.
 */
#define CUCH(status) _cuda_check(status, __LINE__, __FILE__, #status)


inline void print_device_properties()
{
    cudaDeviceProp properties{};
    CUCH(cudaGetDeviceProperties(&properties, 0));

    std::cout << "Device properties:" << std::endl;
    std::cout << "\tName: " << properties.name << std::endl;
    std::cout << "\tCompute capability: " << properties.major << "." << properties.minor << std::endl;
    std::cout << "\tMax threads per block: " << properties.maxThreadsPerBlock << std::endl;
    std::cout << "\tIntegrated: " << properties.integrated << std::endl;
}

/*
 * Kernel wrapper declarations.
 */

void run_my_kernel(float *src);
void run_array_add(const float *array_1, const float *array_2, float *dest, size_t size);
void run_print_thread_idx(int *dest, size_t size);
void run_compute_repulsive(const Point<double> *points, size_t point_size, Point<double> *repulsive_forces_matrix,
        double vertexRepulsion);
void run_compute_compulsive(const Point<double> *points, size_t points_size,
        const Edge<uint32_t> *edges, size_t edges_size,
        const uint32_t *lengths, size_t length_size,
        Point<double> *compulsive_forces_matrix, double edgeCompulsion);



#endif
