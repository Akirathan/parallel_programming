#ifndef CUDA_POTENTIAL_IMPLEMENTATION_HPP
#define CUDA_POTENTIAL_IMPLEMENTATION_HPP

#include "kernels.h"

#include <interface.hpp>
#include <data.hpp>

#include <atomic>
#include <cassert>
#include <cuda_runtime.h>


/*
 * Final implementation of the tested program.
 */
template<typename F = float, typename IDX_T = std::uint32_t, typename LEN_T = std::uint32_t>
class ProgramPotential : public IProgramPotential<F, IDX_T, LEN_T>
{
public:
	typedef F coord_t;		// Type of point coordinates.
	typedef coord_t real_t;	// Type of additional float parameters.
	typedef IDX_T index_t;
	typedef LEN_T length_t;
	typedef Point<coord_t> point_t;
	typedef Edge<index_t> edge_t;

private:
    using Base = IProgramPotential<F, IDX_T, LEN_T>;

    point_t *mCuPoints;
    point_t *mCuForces;
    edge_t *mCuEdges;
    length_t *mCuLengths;

    point_t **mCuTmpRepulsiveForces;
    size_t mCuTmpRepulsiveForcesSize;

public:
    ~ProgramPotential() override
    {
        assert(cudaFree(mCuPoints) == cudaSuccess);
        assert(cudaFree(mCuForces) == cudaSuccess);
        assert(cudaFree(mCuEdges) == cudaSuccess);
        assert(cudaFree(mCuLengths) == cudaSuccess);

        for (size_t i = 0; i < mCuTmpRepulsiveForcesSize; ++i)
            assert(cudaFree(&mCuTmpRepulsiveForces[i]) == cudaSuccess);
        assert(cudaFree(mCuTmpRepulsiveForces) == cudaSuccess);
    }

	void initialize(index_t points, const std::vector<edge_t>& edges, const std::vector<length_t> &lengths, index_t iterations) override
	{
		/*
		 * Initialize your implementation.
		 * Allocate/initialize buffers, transfer initial data to GPU...
		 */
		CUCH(cudaMalloc((void **) &mCuPoints, points * sizeof(point_t)));
        CUCH(cudaMalloc((void **) &mCuForces, points * sizeof(point_t)));
		CUCH(cudaMalloc((void **) &mCuEdges, edges.size() * sizeof(edge_t)));
		CUCH(cudaMalloc((void **) &mCuLengths, lengths.size() * sizeof(length_t)));

		mCuTmpRepulsiveForcesSize = points;
		// Allocate row of pointers (columns).
		CUCH(cudaMalloc((void **) &mCuTmpRepulsiveForces, mCuTmpRepulsiveForcesSize * sizeof(point_t *)));
		for (size_t row_idx = 0; row_idx < mCuTmpRepulsiveForcesSize; row_idx++) {
		    // Allocate inner pointer (one row).
		    CUCH(cudaMalloc(&mCuTmpRepulsiveForces[row_idx], mCuTmpRepulsiveForcesSize * sizeof(point_t)));
		}

        CUCH(cudaMemset(mCuPoints, 0.0, points * sizeof(point_t)));
        CUCH(cudaMemset(mCuForces, 0.0, points * sizeof(point_t)));
		CUCH(cudaMemcpy(mCuEdges, edges.data(), edges.size() * sizeof(edge_t), cudaMemcpyHostToDevice));
		CUCH(cudaMemcpy(mCuLengths, lengths.data(), lengths.size() * sizeof(length_t), cudaMemcpyHostToDevice));
	}


    /**
     * Perform one iteration of the simulation and update positions of the points.
     * @param points These points may be cached on GPU.
     */
	void iteration(std::vector<point_t> &points) override
	{
	    CUCH(cudaMemcpy(mCuPoints, points.data(), points.size() * sizeof(point_t), cudaMemcpyHostToDevice));

		run_compute_repulsive(mCuPoints, points.size(), mCuForces);
	}


    /**
     * Retrieve the velocities buffer from the GPU.
     * This operation is for vreification only and it does not have to be efficient.
     */
	void getVelocities(std::vector<point_t> &velocities) override
	{

	}

private:
    template <typename T>
    void printCudaMatrix(T **cuda_matrix, size_t matrix_size) const
    {
        point_t **tmp_matrix = new point_t*[matrix_size];
        for (size_t i = 0; i < matrix_size; ++i)
            tmp_matrix[i] = new point_t[matrix_size];

        for (size_t row_idx = 0; row_idx < matrix_size; row_idx++) {
            CUCH(cudaMemcpy(&cuda_matrix[row_idx], &tmp_matrix[row_idx], matrix_size * sizeof(point_t),
                            cudaMemcpyDeviceToHost));
        }

        // Print matrix
        for (size_t i = 0; i < matrix_size; i++) {
            for (size_t j = 0; j < matrix_size; j++)
                std::cout << tmp_matrix[i][j] << " ";
            std::cout << std::endl;
        }

        for (size_t i = 0; i < matrix_size; ++i)
            delete[] tmp_matrix[i];
        delete[] tmp_matrix;
    }
};

std::ostream & operator<<(std::ostream &output, const Point<double> &point)
{
    return output << "(" << point.x << "," << point.y << ")";
}


#endif
