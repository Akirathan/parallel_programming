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
    size_t mPointsSize;
    point_t *mCuForces;
    point_t *mCuVelocities;
    point_t *mCuRepulsiveForces;
    point_t *mCuCompulsiveForces;
    edge_t *mCuEdges;
    size_t mEdgesSize;
    length_t *mCuLengths;
    size_t mLengthsSize;
    bool mFirstIteration = true;

public:
    ~ProgramPotential() override
    {
        assert(cudaFree(mCuPoints) == cudaSuccess);
        assert(cudaFree(mCuForces) == cudaSuccess);
        assert(cudaFree(mCuVelocities) == cudaSuccess);
        assert(cudaFree(mCuEdges) == cudaSuccess);
        assert(cudaFree(mCuLengths) == cudaSuccess);
        assert(cudaFree(mCuRepulsiveForces) == cudaSuccess);
        assert(cudaFree(mCuCompulsiveForces) == cudaSuccess);
    }

	void initialize(index_t points, const std::vector<edge_t>& edges, const std::vector<length_t> &lengths, index_t iterations) override
	{
		mPointsSize = points;
		mEdgesSize = edges.size();
		mLengthsSize = lengths.size();

		CUCH(cudaMalloc((void **) &mCuPoints, points * sizeof(point_t)));
        CUCH(cudaMalloc((void **) &mCuForces, points * sizeof(point_t)));
        CUCH(cudaMalloc((void **) &mCuVelocities, points * sizeof(point_t)));
		CUCH(cudaMalloc((void **) &mCuEdges, edges.size() * sizeof(edge_t)));
		CUCH(cudaMalloc((void **) &mCuLengths, lengths.size() * sizeof(length_t)));
		CUCH(cudaMalloc((void **) &mCuRepulsiveForces, points * sizeof(point_t)));
        CUCH(cudaMalloc((void **) &mCuCompulsiveForces, points * sizeof(point_t)));

        CUCH(cudaMemset(mCuPoints, 0.0, points * sizeof(point_t)));
        CUCH(cudaMemset(mCuForces, 0.0, points * sizeof(point_t)));
        CUCH(cudaMemset(mCuVelocities, 0.0, points * sizeof(point_t)));
        CUCH(cudaMemset(mCuRepulsiveForces, 0.0, points * sizeof(point_t)));
        CUCH(cudaMemset(mCuCompulsiveForces, 0.0, points * sizeof(point_t)));
		CUCH(cudaMemcpy(mCuEdges, edges.data(), edges.size() * sizeof(edge_t), cudaMemcpyHostToDevice));
		CUCH(cudaMemcpy(mCuLengths, lengths.data(), lengths.size() * sizeof(length_t), cudaMemcpyHostToDevice));
	}

    /**
     * Perform one iteration of the simulation and update positions of the points.
     * @param points These points may be cached on GPU.
     */
	void iteration(std::vector<point_t> &points) override
	{
        if (mFirstIteration)
            CUCH(cudaMemcpy(mCuPoints, points.data(), points.size() * sizeof(point_t), cudaMemcpyHostToDevice));

        CUCH(cudaMemset(mCuRepulsiveForces, 0.0, points.size() * sizeof(point_t)));
        CUCH(cudaMemset(mCuForces, 0.0, points.size() * sizeof(point_t)));

        run_compute_repulsive(mCuRepulsiveForces, mCuPoints, points.size(), Base::mParams.vertexRepulsion);
        if (Base::mVerbose) {
            std::cout << "Printing repulsive forces:" << std::endl;
            printCudaArray(mCuRepulsiveForces, points.size());
        }

        run_compute_compulsive(mCuForces, mCuPoints, mCuEdges, mEdgesSize, mCuLengths, Base::mParams.edgeCompulsion);
        if (Base::mVerbose) {
            std::cout << "Printing compulsive forces:" << std::endl;
            printCudaArray(mCuForces, points.size());
        }

        CUCH(cudaDeviceSynchronize());

        run_array_sum(mCuForces, mCuRepulsiveForces, points.size());
        if (Base::mVerbose) {
            std::cout << "Printing forces:" << std::endl;
            printCudaArray(mCuForces, points.size());
        }

        CUCH(cudaDeviceSynchronize());

        run_update_velocities(mCuVelocities, mCuForces, points.size(), Base::mParams);

        CUCH(cudaDeviceSynchronize());

        run_update_point_positions(mCuPoints, points.size(), mCuVelocities, Base::mParams);

        copyCudaArrayToVector(points, mCuPoints, mPointsSize);

        mFirstIteration = false;
	}

    /**
     * Retrieve the velocities buffer from the GPU.
     * This operation is for vreification only and it does not have to be efficient.
     */
	void getVelocities(std::vector<point_t> &velocities) override
	{
        copyCudaArrayToVector(velocities, mCuVelocities, mPointsSize);
	}

private:
    template <typename T>
    void printCudaArray(T *cuda_array, size_t size) const
    {
        point_t *tmp_array = new point_t[size];
        CUCH(cudaMemcpy(tmp_array, cuda_array, size * sizeof(T), cudaMemcpyDeviceToHost));

        for (size_t i = 0; i < size; i++)
            std::cout << tmp_array[i] << " ";
        std::cout << std::endl;

        delete[] tmp_array;
    }

    template <typename T>
    void copyCudaArrayToVector(std::vector<T> &vec, T *cuda_array, size_t size) const
    {
	    T *tmp_array = new T[size];

        CUCH(cudaMemcpy(tmp_array, cuda_array, size * sizeof(T), cudaMemcpyDeviceToHost));
	    vec.clear();
	    vec.insert(vec.begin(), tmp_array, tmp_array + size);

	    delete[] tmp_array;
    }
};

std::ostream & operator<<(std::ostream &output, const Point<double> &point)
{
    return output << "(" << point.x << "," << point.y << ")";
}


#endif
