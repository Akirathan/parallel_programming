#ifndef KMEANS_IMPLEMENTATION_HPP
#define KMEANS_IMPLEMENTATION_HPP

#include <iostream>
#include <cassert>
#include <array>

#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <interface.hpp>
#include <exception.hpp>

static const size_t MAX_CLUSTER_COUNT = 256;
static const size_t POINTS_GRAIN_SIZE = 1024;
static const size_t CLUSTERS_GRAIN_SIZE = 16;

struct PointWithAssignment: public point_t {
    size_t idx;
	size_t assignedClusterIdx;

	PointWithAssignment(size_t idx, point_t point)
		: idx(idx),
		assignedClusterIdx(0)
	{
		this->x = point.x;
		this->y = point.y;
	}

	PointWithAssignment()
        : PointWithAssignment(0, {0, 0})
	{}
};

template<typename POINT=point_t>
struct Cluster {
	size_t index;
	POINT centroid;

	Cluster(size_t index, POINT centroid)
		: index(index),
		centroid(centroid)
	{}
};

struct SumCountArrays {
	std::array<size_t, MAX_CLUSTER_COUNT> counts;
	std::array<point_t, MAX_CLUSTER_COUNT> sums;

	SumCountArrays()
	{
		counts.fill(0);
		sums.fill({0, 0});
	};
};


template<typename POINT = point_t, typename ASGN = std::uint8_t, bool DEBUG = false>
class KMeans : public IKMeans<POINT, ASGN, DEBUG>
{
public:
	/*
	 * \brief Perform the initialization of the functor (e.g., allocate memory buffers).
	 * \param points Number of points being clustered.
	 * \param k Number of clusters.
	 * \param iters Number of refining iterations.
	 */
	virtual void init(std::size_t points, std::size_t k, std::size_t iters)
	{
        this->k = k;
	}


	/*
	 * \brief Perform the clustering and return the cluster centroids and point assignment
	 *		yielded by the last iteration.
	 * \note First k points are taken as initial centroids for first iteration.
	 * \param points Vector with input points.
	 * \param k Number of clusters.
	 * \param iters Number of refining iterations.
	 * \param centroids Vector where the final cluster centroids should be stored.
	 * \param assignments Vector where the final assignment of the points should be stored.
	 *		The indices should correspond to point indices in 'points' vector.
	 */
	virtual void compute(const std::vector<POINT> &points, std::size_t k, std::size_t iters,
		std::vector<POINT> &centroids, std::vector<ASGN> &assignments)
	{
		this->assignments = &assignments;

		initPoints(points);
		initClusters();
		initAssignments();
		initCentroids(centroids);

		if (DEBUG) {
			std::cerr << "Starting with clusters: " << std::endl;
			printClusters();
		}

		while (iters > 0) {
		    iters--;
		    bool finalIteration = iters == 0;

			SumCountArrays arrays = computePointsAssignment(finalIteration);
			computeNewCentroids(arrays);
		}

		constructOutput(centroids, assignments);
	}

private:
	using coord_t = typename POINT::coord_t;
	size_t k;
	std::vector<PointWithAssignment> points;
	std::vector<ASGN> *assignments;
	std::vector<Cluster<POINT>> clusters;

	void initPoints(const std::vector<POINT> &points)
	{
	    for (size_t i = 0; i < points.size(); i++) {
	    	this->points.emplace_back(i, points[i]);
	    }
	}

	void initClusters()
	{
		for (size_t i = 0; i < k; i++) {
			POINT &point = points[i];

			clusters.emplace_back(i, point);
		}
	}

	void initAssignments()
	{
	    assert(assignments);
		assignments->resize(points.size());
	}

	void initCentroids(std::vector<POINT> &centroids)
	{
		centroids.resize(k);
	}

	// First part of the algorithm -- assign all the points to nearest cluster.
	SumCountArrays computePointsAssignment(bool finalIteration)
	{
		SumCountArrays finalArrays =
            tbb::parallel_reduce(
            		tbb::blocked_range<size_t>(0, points.size(), POINTS_GRAIN_SIZE),
                    SumCountArrays(),
                    [&](const tbb::blocked_range<size_t> &range, SumCountArrays arrays) -> SumCountArrays {
                        for (size_t i = range.begin(); i != range.end(); i++) {
                            Cluster<POINT> &nearestCluster = getNearestCluster(points[i]);

                            if (finalIteration) {
                                assignPointIdxToCluster(i, nearestCluster);
                            }

                            arrays.counts[nearestCluster.index]++;
                            arrays.sums[nearestCluster.index].x += points[i].x;
                            arrays.sums[nearestCluster.index].y += points[i].y;
                        }
                        return arrays;
                    },
                    [](const SumCountArrays &arrays1, const SumCountArrays &arrays2) -> SumCountArrays {
                        SumCountArrays resArrays;
                        for (size_t i = 0; i < arrays1.sums.max_size(); i++) {
                            resArrays.sums[i].x = arrays1.sums[i].x + arrays2.sums[i].x;
                            resArrays.sums[i].y = arrays1.sums[i].y + arrays2.sums[i].y;
                            resArrays.counts[i] = arrays1.counts[i] + arrays2.counts[i];
                        }
                        return resArrays;
                    }
            );

		if (DEBUG) {
			std::cerr << "computePointsAssignment finished" << std::endl;
			printClusters();
		}

		return finalArrays;
	}

	void computeNewCentroids(const SumCountArrays &arrays)
	{
		tbb::parallel_for(tbb::blocked_range<size_t>(0, k, CLUSTERS_GRAIN_SIZE),
            [&](const tbb::blocked_range<size_t> &range) {
                for (size_t i = range.begin(); i != range.end(); i++) {
                    if (arrays.counts[i] == 0) {
                        continue; // If the cluster is empty, keep its previous centroid.
                    }
                    clusters[i].centroid.x = arrays.sums[i].x / (std::int64_t)arrays.counts[i];
                    clusters[i].centroid.y = arrays.sums[i].y / (std::int64_t)arrays.counts[i];
                }
            }
        );
	}

	Cluster<POINT> & getNearestCluster(const POINT &point)
	{
		coord_t minDist = distance(point, clusters[0].centroid);
		size_t nearestIdx = 0;
		for (std::size_t i = 1; i < k; ++i) {
			coord_t dist = distance(point, clusters[i].centroid);
			if (dist < minDist) {
				minDist = dist;
				nearestIdx = i;
			}
		}

		return clusters[nearestIdx];
	}

	static coord_t distance(const POINT &point, const POINT &centroid)
	{
		std::int64_t dx = (std::int64_t)point.x - (std::int64_t)centroid.x;
		std::int64_t dy = (std::int64_t)point.y - (std::int64_t)centroid.y;
		return (coord_t)(dx*dx + dy*dy);
	}

	void assignPointIdxToCluster(const size_t pointIdx, const Cluster<POINT> &cluster)
	{
		(*assignments)[pointIdx] = static_cast<ASGN>(cluster.index);
	}

	void constructOutput(std::vector<POINT> &centroids, std::vector<ASGN> &assignments)
	{
		assignments = *this->assignments;

		for (size_t i = 0; i < clusters.size(); i++) {
			centroids[i].x = clusters[i].centroid.x;
			centroids[i].y = clusters[i].centroid.y;
		}
	}

	void printClusters()
	{
		for (const auto &cluster: clusters) {
			std::cerr << "Cluster: index=" << cluster.index
					  << ", centroid.x=" << cluster.centroid.x << ", centroid.y=" << cluster.centroid.y << std::endl;
		}
	}
};


#endif
