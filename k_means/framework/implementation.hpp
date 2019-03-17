#ifndef KMEANS_IMPLEMENTATION_HPP
#define KMEANS_IMPLEMENTATION_HPP

#include <iostream>
#include <cassert>
#include <array>

#include <tbb/parallel_for.h>

#include <interface.hpp>
#include <exception.hpp>

/**
 *
 * @tparam T
 * @param sourceVector The second half of this vector will be moved to destVector
 * @param destVector
 */
template<typename T>
void half_split_vector(std::vector<T> &sourceVector, std::vector<T> &destVector)
{
	auto beginIt = sourceVector.begin();
	auto middleIt = beginIt + sourceVector.size()/2;
	auto endIt = sourceVector.end();

	destVector.insert(destVector.begin(), middleIt, endIt);
	sourceVector.erase(middleIt, endIt);
}


struct PointWithAssignment: public point_t {
    size_t idx;
	size_t assignedClusterIdx;

	PointWithAssignment(size_t idx)
		: idx(idx),
		assignedClusterIdx(0)
	{}

	PointWithAssignment()
        : PointWithAssignment(0)
	{}
};


class PointRange {
public:
	PointRange(const std::vector<PointWithAssignment> &points)
		: points(points)
	{
	}

	PointRange(PointRange &otherRange, tbb::split)
	{
		assert(points.empty());
		half_split_vector(otherRange.points, points);
	}

	const std::vector<PointWithAssignment> & get_points() const
	{
		return points;
	}

	bool is_divisible() const
	{
		return points.size() > minRange;
	}

	bool empty() const
	{
		return points.empty();
	}

private:
	static const size_t minRange = 256;
	std::vector<PointWithAssignment> points;
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

template<typename POINT=point_t>
class ClusterRange {
public:
	ClusterRange(std::vector<Cluster<POINT>> &clusters)
	{
		for (auto &cluster : clusters) {
			this->clusters.push_back(&cluster);
		}
	}

	ClusterRange(ClusterRange &other, tbb::split)
	{
	    assert(clusters.empty());
		half_split_vector(other.clusters, clusters);
	}

	std::vector<Cluster<POINT> *> & get_clusters()
	{
		return clusters;
	}

    const std::vector<Cluster<POINT> *> & get_clusters() const
    {
        return clusters;
    }

	bool is_divisible() const
	{
		return clusters.size() > minRange;
	}

	bool empty() const
	{
		return clusters.empty();
	}

private:
	const size_t minRange = 256;
	std::vector<Cluster<POINT> *> clusters;
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
		this->iters = iters;
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
		    resetBeforeIteration();
			computePointsAssignment();
			computeNewCentroids();
		}

		constructOutput(centroids, assignments);
	}

private:
	using coord_t = typename POINT::coord_t;
	static const size_t MAX_CLUSTER_COUNT = 256;
	size_t k;
	size_t iters;
	std::vector<PointWithAssignment> points;
	std::vector<ASGN> *assignments;
	std::vector<Cluster<POINT>> clusters;

	void initPoints(const std::vector<POINT> &points)
	{
	    for (size_t i = 0; i < points.size(); i++) {
	    	this->points.emplace_back(i);
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

	void resetBeforeIteration()
	{
	    // TODO
	}

	void constructOutput(std::vector<POINT> &centroids, std::vector<ASGN> &assignments)
	{
		assignments = *this->assignments;

		for (size_t i = 0; i < clusters.size(); i++) {
			centroids[i].x = clusters[i].centroid.x;
			centroids[i].y = clusters[i].centroid.y;
		}
	}

	// First part of the algorithm -- assign all the points to nearest cluster.
	void computePointsAssignment()
	{
		PointRange pointRange(points);

		tbb::parallel_for(pointRange, [&](const PointRange &range)
		{
			for (const PointWithAssignment &point : range.get_points()) {

				Cluster<POINT> &nearestCluster = getNearestCluster(point);

				assignPointIdxToCluster(point.idx, nearestCluster);
				if (DEBUG) {
					std::cerr << "Assigning point with index " << point.idx
					          << " to cluster with index " << nearestCluster.index << std::endl;
				}
			}
		});

		if (DEBUG) {
			std::cerr << "computePointsAssignment finished" << std::endl;
			printClusters();
		}
	}

	void computeNewCentroids()
	{
		// Compute temporary values - sums and counts for every cluster.
		std::array<size_t, MAX_CLUSTER_COUNT> counts;
		std::array<POINT, MAX_CLUSTER_COUNT> sums;
		counts.fill(0);
		sums.fill({0, 0});

		for (const PointWithAssignment &point : points) {
			sums[point.assignedClusterIdx].x += point.x;
			sums[point.assignedClusterIdx].y += point.y;
			counts[point.assignedClusterIdx]++;
		}

		// Compute new centroids
		tbb::parallel_for(tbb::blocked_range<size_t>(0, k), [&](const tbb::blocked_range<size_t> &range)
		{
			for (size_t i = range.begin(); i != range.end(); i++) {
				if (counts[i] == 0) {
					continue; // If the cluster is empty, keep its previous centroid.
				}
				clusters[i].centroid.x = sums[i].x / (std::int64_t)counts[i];
				clusters[i].centroid.y = sums[i].y / (std::int64_t)counts[i];
			}
		});
	}

	Cluster<POINT> & getNearestCluster(const POINT &point)
	{
		coord_t minDist = distance(point, clusters[0].centroid);
		size_t nearestIdx = 0;
		for (std::size_t i = 1; i < clusters.size(); ++i) {
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
		// We do not have to count sqrt here.
		return (coord_t)(dx*dx + dy*dy);
	}

	void assignPointIdxToCluster(const size_t pointIdx, const Cluster<POINT> &cluster)
	{
		(*assignments)[pointIdx] = static_cast<ASGN>(cluster.index);
		points[pointIdx].assignedClusterIdx = cluster.index;
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
