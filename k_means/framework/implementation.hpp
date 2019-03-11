#ifndef KMEANS_IMPLEMENTATION_HPP
#define KMEANS_IMPLEMENTATION_HPP

#include <iostream>
#include <cassert>

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
	auto middleIt = beginIt - sourceVector.size()/2;
	auto endIt = sourceVector.end();

	destVector.insert(destVector.begin(), middleIt, endIt);
	sourceVector.erase(middleIt, endIt);
}


class PointRange {
public:
	PointRange(const std::vector<std::pair<point_t, size_t>> &points)
		: points(points)
	{
	}

	PointRange(PointRange &otherRange, tbb::split)
	{
		assert(points.empty());
		half_split_vector(otherRange.points, points);
	}

	const std::vector<std::pair<point_t, size_t>> & get_points() const
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
	static const size_t minRange = 2;
	std::vector<std::pair<point_t, size_t>> points;
};


template<typename POINT=point_t>
struct Cluster {
	size_t index;
	size_t count;
	POINT sum;
	POINT centroid;
};

template<typename POINT=point_t>
class ClusterRange {
public:
	ClusterRange(const std::vector<Cluster<POINT>> &clusters)
		: clusters(clusters)
	{}

	ClusterRange(ClusterRange &other, tbb::split)
	{
	    assert(clusters.empty());
		half_split_vector(other.clusters, clusters);
	}

	std::vector<Cluster<POINT>> & get_clusters()
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
	const size_t minRange = 2;
	std::vector<Cluster<POINT>> clusters;
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
		sums.resize(k);
		counts.resize(k);
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
		this->points = &points;

		initClusters();
		initAssignments(assignments);

		while (iters > 0) {
		    iters--;
			computePointsAssignment(assignments);
			computeNewCentroids();
		}

		constructOutput(centroids, assignments);
		std::cout << std::endl;
	}

private:
	using coord_t = typename POINT::coord_t;
	size_t k;
	size_t iters;
	const std::vector<POINT> *points;
	std::vector<POINT> sums;
	std::vector<size_t> counts;
	std::vector<Cluster<POINT>> clusters;

	std::vector<std::pair<POINT, size_t>> createPointIdxPairVector()
	{
		assert(points);
		std::vector<std::pair<POINT, size_t>> vec;
		for (size_t i = 0; i < points->size(); i++) {
			vec.push_back(std::make_pair((*points)[i], i));
		}
		return vec;
	}

	void initCentroids(std::vector<POINT> &centroids)
	{
		assert(k > 0 && k < 256);
		assert(points);

		centroids.resize(k);
		for (size_t i = 0; i < k; i++) {
			centroids[i] = (*points)[i];
		}
	}

	void initClusters()
	{
		assert(points);

		clusters.resize(k);

		for (size_t i = 0; i < k; i++) {
			POINT point = (*points)[i];

			Cluster<POINT> cluster;
			cluster.index = i;
			cluster.count = 1;
			cluster.sum.x = point.x;
			cluster.sum.y = point.y;
			cluster.centroid = point;

			clusters.emplace_back(std::move(cluster));
		}
	}

	void initAssignments(std::vector<ASGN> &assignments)
	{
	    assert(points);
		assignments.resize(points->size());
	}

	void constructOutput(std::vector<POINT> &centroids, std::vector<ASGN> &assignments)
	{
		// TODO
	}

	// First part of the algorithm -- assign all the points to nearest cluster.
	void computePointsAssignment(std::vector<ASGN> &assignments)
	{
		std::vector<std::pair<POINT, size_t>> pointIdxVector = createPointIdxPairVector();
		PointRange pointRange(pointIdxVector);

		tbb::parallel_for(pointRange, [&](const PointRange &range)
		{
			for (const auto &item : range.get_points()) {
				point_t point = item.first;
				size_t pointIdx = item.second;

				Cluster<POINT> &nearestCluster = getNearestCluster(point);
				assignments[pointIdx] = static_cast<ASGN>(nearestCluster.index);

				nearestCluster.sum.x += point.x;
				nearestCluster.sum.y += point.y;
				nearestCluster.count++;
			}
		});
	}

	void computeNewCentroids()
	{
		ClusterRange<POINT> clusterRange(clusters);
		tbb::parallel_for(clusterRange, [&](const ClusterRange<POINT> &range)
		{
			for (auto &cluster : range.get_clusters()) {
				if (cluster.count == 0) {
					continue; // If the cluster is empty, keep its previous centroid.
				}
				cluster.centroid.x = cluster.sum.x / (std::int16_t)cluster.count;
				cluster.centroid.y = cluster.sum.y / (std::int16_t)cluster.count;
			}
		});
	}

	Cluster<POINT> & getNearestCluster(const POINT &point)
	{
		coord_t minDist = distance(point, clusters[0].centroid);
		Cluster<POINT> &nearest = clusters[0];
		for (std::size_t i = 1; i < clusters.size(); ++i) {
			coord_t dist = distance(point, clusters[i].centroid);
			if (dist < minDist) {
				minDist = dist;
				nearest = clusters[i];
			}
		}

		return nearest;
	}

	static coord_t distance(const POINT &point, const POINT &centroid)
	{
		std::int64_t dx = (std::int64_t)point.x - (std::int64_t)centroid.x;
		std::int64_t dy = (std::int64_t)point.y - (std::int64_t)centroid.y;
		// We do not have to count sqrt here.
		return (coord_t)(dx*dx + dy*dy);
	}
};


#endif
