#ifndef KMEANS_IMPLEMENTATION_HPP
#define KMEANS_IMPLEMENTATION_HPP

#include <iostream>
#include <cassert>

#include <tbb/parallel_for.h>

#include <interface.hpp>
#include <exception.hpp>

class PointRange {
public:
	PointRange(const std::vector<std::pair<point_t, size_t>> &points)
		: points(points)
	{
	}

	PointRange(PointRange &otherRange, tbb::split)
	{
		half_split(otherRange);
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

	void half_split(PointRange &otherRange)
	{
		std::vector<std::pair<point_t, size_t>> tmpPoints = otherRange.points;
		auto beginIt = tmpPoints.begin();
		auto endIt = tmpPoints.end();
		auto halfIt = endIt - tmpPoints.size()/2;

		assert(points.empty());
		points.insert(points.begin(), beginIt, halfIt);

		std::vector<std::pair<point_t, size_t>> &otherPoints = otherRange.points;
		otherPoints.clear();
		otherPoints.insert(otherPoints.begin(), halfIt, endIt);
	}
};

template<typename POINT=point_t>
struct Cluster {
	size_t index;
	size_t count;
	POINT sum;
	POINT centroid;
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
			// Parallel for - assign all points. ////
			// Construct range from points
			std::vector<std::pair<POINT, size_t>> pointIdxVector;
			createPointIdxPairVector(pointIdxVector);
			PointRange pointRange(pointIdxVector);

			tbb::parallel_for(pointRange, [&](const PointRange &range) {
				computePointsAssignment(range, centroids, assignments);
			});
		}
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

	void createPointIdxPairVector(std::vector<std::pair<POINT, size_t>> &vec)
	{
		assert(points);
		for (size_t i = 0; i < points->size(); i++) {
			vec.push_back(std::make_pair((*points)[i], i));
		}
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

	// First part of the algorithm -- assign all the points to nearest cluster.
	void computePointsAssignment(const PointRange &range, const std::vector<POINT> &centroids,
			std::vector<ASGN> &assignments)
	{
		for (size_t i = 0; i < range.get_points().size(); i++) {
			point_t point = range.get_points()[i].first;
			size_t pointIdx = range.get_points()[i].second;

			Cluster<POINT> &nearestCluster = getNearestCluster(point);
			assignments[pointIdx] = static_cast<ASGN>(nearestCluster.index);

			nearestCluster.sum.x += point.x;
			nearestCluster.sum.y += point.y;
			nearestCluster.count++;
		}
	}
		}
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
