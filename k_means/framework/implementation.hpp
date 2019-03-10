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
		initCentroids(centroids, points, k);
		initAssignments(assignments, points);

	    // Parallel for - assign all points. ////
	    // Construct range from points
	    std::vector<std::pair<POINT, size_t>> pointIdxVector;
	    createPointIdxPairVector(pointIdxVector, points);
		PointRange pointRange(pointIdxVector);

		tbb::parallel_for(pointRange, [&](const PointRange &range) {
			computePointsAssignment(range, centroids, assignments);
		});
		std::cout << std::endl;
	}

private:
	using coord_t = typename POINT::coord_t;
	std::vector<POINT> sums;
	std::vector<size_t> counts;

	void createPointIdxPairVector(std::vector<std::pair<POINT, size_t>> &vec, const std::vector<POINT> &points)
	{
		for (size_t i = 0; i < points.size(); i++) {
			vec.push_back(std::make_pair(points[i], i));
		}
	}

	void initCentroids(std::vector<POINT> &centroids, const std::vector<POINT> &points, size_t k)
	{
		assert(k > 0 && k < 256);

		centroids.resize(k);
		for (size_t i = 0; i < k; i++) {
			centroids[i] = points[i];
		}
	}

	void initAssignments(std::vector<ASGN> &assignments, const std::vector<POINT> &points)
	{
		assignments.resize(points.size());
	}

	// First part of the algorithm -- assign all the points to nearest cluster.
	void computePointsAssignment(const PointRange &range, const std::vector<POINT> &centroids,
								 std::vector<ASGN> &assignments)
	{
		for (size_t i = 0; i < range.get_points().size(); i++) {
			point_t point = range.get_points()[i].first;
			size_t pointIdx = range.get_points()[i].second;

			size_t nearest = getNearestCluster(point, centroids);
			assignments[pointIdx] = static_cast<ASGN>(nearest);
			sums[nearest].x += point.x;
			sums[nearest].y += point.y;
			++counts[nearest];
		}
	}

	static std::size_t getNearestCluster(const POINT &point, const std::vector<POINT> &centroids)
	{
		coord_t minDist = distance(point, centroids[0]);
		std::size_t nearest = 0;
		for (std::size_t i = 1; i < centroids.size(); ++i) {
			coord_t dist = distance(point, centroids[i]);
			if (dist < minDist) {
				minDist = dist;
				nearest = i;
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
