#ifndef KMEANS_IMPLEMENTATION_HPP
#define KMEANS_IMPLEMENTATION_HPP

#include <iostream>
#include <cassert>

#include <tbb/parallel_for.h>

#include <interface.hpp>
#include <exception.hpp>

class PointRange {
public:
	PointRange(const std::vector<point_t> &points)
		: points(points)
	{
	}

	PointRange(PointRange &otherRange, tbb::split)
	{
		half_split(otherRange);
	}

	const std::vector<point_t> & get_points() const
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
	std::vector<point_t> points;

	void half_split(PointRange &otherRange)
	{
		std::vector<point_t> tmpPoints = otherRange.points;
		auto beginIt = tmpPoints.begin();
		auto endIt = tmpPoints.end();
		auto halfIt = endIt - tmpPoints.size()/2;

		assert(points.empty());
		points.insert(points.begin(), beginIt, halfIt);

		std::vector<point_t> &otherPoints = otherRange.points;
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
	    // Parallel for - assign all points. ////
	    // Construct range from points
		PointRange pointRange(points);
		tbb::parallel_for(pointRange, [&](const PointRange &range){pointsAssignment(range);});
		std::cout << std::endl;
	}

private:
	using coord_t = typename POINT::coord_t;

	void pointsAssignment(const PointRange &range)
	{

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
