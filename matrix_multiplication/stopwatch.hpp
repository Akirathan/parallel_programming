/*
 * Author: Martin Krulis <krulis@ksi.mff.cuni.cz>
 * Last Modification: 1.7.2013
 * License: CC 3.0 BY-NC (http://creativecommons.org/)
 */
#ifndef BPPLIB_SYSTEM_STOPWATCH_HPP
#define BPPLIB_SYSTEM_STOPWATCH_HPP


#include <unistd.h>
#include <time.h>
#include <sys/times.h>
#include <sys/time.h>

/**
 * \brief Implementation of high precision wall-time stopwatch based on system timers.
 */
class Stopwatch
{
private:
	typedef unsigned long long ticks_t;

	ticks_t mStartTime;
	double mLastInterval;
	bool mTiming;

	/**
	 * \brief Get current system timer status in ticks.
	 */
	ticks_t now()
	{
		struct timespec ts;
		::clock_gettime(CLOCK_REALTIME, &ts);
		return static_cast<ticks_t>(ts.tv_sec) * 1000000000UL + static_cast<ticks_t>(ts.tv_nsec);
	}


	/**
	 * Measure current time and update mLastInterval.
	 */
	void measureTime()
	{
		mLastInterval = static_cast<double>((now() - mStartTime)*1E-9);
	}


public:
	/**
	 * \brief Create new stopwatch. The stopwatch are not running when created.
	 */
	Stopwatch() : mTiming(false) { }

	/**
	 * \brief Create new stopwatch (and optionaly start it).
	 * \param start If start is true, the stapwatch are started immediately.
	 */
	Stopwatch(bool start)
	{
		if (start) this->start();
	}

	/**
	 * \brief Start the stopwatch. If the stopwatch are already timing, they are reset.
	 */
	void start()
	{
		mTiming = true;
		mStartTime = now();
	}

	/**
	 * \brief Stop the stopwatch. Multiple invocation has no effect.
	 */
	void stop()
	{
		if (mTiming == false) return;
		mTiming = false;
		measureTime();
	}

	/**
	 * \brief Return elapsed time in seconds since start method has been called.
	 *		If the stopwatch is not timing, last measured interval is returned.
	 */
	double getSeconds()
	{
		if (mTiming)
			measureTime();
		return mLastInterval;
	}

	/**
	 * \brief Return elapsed time in miliseconds since start method has been called.
	 *		If the stopwatch is not timing, last measured interval is returned.
	 */
	double getMiliseconds()
	{
		return getSeconds() * 1000.0;
	}
};

#endif
