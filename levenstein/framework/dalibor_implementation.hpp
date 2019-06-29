#ifndef DALIBOR_IMPLEMENTATION_HPP
#define DALIBOR_IMPLEMENTATION_HPP

#include <memory>

#include "internal/interface.hpp"
#include "internal/exception.hpp"

#include <iostream>
#include <thread>

union MyIndexer
{
	std::size_t rowIdx;
	char fills[64];
};


template<typename C = char, typename DIST = std::size_t, bool DEBUG = false>
class DaliborEditDistance : public IEditDistance<C, DIST, DEBUG>
{
public:
	/*
	 * \brief Perform the initialization of the functor (e.g., allocate memory buffers).
	 * \param len1, len2 Lengths of first and second string respectively.
	 */
	virtual void init(DIST len1, DIST len2)
	{
		if (len1 < len2)
		{
			std::swap(len1, len2);
		}

		if (len1 < 32000 && len2 < 32000)
		{
			low = true;
			N = 32;
		}
		else
		{
			low = false;
			N = 64;
		}

		workingOnRow.resize(N);
		for (size_t i = 0; i < N; i++)
		{
			workingOnRow[i].rowIdx = 0;
		}

		lastItem.resize(len2);
		for (size_t i = 0; i < len2; i++)
		{
			lastItem[i] = i + 1;
		}

		block = len1 / N;

		size_t counter = 1;
		oneRow.resize(N);
		for (size_t i = 0; i < N; i++)
		{
			oneRow[i].resize(block);
			for (size_t j = 0; j < block; j++)
			{
				oneRow[i][j] = counter;
				counter++;
			}
		}

		if (DEBUG) {
            std::cout << "<";
            for (size_t i = 0; i < N; i++)
            {
                std::cout << "{";
                for (size_t j = 0; j < block; j++)
                {
                    std::cout << oneRow[i][j] << ",";
                }

                std::cout << "}" << std::endl;
            }
            std::cout << ">" << std::endl;
		}
	}

	std::size_t N = 64;

	std::size_t block;

	std::vector<MyIndexer> workingOnRow;
	std::vector<std::size_t> lastItem;

	bool low;

	/*
	 * \brief Compute the distance between two strings.
	 * \param str1, str2 Strings to be compared.
	 * \result The computed edit distance.
	 */
	virtual DIST compute(const std::vector<C> &str1, const std::vector<C> &str2)
	{
		if (str1.size() >= str2.size())
		{
			if (low)
				return compute_low(str1, str2);
			else
				return compute_high(str1, str2);
		}
		else
		{
			if (low)
				return compute_low(str2, str1);
			else
				return compute_high(str2, str1);
		}
	}

	const DIST &compute_low(const std::vector<C> & str1, const std::vector<C> & str2)
	{
		const C* s1 = &str1[0];
		const C* s2 = &str2[0];

		//std::size_t l1 = str1.size();
		std::size_t l2 = str2.size();

		#pragma omp parallel for shared(oneRow, workingOnRow, lastItem) num_threads(32)
		for (std::size_t t = 0; t < N; ++t)
		{
			size_t ib = t * block;
			size_t ie = std::min((t + 1) * block, l2);

			if (DEBUG)
                std::cout << ib << " az " << ie << std::endl;

			DIST lastUpper;
			DIST lastLeft = oneRow[t][0] - 1;

			for (size_t row = 0; row < l2; ++row)
			{
				while (t != 0 && workingOnRow[t - 1].rowIdx <= row)
				{
					std::this_thread::yield();
				}

				//if (row == 0) std::cout << " Working: " << t << std::endl;

				lastUpper = lastLeft;
				lastLeft = lastItem[row];

				DIST lastRowU = lastUpper;
				DIST lastRowL = lastLeft;

				for (size_t i = 0; i < block; ++i)
				{
					DIST d1 = oneRow[t][i] + 1; // Up
					DIST d2 = lastRowL + 1; // Left
					DIST d3 = lastRowU + (s1[ib + i] == s2[row] ? 0 : 1); // Left Up

					lastRowU = oneRow[t][i];
					lastRowL = oneRow[t][i] = std::min(std::min(d1, d2), d3);
				}

				lastItem[row] = oneRow[t][block - 1];
				workingOnRow[t].rowIdx = row + 1;

				if (DEBUG)
                    if (row == 0)
                        std::cout << " Done: " << t << "Row: " << workingOnRow[t].rowIdx << std::endl;
			}
		}

		return lastItem[l2 - 1];
	}

	const DIST &compute_high(const std::vector<C> & str1, const std::vector<C> & str2)
	{
		const C* s1 = &str1[0];
		const C* s2 = &str2[0];

		//std::size_t l1 = str1.size();
		std::size_t l2 = str2.size();

		#pragma omp parallel for shared(oneRow, workingOnRow, lastItem) num_threads(64)
		for (std::size_t t = 0; t < N; ++t)
		{
			size_t ib = t * block;
			//size_t ie = std::min((t + 1) * block, l2);

			//std::cout << ib << " az " << ie << std::endl;

			DIST lastUpper;
			DIST lastLeft = oneRow[t][0] - 1;

			for (size_t row = 0; row < l2; ++row)
			{
				while (t != 0 && workingOnRow[t - 1].rowIdx <= row)
				{
					std::this_thread::yield();
				}

				//if (row == 0) std::cout << " Working: " << t << std::endl;

				lastUpper = lastLeft;
				lastLeft = lastItem[row];

				DIST lastRowU = lastUpper;
				DIST lastRowL = lastLeft;

				for (size_t i = 0; i < block; ++i)
				{
					DIST d1 = oneRow[t][i] + 1; // Up
					DIST d2 = lastRowL + 1; // Left
					DIST d3 = lastRowU + (s1[ib + i] == s2[row] ? 0 : 1); // Left Up

					lastRowU = oneRow[t][i];
					lastRowL = oneRow[t][i] = std::min(std::min(d1, d2), d3);
				}

				lastItem[row] = oneRow[t][block - 1];
				workingOnRow[t].rowIdx = row + 1;

				//if (row == 0) std::cout << " Done: " << t << "Row: " << workingOnRow[t] << std::endl;
			}
		}

		return lastItem[l2 - 1];
	}

	std::vector<std::vector<std::size_t>> oneRow;

};


#endif
