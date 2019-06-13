#ifndef LEVENSHTEIN_IMPLEMENTATION_HPP
#define LEVENSHTEIN_IMPLEMENTATION_HPP

#include <interface.hpp>
#include <exception.hpp>
#include <vector>
#include <iostream>
#include <algorithm>
#include <atomic>
#include <omp.h>


template<typename C = char, typename DIST = std::size_t, bool DEBUG = false>
class EditDistance : public IEditDistance<C, DIST, DEBUG>
{
public:

    EditDistance() :
            mInputArray1{nullptr},
            mInputArray2{nullptr},
            mTotalRowsCount{0},
            mTotalColsCount{0}
    {}

	/*
	 * \brief Perform the initialization of the functor (e.g., allocate memory buffers).
	 * \param len1, len2 Lengths of first and second string respectively.
	 */
	void init(DIST len1, DIST len2) override
	{
	    mTotalColsCount = static_cast<size_t>(len1) + 1;
	    mTotalRowsCount = static_cast<size_t>(len2) + 1;

	    mRectangle.resize(omp_get_num_procs());
	    mFlagsRectangle.resize(omp_get_num_procs());
	    for (std::vector<DIST> &row : mRectangle) {
	        row.resize(mTotalColsCount, 0);
	    }
	    for (std::vector<bool> &row : mFlagsRectangle) {
	        row.resize(mTotalColsCount, 0);
	    }

        reinitializeRectangle();
	}

	/*
	 * \brief Compute the distance between two strings.
	 * \param str1, str2 Strings to be compared.
	 * \result The computed edit distance.
	 */
	DIST compute(const std::vector<C> &str1, const std::vector<C> &str2) override
	{
        mInputArray1 = &str1;
        mInputArray2 = &str2;
        reinitializeRectangle();

        size_t upper_row_idx = 0;
        for (;
             upper_row_idx <= mTotalRowsCount - mRectangle.size();
             upper_row_idx += mRectangle.size() - 1)
        {
            computeRectangle(upper_row_idx);
            if (DEBUG)
                logRectangle();
            prepareRectangleForNextIteration();
            if (DEBUG) {
                std::cout << "After reinitialization:" << std::endl;
                logRectangle();
            }
        }

        // Compute rest.
        size_t last_rectangle_i = mRectangle.size() - 1;
        size_t last_rectangle_j = mTotalColsCount - 1;
        for (size_t total_i = upper_row_idx + 1, rectangle_i = 1;
             total_i < mTotalRowsCount && rectangle_i < mRectangle.size();
             ++total_i, ++rectangle_i)
        {
            for (size_t j = 1; j < mTotalColsCount; ++j) {
                DIST upper = mRectangle[rectangle_i-1][j];
                DIST left_upper = mRectangle[rectangle_i-1][j-1];
                DIST left = mRectangle[rectangle_i][j-1];
                DIST a = (*mInputArray1)[j-1];
                DIST b = (*mInputArray2)[total_i-1];
                DIST dist = computeDistance(upper, left_upper, left, a, b);
                mRectangle[rectangle_i][j] = dist;
                last_rectangle_i = rectangle_i;
                last_rectangle_j = j;
            }
        }

        mInputArray1 = nullptr;
        mInputArray2 = nullptr;

        return mRectangle[last_rectangle_i][last_rectangle_j];
	}

private:
    const std::vector<C> *mInputArray1;
    const std::vector<C> *mInputArray2;
    std::vector<std::vector<DIST>> mRectangle;
    std::vector<std::vector<bool>> mFlagsRectangle;
	size_t mTotalRowsCount;
	size_t mTotalColsCount;

	void computeRectangle(size_t upper_row_idx)
    {

#pragma omp parallel for shared(upper_row_idx)
	    // Every thread computes one row.
        for (size_t i = 1; i < mRectangle.size(); ++i) {
            const size_t total_i = upper_row_idx + i;
            for (size_t j = 1; j < mTotalColsCount; ++j) {
                while (!mFlagsRectangle[i-1][j] || !mFlagsRectangle[i-1][j-1]) {
                    // Active wait for another thread.
                }
                DIST upper = mRectangle[i-1][j];
                DIST left_upper = mRectangle[i-1][j-1];
                DIST left = mRectangle[i][j-1];
                DIST a = (*mInputArray1)[j-1];
                DIST b = (*mInputArray2)[total_i-1];
                DIST dist = computeDistance(upper, left_upper, left, a, b);
                mRectangle[i][j] = dist;
                mFlagsRectangle[i][j] = true;
            }
        }
    }

	DIST computeDistance(DIST upper, DIST left_upper, DIST left, DIST a, DIST b) const
    {
		DIST first = upper + 1;
		DIST second = left_upper + (a == b ? 0 : 1);
		DIST third = left + 1;
		return std::min({first, second, third});
    }

    void reinitializeRectangle()
    {
        for (auto &&row : mFlagsRectangle) {
            std::fill(row.begin(), row.end(), false);
        }

        // Initialize first row.
        for (size_t i = 0; i < mRectangle[0].size(); ++i) {
            mRectangle[0][i] = static_cast<DIST>(i);
            mFlagsRectangle[0][i] = true;
        }

        // Initialize first column.
        for (size_t j = 0; j < mRectangle.size(); ++j) {
            mRectangle[j][0] = static_cast<DIST>(j);
            mFlagsRectangle[j][0] = true;
        }
    }

    void prepareRectangleForNextIteration()
    {
        const size_t rectangle_rows = mRectangle.size();
        const size_t rectangle_cols = mRectangle[0].size();

        for (auto &&row : mFlagsRectangle)
            for (auto &&item : row)
                item = false;

        // Copy last row to first row.
        for (size_t j = 0; j < rectangle_cols; ++j) {
            mRectangle[0][j] = mRectangle[rectangle_rows - 1][j];
            mFlagsRectangle[0][j] = true;
        }

        // Reinitialize first column.
        for (size_t i = 1; i < rectangle_rows; ++i) {
            mRectangle[i][0] = mRectangle[0][0] + i;
            mFlagsRectangle[i][0] = true;
        }
    }

    void logRectangle() const
    {
        std::cout << "Rectangle:" << std::endl;
        for (size_t i = 0; i < mRectangle.size(); ++i) {
            for (size_t j = 0; j < mRectangle[0].size(); ++j)
                std::cout << mRectangle[i][j] << " ";
            std::cout << std::endl;
        }

        std::cout << "Flags rectangle:" << std::endl;
        for (size_t i = 0; i < mFlagsRectangle.size(); ++i) {
            for (size_t j = 0; j < mFlagsRectangle[0].size(); ++j)
                std::cout << mFlagsRectangle[i][j] << " ";
            std::cout << std::endl;
        }
    }

};


#endif
