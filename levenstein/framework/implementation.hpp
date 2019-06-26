#ifndef LEVENSHTEIN_IMPLEMENTATION_HPP
#define LEVENSHTEIN_IMPLEMENTATION_HPP

#include <interface.hpp>
#include <exception.hpp>
#include <cassert>
#include <vector>
#include <iostream>
#include <algorithm>
#include <atomic>
#include <omp.h>
#include <thread>


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
	    mTotalColsCount = std::max(len1, len2) + 1;
	    mTotalRowsCount = std::min(len1, len2) + 1;
	    assert(mTotalColsCount >= mTotalRowsCount);

	    // TODO: Poladit tuhle velikost.
	    mBlockSize = omp_get_num_procs();

	    mLastItemsInCol.resize(mTotalColsCount);
	    for (size_t i = 0; i < mTotalColsCount; ++i)
	        mLastItemsInCol[i] = i;

	    mActualIndexes.resize(mBlockSize);
	}

	/*
	 * \brief Compute the distance between two strings.
	 * \param str1, str2 Strings to be compared.
	 * \result The computed edit distance.
	 */
	DIST compute(const std::vector<C> &str1, const std::vector<C> &str2) override
	{
        if (str1.size() > str2.size()) {
            mInputArray1 = &str1;
            mInputArray2 = &str2;
        }
        else {
	        mInputArray1 = &str2;
	        mInputArray2 = &str1;
	    }
	    assert(mInputArray1->size() >= mInputArray2->size());

        size_t upper_row_idx = 1;
        for (;
             upper_row_idx <= mTotalRowsCount - mBlockSize;
             upper_row_idx = std::min(upper_row_idx + mBlockSize, mTotalRowsCount))
        {
            computeStripe(upper_row_idx);
            if (DEBUG)
                logIteration();
            prepareForNextIteration();
            if (DEBUG) {
                std::cout << "After reinitialization:" << std::endl;
                logIteration();
            }
        }

        // Compute rest.
        for (size_t row = upper_row_idx + 1; row < mTotalRowsCount; ++row) {
            DIST left_upper = row - 1;
            DIST left = left_upper + 1;
            DIST b = (*mInputArray2)[row - 1];

            for (size_t col = 1; col < mTotalColsCount; ++col) {
                DIST upper = mLastItemsInCol[col];
                DIST a = (*mInputArray1)[col - 1];
                DIST dist = computeDistance(upper, left_upper, left, a, b);

                mLastItemsInCol[col] = dist;

                left_upper = upper;
                left = dist;
            }
        }

        mInputArray1 = nullptr;
        mInputArray2 = nullptr;

        return mLastItemsInCol[mTotalColsCount - 1];
	}

private:
    const std::vector<C> *mInputArray1;
    const std::vector<C> *mInputArray2;
    std::vector<DIST> mLastItemsInCol;
    std::vector<size_t> mActualIndexes;
	size_t mTotalRowsCount;
	size_t mTotalColsCount;
	size_t mBlockSize;

	void computeStripe(size_t upper_row_idx)
    {

#pragma omp parallel for shared(upper_row_idx)
	    // Every thread computes one row.
        for (size_t thread_idx = 0; thread_idx < mBlockSize; ++thread_idx) {
            const size_t total_i = upper_row_idx + thread_idx;
            assert(total_i > 0);
            // left_upper and left are part of first column.
            DIST left_upper = total_i - 1;
            DIST left = left_upper + 1;
            DIST b = (*mInputArray2)[total_i - 1];
            for (size_t col = 1; col < mTotalColsCount; ++col) {
                while (thread_idx > 0 && mActualIndexes[thread_idx - 1] < col) {
                    // TODO std::this_thread::yield();
                }
                DIST upper = mLastItemsInCol[col];
                DIST a = (*mInputArray1)[col - 1];
                DIST dist = computeDistance(upper, left_upper, left, a, b);

                mLastItemsInCol[col] = dist;
                mActualIndexes[thread_idx] += 1;

                left_upper = upper;
                left = dist;
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

    void prepareForNextIteration()
    {
	    for (size_t &index : mActualIndexes)
	        index = 0;
    }

    void logIteration() const
    {
        printVector(std::cout, mLastItemsInCol, "Last items");
        printVector(std::cout, mActualIndexes, "Actual indexes");
    }

    template <typename T>
    void printVector(std::ostream &output, const std::vector<T> &vector, const std::string &vector_name) const
    {
	    output << vector_name << ": [";
	    const T &last_item = vector[vector.size() - 1];
	    for (auto &&item : vector) {
            if (item == last_item)
	            output << item;
            else
                output << item << ", ";
	    }
	    output << "]" << std::endl;
    }

};


#endif
