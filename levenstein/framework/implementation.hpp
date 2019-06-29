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
#include <cmath>


struct Index {
    size_t idx;
    char _filling[64];

    bool operator==(const Index &rhs) const
    {
        return idx == rhs.idx;
    }

    bool operator!=(const Index &rhs) const
    {
        return !(rhs == *this);
    }

    friend std::ostream &operator<<(std::ostream &os, const Index &index)
    {
        os << "idx: " << index.idx;
        return os;
    }
};

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

	    if (len1 >= 32 * 1024 && len2 >= 32 * 1024)
	        mThreadCount = 64;
	    else
            mThreadCount = 32;

	    // For deubg - for small sizes of strings.
	    if (mThreadCount > mTotalRowsCount)
	        mThreadCount = mTotalRowsCount / 2;

	    mBlockSize = std::ceil((double)mTotalRowsCount / (double)mThreadCount);

	    mLastItemsInCol.resize(mTotalColsCount);
	    for (size_t i = 0; i < mTotalColsCount; ++i)
	        mLastItemsInCol[i] = i;

	    mThreadLefts.resize(mThreadCount);
	    size_t counter = 0;
	    for (size_t thread_idx = 0; thread_idx < mThreadCount; ++thread_idx) {
            mThreadLefts[thread_idx].resize(mBlockSize);
            for (size_t row = 0; row < mBlockSize; ++row) {
                mThreadLefts[thread_idx][row] = counter;
                counter++;
            }
        }

	    mActualIndexes.resize(mThreadCount);
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

        computeInParallel();

        mInputArray1 = nullptr;
        mInputArray2 = nullptr;

        return mLastItemsInCol[mTotalColsCount - 1];
	}

private:
    const std::vector<C> *mInputArray1;
    const std::vector<C> *mInputArray2;
    std::vector<DIST> mLastItemsInCol;
    std::vector<Index> mActualIndexes;
    /// mThreadLefts[t][r] ... value of left for a thread t on row r.
    std::vector<std::vector<DIST>> mThreadLefts;
    size_t mThreadCount;
    size_t mBlockSize;
	size_t mTotalRowsCount;
	size_t mTotalColsCount;

	void computeInParallel()
    {
        #pragma omp parallel for shared(mLastItemsInCol, mActualIndexes) num_threads(mThreadCount)
	    // Every thread computes block_size rows.
        for (size_t thread_idx = 0; thread_idx < mThreadCount; ++thread_idx) {
            size_t block_row_begin = thread_idx * mBlockSize;
            size_t block_row_end = std::min((thread_idx + 1) * mBlockSize, mTotalRowsCount);
            if (DEBUG)
                std::cout << "Thread (" << thread_idx << "): block_row_begin=" << block_row_begin
                          << " block_row_end=" << block_row_end << std::endl;

            // left_upper and left are part of first column.
            DIST left_upper = block_row_begin > 0 ? block_row_begin - 1 : 0;
            for (size_t col = 1; col < mTotalColsCount; ++col) {
                while (thread_idx > 0 && mActualIndexes[thread_idx - 1].idx < col) {
                    std::this_thread::yield();
                }

                DIST upper = mLastItemsInCol[col];
                DIST last_upper_for_col = upper;
                const DIST a = (*mInputArray1)[col - 1];
                DIST dist = 0;
                for (size_t total_row = block_row_begin, thread_lefts_row = 0;
                     total_row < block_row_end;
                     ++total_row, ++thread_lefts_row)
                {
                    if (total_row == 0)
                        continue;

                    DIST b = (*mInputArray2)[total_row - 1];
                    DIST left = mThreadLefts[thread_idx][thread_lefts_row];

                    if (DEBUG)
                        logOneCompute(total_row, col, upper, left_upper, left, a, b);
                    dist = computeDistance(upper, left_upper, left, a, b);

                    left_upper = left;
                    upper = dist;
                    mThreadLefts[thread_idx][thread_lefts_row] = dist;
                }

                mLastItemsInCol[col] = dist;
                mActualIndexes[thread_idx].idx = col;

                left_upper = last_upper_for_col;
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

    void logOneCompute(size_t row, size_t col, DIST upper, DIST left_upper, DIST left, DIST a, DIST b) const
    {
        std::cout << "Computing element at row=" << row << ", col=" << col << " with: upper=" << upper
                  << " left_upper=" << left_upper << " left=" << left << " a=" << (char)a << " b=" << (char)b
                  << std::endl;
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
