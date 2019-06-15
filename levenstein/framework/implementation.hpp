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
            mTotalColsCount{0},
            mDiagonalLen{0},
            mLastDiagonalLen{0},
            mLastLastDiagonalLen{0},
            mIsInFirstHalfOfDiagonals{false}
    {}

	/*
	 * \brief Perform the initialization of the functor (e.g., allocate memory buffers).
	 * \param len1, len2 Lengths of first and second string respectively.
	 */
	void init(DIST len1, DIST len2) override
	{
	    mTotalColsCount = static_cast<size_t>(len1) + 1;
	    mTotalRowsCount = static_cast<size_t>(len2) + 1;
	    size_t diag_len = std::min(mTotalColsCount, mTotalRowsCount);

	    mDiagonal.resize(diag_len);
        mLastDiagonal.resize(diag_len);
        mLastLastDiagonal.resize(diag_len);

        mLastLastDiagonal[0] = 0;
        mLastDiagonal[0] = 1;
        mLastDiagonal[1] = 1;
        mDiagonal[0] = 2;
        mDiagonal[2] = 2;

        mLastLastDiagonalLen = 1;
        mLastDiagonalLen = 2;
        mDiagonalLen = 3;
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

        size_t diagonal_count = mTotalRowsCount + mTotalRowsCount - 1;
        for (size_t diag_idx = 2; diag_idx < diagonal_count; ++diag_idx) {
            if (DEBUG)
                std::cout << "================" << std::endl;

            size_t start_row = std::min(diag_idx, mTotalColsCount - 1);
            size_t start_col = diag_idx - start_row;
            mDiagonalLen = std::min(start_row, mTotalColsCount - 1 - start_col) + 1;
            mIsInFirstHalfOfDiagonals = diag_idx < mTotalRowsCount;

            computeDiagonal(start_row, start_col, diag_idx);
            if (DEBUG)
                logDiagonals();
            prepareDiagonalsForNextIteration(diag_idx);
        }

        mInputArray1 = nullptr;
        mInputArray2 = nullptr;

        return mDiagonal[0];
	}

private:
    static constexpr size_t chunk_size = 2;
    const std::vector<C> *mInputArray1;
    const std::vector<C> *mInputArray2;
    size_t mTotalRowsCount;
    size_t mTotalColsCount;
    std::vector<DIST> mDiagonal;
    std::vector<DIST> mLastDiagonal;
    std::vector<DIST> mLastLastDiagonal;
    size_t mDiagonalLen;
    size_t mLastDiagonalLen;
    size_t mLastLastDiagonalLen;
    bool mIsInFirstHalfOfDiagonals;

    /// \param start_row Index of row where current diagonal begins.
    /// \param start_col Index of column where current diagonal begins.
    /// \param diag_idx Index of current diagonal.
	void computeDiagonal(size_t start_row, size_t start_col, size_t diag_idx)
    {
	    size_t last_idx_in_diag = 0;
//#pragma omp parallel for shared(start_row, start_col, last_idx_in_diag)
	    for (size_t start_idx_in_diag = 0; start_idx_in_diag < mDiagonalLen; start_idx_in_diag += chunk_size) {
	        if (start_idx_in_diag == 0 && mIsInFirstHalfOfDiagonals)
	            start_idx_in_diag++;
	        else if (start_idx_in_diag == mDiagonalLen - 1 && mIsInFirstHalfOfDiagonals)
                break;

	        // If in last chunk.
	        if (start_idx_in_diag + 2*chunk_size > mDiagonalLen)
	            last_idx_in_diag = start_idx_in_diag + chunk_size;

	        // Iterate one chunk in diagonal.
	        size_t end_idx_in_diag = mIsInFirstHalfOfDiagonals ? mDiagonalLen - 1 : mDiagonalLen;
	        for (size_t idx_in_diag = start_idx_in_diag;
	             idx_in_diag < start_idx_in_diag + chunk_size && idx_in_diag < end_idx_in_diag;
	             ++idx_in_diag)
	        {
                mDiagonal[idx_in_diag] = computeAtDiagonal(diag_idx, start_row, start_col, idx_in_diag);
	        }
	    }

	    // Compute rest of diagonal.
	    for (size_t idx_in_diag = last_idx_in_diag; idx_in_diag < mDiagonalLen; idx_in_diag++) {
	        if (idx_in_diag == mDiagonalLen - 1 && mIsInFirstHalfOfDiagonals)
	            break;
            mDiagonal[idx_in_diag] = computeAtDiagonal(diag_idx, start_row, start_col, idx_in_diag);
        }
    }

    /**
     * @param diag_idx Index of current diagonal.
     * @param start_row Index of row where current diagonal begins.
     * @param start_col Index of column where current diagonal begins.
     * @param idx_in_diagonal Index into the current diagonal.
     * @return Levenshtein distance.
     */
    DIST computeAtDiagonal(size_t diag_idx, size_t start_row, size_t start_col, size_t idx_in_diagonal) const
    {
        size_t total_row = start_row - idx_in_diagonal;
        size_t total_col = start_col + idx_in_diagonal;

        DIST upper = mIsInFirstHalfOfDiagonals ? mLastDiagonal[idx_in_diagonal] : mLastDiagonal[idx_in_diagonal + 1];
        DIST left_upper = mIsInFirstHalfOfDiagonals ? mLastLastDiagonal[idx_in_diagonal - 1] : mLastLastDiagonal[idx_in_diagonal + 1];
        if (isRightAfterHalf(diag_idx))
            left_upper = mLastLastDiagonal[idx_in_diagonal];
        DIST left = mIsInFirstHalfOfDiagonals ? mLastDiagonal[idx_in_diagonal - 1] : mLastDiagonal[idx_in_diagonal];
        DIST a = (*mInputArray1)[total_col - 1];
        DIST b = (*mInputArray2)[total_row - 1];
        if (DEBUG)
            std::cout << "\tComputing diagonal at start_row=" << start_row << ", start_col=" << start_col
                      << ", upper=" << upper << ", left_upper=" << left_upper << ", left=" << left
                      << ", a=" << (char)a << ", b=" << (char)b << std::endl;
        return computeDistance(upper, left_upper, left, a, b);
    }

	DIST computeDistance(DIST upper, DIST left_upper, DIST left, DIST a, DIST b) const
    {
		DIST first = upper + 1;
		DIST second = left_upper + (a == b ? 0 : 1);
		DIST third = left + 1;
		return std::min({first, second, third});
    }

    void prepareDiagonalsForNextIteration(size_t diag_idx)
    {
        // lastLastDiag <-- lastDiag.
        for (size_t k = 0; k < mLastDiagonalLen; ++k) {
            mLastLastDiagonal[k] = mLastDiagonal[k];
        }
        mLastLastDiagonalLen = mLastDiagonalLen;

        // lastDiag <-- diag.
        for (size_t k = 0; k < mDiagonalLen; ++k) {
            mLastDiagonal[k] = mDiagonal[k];
        }
        mLastDiagonalLen = mDiagonalLen;

        // If we iterate through first half of diagonals, we need to set elements at first column and first row.
        if (diag_idx < mTotalRowsCount - 1) {
            mDiagonal[0]++;
            size_t next_diag_len = mDiagonalLen + 1;
            mDiagonal[next_diag_len - 1] = mDiagonal[0];
        }
    }

    /// Returns true if we are currently on a diagonal that is right after the middle diagonal.
    /// This diagonal needs special index handling.
    bool isRightAfterHalf(size_t diag_idx) const
    {
	    // idx(last_diagonal) == rows - 1.
	    return diag_idx == mTotalRowsCount;
    }

    void logDiagonals() const
    {
        std::cout << "diagLen = " << mDiagonalLen << ", lastDiagLen = " << mLastDiagonalLen
                  << ", lastLastDiagonalLen = " << mLastLastDiagonalLen << std::endl;

        std::cout << "lastLastDiagonal: [";
        logVector(mLastLastDiagonal, mLastLastDiagonalLen);
        std::cout << "]" << std::endl;

        std::cout << "LastDiagonal: [";
        logVector(mLastDiagonal, mLastDiagonalLen);
        std::cout << "]" << std::endl;

        std::cout << "Diagonal: [";
        logVector(mDiagonal, mDiagonalLen);
        std::cout << "]" << std::endl;
    }

    template <typename T>
    void logVector(const std::vector<T> &vec, size_t elem_count) const
    {
	    for (size_t i = 0; i < elem_count; i++) {
	        if (i == elem_count - 1)
                std::cout << vec[i];
	        else
                std::cout << vec[i] << ", ";
        }
    }

};


#endif
