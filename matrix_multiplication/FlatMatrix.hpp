//
// Created by pal on 21.6.19.
//

#ifndef MATRIX_MULT_FLATMATRIX_HPP
#define MATRIX_MULT_FLATMATRIX_HPP

#include <cassert>
#include <vector>
#include <ostream>

/**
 * Row-based flat matrix. Just a wrapper around buffer with flat-indexing.
 * @tparam T
 */
template <typename T>
class FlatMatrix {
public:
    /// Does not copy the content.
    FlatMatrix(T *buf, size_t rows_count, size_t cols_count) :
        mColsCount{cols_count},
        mRowsCount{rows_count},
        mTotalSize{rows_count * cols_count},
        mBuff{buf},
        mAllocated{false}
    {}

    /// Copies the contents of the vector.
    FlatMatrix(const std::vector<T> &vec, size_t rows_count, size_t cols_count) :
        mColsCount{cols_count},
        mRowsCount{rows_count},
        mTotalSize{rows_count * cols_count},
        mBuff{nullptr},
        mAllocated{true}
    {
        mBuff = new T[mTotalSize];
        std::copy(vec.begin(), vec.end(), mBuff);
    }

    ~FlatMatrix()
    {
        if (mAllocated)
            delete[] mBuff;
    }

    T * getBuffer()
    {
        return mBuff;
    }

    const T * getBuffer() const
    {
        return mBuff;
    }

    size_t getTotalSize() const
    {
        return mTotalSize;
    }

    T & at(size_t row, size_t col)
    {
        return mBuff[getFlatIndex(row, col)];
    }

    const T & at(size_t row, size_t col) const
    {
        return mBuff[getFlatIndex(row, col)];
    }

    friend std::ostream &operator<<(std::ostream &os, const FlatMatrix &matrix)
    {
        os << "mColsCount: " << matrix.mColsCount << " mRowsCount: " << matrix.mRowsCount << " mTotalSize: "
           << matrix.mTotalSize << " mBuff: " << matrix.mBuff << " mAllocated: " << matrix.mAllocated;
        return os;
    }

private:
    size_t mColsCount;
    size_t mRowsCount;
    size_t mTotalSize;
    T *mBuff;
    bool mAllocated;

    size_t getFlatIndex(size_t row, size_t col) const
    {
        size_t flat_idx = row * mColsCount + col;
        assert(flat_idx < mTotalSize);
        return flat_idx;
    }
};

#endif //MATRIX_MULT_FLATMATRIX_HPP
