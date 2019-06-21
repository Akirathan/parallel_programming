//
// Created by pal on 21.6.19.
//

#ifndef MATRIX_MULT_FLATMATRIX_HPP
#define MATRIX_MULT_FLATMATRIX_HPP

#include <cassert>
#include <vector>

/**
 * Row-based flat matrix. Just a wrapper around buffer with flat-indexing.
 * @tparam T
 */
template <typename T>
class FlatMatrix {
public:
    FlatMatrix(T *buf, size_t rows_count, size_t cols_count)
        : mBuf{buf},
        mColsCount{cols_count},
        mRowsCount{rows_count},
        mTotalSize{rows_count * cols_count}
    {}

    T * getBuffer()
    {
        return mBuf;
    }

    const T * getBuffer() const
    {
        return mBuf;
    }

    size_t getTotalSize() const
    {
        return mTotalSize;
    }

    T & at(size_t row, size_t col)
    {
        return mBuf[getFlatIndex(row, col)];
    }

    const T & at(size_t row, size_t col) const
    {
        return mBuf[getFlatIndex(row, col)];
    }

private:
    T *mBuf;
    size_t mColsCount;
    size_t mRowsCount;
    size_t mTotalSize;

    size_t getFlatIndex(size_t row, size_t col) const
    {
        size_t flat_idx = row * mColsCount + col;
        assert(flat_idx < mTotalSize);
        return flat_idx;
    }
};

#endif //MATRIX_MULT_FLATMATRIX_HPP
