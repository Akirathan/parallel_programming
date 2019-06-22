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
    FlatMatrix(const T *buf, size_t rows_count, size_t cols_count) :
        mColsCount{cols_count},
        mRowsCount{rows_count},
        mTotalSize{rows_count * cols_count},
        mContent{buf, buf + mTotalSize}
    {}

    FlatMatrix(const std::vector<T> &vec, size_t rows_count, size_t cols_count) :
        FlatMatrix<T>(&vec[0], rows_count, cols_count)
    {}

    T * getBuffer()
    {
        return mContent;
    }

    const T * getBuffer() const
    {
        return mContent;
    }

    size_t getTotalSize() const
    {
        return mTotalSize;
    }

    T & at(size_t row, size_t col)
    {
        return mContent[getFlatIndex(row, col)];
    }

    const T & at(size_t row, size_t col) const
    {
        return mContent[getFlatIndex(row, col)];
    }

private:
    size_t mColsCount;
    size_t mRowsCount;
    size_t mTotalSize;
    std::vector<T> mContent;

    size_t getFlatIndex(size_t row, size_t col) const
    {
        size_t flat_idx = row * mColsCount + col;
        assert(flat_idx < mTotalSize);
        return flat_idx;
    }
};

#endif //MATRIX_MULT_FLATMATRIX_HPP
