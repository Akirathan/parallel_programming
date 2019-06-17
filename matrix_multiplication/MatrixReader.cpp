//
// Created by pal on 17.6.19.
//

#include "MatrixReader.hpp"
#include <fstream>

MatrixReader::MatrixReader(const std::string &file_name)
    : mFile{file_name},
    mColsCount{0},
    mRowsCount{0}
{
    int32_t cols;
    mFile >> cols;
    mColsCount = static_cast<size_t>(cols);

    int32_t rows;
    mFile >> rows;
    mRowsCount = static_cast<size_t>(rows);
}

size_t MatrixReader::getColsCount() const
{
    return mColsCount;
}

size_t MatrixReader::getRowsCount() const
{
    return mRowsCount;
}
