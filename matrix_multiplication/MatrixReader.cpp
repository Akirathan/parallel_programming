//
// Created by pal on 17.6.19.
//

#include "MatrixReader.hpp"
#include <fstream>

MatrixReader::MatrixReader(const std::string &file_name)
    : mFile{file_name, std::ios::in | std::ios::binary},
    mColsCount{0},
    mRowsCount{0}
{
    int32_t cols;
    mFile.read((char *)&cols, sizeof(int32_t));
    mColsCount = static_cast<size_t>(cols);

    int32_t rows;
    mFile.read((char *)&rows, sizeof(int32_t));
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

FlatMatrix<float> MatrixReader::loadRectangle(size_t upper_left_row, size_t upper_left_col, size_t width, size_t height)
{
    std::vector<float> stripe(width * height);
    size_t stripe_idx = 0;
    for (size_t row = upper_left_row; row < upper_left_row + height; ++row) {
        size_t elems_from_start_of_row = upper_left_col;
        mFile.seekg(elems_from_start_of_row * sizeof(float), std::ios::cur);

        char *buff = (char *)&stripe[stripe_idx];
        mFile.read(buff, width * sizeof(float));

        size_t elems_in_rest_of_row = mColsCount - (upper_left_col + width);
        mFile.seekg(elems_in_rest_of_row * sizeof(float), std::ios::cur);
    }

    return FlatMatrix<float>(stripe, height, width);
}
