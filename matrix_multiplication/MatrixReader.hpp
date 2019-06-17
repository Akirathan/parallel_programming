//
// Created by pal on 17.6.19.
//

#ifndef MATRIX_MULT_MATRIXREADER_HPP
#define MATRIX_MULT_MATRIXREADER_HPP

#include <string>
#include <vector>

class MatrixReader {
public:
    MatrixReader(const std::string &file_name);
    std::vector<float> loadStripe(size_t upper_left_row, size_t upper_left_col, size_t width, size_t height);

private:
    size_t mColsCount;
    size_t mRowsCount;
};


#endif //MATRIX_MULT_MATRIXREADER_HPP
