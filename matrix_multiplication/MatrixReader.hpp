//
// Created by pal on 17.6.19.
//

#ifndef MATRIX_MULT_MATRIXREADER_HPP
#define MATRIX_MULT_MATRIXREADER_HPP

#include <string>
#include <vector>
#include <fstream>

class MatrixReader {
public:
    explicit MatrixReader(const std::string &file_name);
    size_t getColsCount() const;
    size_t getRowsCount() const;
    std::vector<float> loadStripe(size_t upper_left_row, size_t upper_left_col, size_t width, size_t height);

private:
    std::ifstream mFile;
    size_t mColsCount;
    size_t mRowsCount;
};


#endif //MATRIX_MULT_MATRIXREADER_HPP
