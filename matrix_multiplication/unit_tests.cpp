//
// Created by pal on 21.6.19.
//

#include <cstdio>
#include <functional>
#include <vector>
#include <fstream>
#include <iostream>
#include "exception.hpp"
#include "FlatMatrix.hpp"
#include "MatrixReader.hpp"

using test_t = std::pair<std::string, std::function<void(void)>>;

#define _assert_msg(condition, msg) \
    if (!(condition)) \
        throw RuntimeError() << __FILE__ << ": " << __LINE__ << ": " << msg;

#define _assert(condition) _assert_msg(condition, "")


static void test_flat_matrix()
{
}

static void write_matrix(const std::string &file_name, const std::vector<std::vector<float>> &matrix)
{
    std::ofstream output{file_name, std::ios::binary};
    int32_t rows = matrix.size();
    int32_t cols = matrix[0].size();

    output.write((char *)&cols, sizeof(int32_t));
    output.write((char *)&rows, sizeof(int32_t));
    for (int32_t i = 0; i < rows; ++i) {
        for (int32_t j = 0; j < cols; ++j) {
            output.write((char *)&matrix[i][j], sizeof(float));
        }
    }
}

static void matrix_reader_rectangle_test(const std::vector<std::vector<float>> &matrix, MatrixReader &reader,
                                         size_t upper_left_row, size_t upper_left_col, size_t width, size_t height)
{
    FlatMatrix<float> rectangle = reader.loadRectangle(upper_left_row, upper_left_col, width, height);
    for (size_t rect_i = 0; rect_i < height; rect_i++) {
        for (size_t rect_j = 0; rect_j < width; rect_j++) {
            size_t total_i = rect_i + upper_left_row;
            size_t total_j = rect_j + upper_left_col;
            _assert(rectangle.at(rect_i, rect_j) == matrix[total_i][total_j]);
        }
    }
}

static void test_matrix_reader()
{
    const std::string file_name = "./tmp_matrix.bin";

    std::vector<std::vector<float>> matrix = {
            {1, 2, 3, 23},
            {4, 5, 6, 23},
            {7, 8, 9, 23}
    };
    const size_t rows = matrix.size();
    const size_t cols = matrix[0].size();
    write_matrix(file_name, matrix);

    MatrixReader reader{file_name};

    _assert(reader.getColsCount() == cols);
    _assert(reader.getRowsCount() == rows);
    // Test full matrix load.
    matrix_reader_rectangle_test(matrix, reader, 0, 0, reader.getColsCount(), reader.getRowsCount());

    // Test partial matrix read - read just submatrix.
    matrix_reader_rectangle_test(matrix, reader, 1, 1, 2, 2);
    matrix_reader_rectangle_test(matrix, reader, 1, 2, 2, 1);

    std::remove(file_name.c_str());
}

static void run_one_test(const test_t &test)
{
    std::cout << "==========================" << std::endl;
    std::cout << "Running test: " << test.first << std::endl;
    try {
        test.second();
    }
    catch (StreamException &exception) {
        std::cout << "Test " << test.first << " failed with " << exception.what() << std::endl;
        std::cout << "Exiting ..." << std::endl;
        exit(1);
    }
    std::cout << "Test " << test.first << " passed" << std::endl;
    std::cout << "==========================" << std::endl;
}

static std::vector<test_t> tests = {
        {"Flat matrix tests", test_flat_matrix},
        {"Matrix reader tests", test_matrix_reader}
};

int main()
{
    std::cout << "Running all tests..." << std::endl;
    for (auto &&test : tests) {
        run_one_test(test);
    }
    std::cout << "All tests passed..." << std::endl;
}