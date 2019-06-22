//
// Created by pal on 21.6.19.
//

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
        throw RuntimeError() << __FILE__ << ": " << __LINE__ << ": " << msg

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

static void test_matrix_reader()
{
    const std::string file_name = "/home/pal/dev/parallel_programming/matrix_multiplication/exec/matrix";

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
    FlatMatrix<float> read_matrix = reader.loadRectangle(0, 0, reader.getColsCount(), reader.getRowsCount());
    for (size_t i = 0; i < reader.getRowsCount(); ++i)
        for (size_t j = 0; j < reader.getColsCount(); ++j)
            _assert(read_matrix.at(i, j) == matrix[i][j]);
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