//
// Created by pal on 21.6.19.
//

#ifndef MATRIX_MULT_COMMON_HPP
#define MATRIX_MULT_COMMON_HPP

#include <mpi.h>
#include <vector>
#include "exception.hpp"

constexpr int MASTER_RANK = 0;
constexpr bool DEBUG = true;
constexpr size_t ROWS_MAX_BLOCK_SIZE = 32;
constexpr size_t COLS_MAX_BLOCK_SIZE = 32;

enum class Tag {
    from_master = 1,
    from_worker = 2
};

struct stripes_t {
    std::vector<std::vector<float>> row_stripe;
    std::vector<std::vector<float>> column_stripe;
};

struct block_sizes_t {
    /// Size of rows block from first matrix.
    size_t rows_block_size = 0;
    /// Size of columns block from second matrix.
    size_t cols_block_size = 0;
    /// Length of the block.
    size_t length = 0;
};

struct matrices_sizes_t {
    size_t a_rows;
    size_t a_cols;
    size_t b_rows;
    size_t b_cols;
    size_t result_rows;
    size_t result_cols;

    matrices_sizes_t(size_t a_rows, size_t a_cols, size_t b_rows, size_t b_cols)
        : a_rows{a_rows},
        a_cols{a_cols},
        b_rows{b_rows},
        b_cols{b_cols},
        result_rows{a_rows},
        result_cols{b_cols}
    {}
};

inline void _mpi_check(int err, int line, const char *src_file, const char *err_msg = nullptr)
{
    if (err != 0)
        throw RuntimeError() << "MPI Error (" << err << ")\n"
                             << "at " << src_file << "[" << line << "]: " << err_msg;
}

#define CHECK(cmd)  _mpi_check(cmd, __LINE__, __FILE__, #cmd)

inline int get_rank()
{
    int rank = 0;
    CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    return rank;
}

#endif //MATRIX_MULT_COMMON_HPP
