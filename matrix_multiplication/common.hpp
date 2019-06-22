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

/**
 * This message is send by master to workers and it represents two submatrices that should be multiplied on worker.
 */
struct submatrices_message_t {
    int a_row_start;
    /// Index of last row + 1 ie. this "points" behind the end of rows (as in iterator conventions).
    int a_row_end;
    int a_col_start;
    int a_col_end;
    int b_row_start;
    int b_row_end;
    int b_col_start; // 7
    int b_col_end; // 8
    float a_buffer[ROWS_MAX_BLOCK_SIZE * COLS_MAX_BLOCK_SIZE];
    float b_buffer[ROWS_MAX_BLOCK_SIZE * COLS_MAX_BLOCK_SIZE];

    size_t get_a_buffer_size() const
    {
        return (a_row_end - a_row_start) * (a_col_end - a_col_start);
    }

    size_t get_b_buffer_size() const
    {
        return (b_row_end - b_row_start) * (b_col_end - b_col_start);
    }

    bool operator==(const submatrices_message_t &rhs) const
    {
        return a_row_start == rhs.a_row_start &&
               a_row_end == rhs.a_row_end &&
               a_col_start == rhs.a_col_start &&
               a_col_end == rhs.a_col_end &&
               b_row_start == rhs.b_row_start &&
               b_row_end == rhs.b_row_end &&
               b_col_start == rhs.b_col_start &&
               b_col_end == rhs.b_col_end &&
               buffers_equal(a_buffer, rhs.a_buffer, get_a_buffer_size()) &&
               buffers_equal(b_buffer, rhs.b_buffer, get_b_buffer_size());
    }

    bool operator!=(const submatrices_message_t &rhs) const
    {
        return !(rhs == *this);
    }

private:
    bool buffers_equal(const float *buff1, const float *buff2, size_t size) const
    {
        for (size_t i = 0; i < size; ++i)
            if (buff1[i] != buff2[i])
                return false;
        return true;
    }
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

void create_submatrices_message_datatype(MPI_Datatype *submatrices_message_datatype);

#endif //MATRIX_MULT_COMMON_HPP
