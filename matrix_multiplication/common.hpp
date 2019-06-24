//
// Created by pal on 21.6.19.
//

#ifndef MATRIX_MULT_COMMON_HPP
#define MATRIX_MULT_COMMON_HPP

#include <mpi.h>
#include <vector>
#include <ostream>
#include "exception.hpp"

enum class DebugLevel {
    Debug,
    Info,
    None
};


constexpr DebugLevel DEBUG = DebugLevel::Info;
constexpr int MASTER_RANK = 0;
constexpr size_t ROWS_BLOCK_SIZE = 32;
constexpr size_t COLS_BLOCK_SIZE = 32;

inline bool is_debug_level(DebugLevel debug_level)
{
    if (DEBUG == DebugLevel::None)
        return false;
    else if (DEBUG == DebugLevel::Debug)
        return true;
    else if (DEBUG == DebugLevel::Info) {
        if (debug_level == DebugLevel::Info)
            return true;
        else
            return false;
    }
}

enum class Tag {
    from_master = 1,
    from_worker = 2
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
    float a_buffer[ROWS_BLOCK_SIZE * COLS_BLOCK_SIZE];
    float b_buffer[ROWS_BLOCK_SIZE * COLS_BLOCK_SIZE];

    size_t get_a_buffer_size() const;
    size_t get_b_buffer_size() const;
    size_t get_a_rows_count() const;
    size_t get_a_cols_count() const;
    size_t get_b_rows_count() const;
    size_t get_b_cols_count() const;
    bool operator==(const submatrices_message_t &rhs) const;
    bool operator!=(const submatrices_message_t &rhs) const;
    friend std::ostream &operator<<(std::ostream &os, const submatrices_message_t &message);
};

struct result_submatrix_message_t {
    int result_row_start;
    int result_row_end;
    int result_col_start;
    int result_col_end;
    float result_buffer[ROWS_BLOCK_SIZE * COLS_BLOCK_SIZE];

    size_t get_result_rows_count() const;
    size_t get_result_cols_count() const;
    bool operator==(const result_submatrix_message_t &rhs) const;
    bool operator!=(const result_submatrix_message_t &rhs) const;
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

template <typename T>
inline void print_buffer(std::ostream &os, const T *buffer, size_t size)
{
    os << "[";
    for (size_t i = 0; i < size; i++) {
        if (i != size - 1)
            os << buffer[i] << ", ";
        else
            os << buffer[i] << "]";
    }
}

void create_submatrices_message_datatype(MPI_Datatype *submatrices_message_datatype);
void create_result_message_datatype(MPI_Datatype *result_message_datatype);

#endif //MATRIX_MULT_COMMON_HPP
