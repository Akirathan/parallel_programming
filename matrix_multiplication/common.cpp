//
// Created by pal on 22.6.19.
//

#include "common.hpp"
#include <iostream>

template <typename T>
static bool buffers_equal(const T *buff1, const T *buff2, size_t size)
{
    for (size_t i = 0; i < size; ++i)
        if (buff1[i] != buff2[i])
            return false;
    return true;
}

void create_submatrices_message_datatype(MPI_Datatype *submatrices_message_datatype)
{
    constexpr int count = 10;
    int block_lengths[count] = {1, 1, 1, 1, 1, 1, 1, 1,
                                ROWS_BLOCK_SIZE * COLS_BLOCK_SIZE, // a_buffer
                                ROWS_BLOCK_SIZE * COLS_BLOCK_SIZE // b_buffer
    };
    MPI_Aint displacements[count] = {0, 4, 8, 12, 16, 20, 24, 28,
                                     32, // a_buffer
                                     32 + (ROWS_BLOCK_SIZE * COLS_BLOCK_SIZE * sizeof(float)) // b_buffer
    };
    MPI_Datatype types[count] = {MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT,
                                 MPI_INT, MPI_INT, MPI_INT, MPI_FLOAT, MPI_FLOAT};
    CHECK(MPI_Type_create_struct(count, block_lengths, displacements, types, submatrices_message_datatype));
    CHECK(MPI_Type_commit(submatrices_message_datatype));

    int type_size = 0;
    CHECK(MPI_Type_size(*submatrices_message_datatype, &type_size));
    if (is_debug_level(DebugLevel::Info))
        std::cout << "Size of submatrices_message type = " << type_size << std::endl;
}

void create_result_message_datatype(MPI_Datatype *result_message_datatype)
{
    constexpr int count = 5;
    int block_lengths[count] = {1, 1, 1, 1, ROWS_BLOCK_SIZE * COLS_BLOCK_SIZE};
    MPI_Aint displacements[count] = {0, 4, 8, 12, 16};
    MPI_Datatype types[count] = {MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_FLOAT};

    CHECK(MPI_Type_create_struct(count, block_lengths, displacements, types, result_message_datatype));
    CHECK(MPI_Type_commit(result_message_datatype));

    int type_size = 0;
    CHECK(MPI_Type_size(*result_message_datatype, &type_size));
    if (is_debug_level(DebugLevel::Info))
        std::cout << "Size of result_message type = " << type_size << std::endl;
}

bool submatrices_message_t::operator==(const submatrices_message_t &rhs) const
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

bool submatrices_message_t::operator!=(const submatrices_message_t &rhs) const
{
    return !(rhs == *this);
}

size_t submatrices_message_t::get_a_buffer_size() const
{
    return (a_row_end - a_row_start) * (a_col_end - a_col_start);
}

size_t submatrices_message_t::get_b_buffer_size() const
{
    return (b_row_end - b_row_start) * (b_col_end - b_col_start);
}

size_t submatrices_message_t::get_a_rows_count() const
{
    return a_row_end - a_row_start;
}

size_t submatrices_message_t::get_a_cols_count() const
{
    return a_col_end - a_col_start;
}

size_t submatrices_message_t::get_b_rows_count() const
{
    return b_row_end - b_row_start;
}

size_t submatrices_message_t::get_b_cols_count() const
{
    return b_col_end - b_col_start;
}

std::ostream &operator<<(std::ostream &os, const submatrices_message_t &message)
{
    os << "a_row_start: " << message.a_row_start << " a_row_end: " << message.a_row_end << " a_col_start: "
       << message.a_col_start << " a_col_end: " << message.a_col_end << " b_row_start: " << message.b_row_start
       << " b_row_end: " << message.b_row_end << " b_col_start: " << message.b_col_start << " b_col_end: "
       << message.b_col_end;

    os << " a_buffer: ";
    if (is_debug_level(DebugLevel::Debug))
        print_buffer(os, message.a_buffer, message.get_a_buffer_size());
    else
        os << message.a_buffer;

    os << " b_buffer: ";
    if (is_debug_level(DebugLevel::Debug))
        print_buffer(os, message.b_buffer, message.get_b_buffer_size());
    else
        os << message.b_buffer;

    return os;
}

size_t result_submatrix_message_t::get_result_rows_count() const
{
    return result_row_end - result_row_start;
}

size_t result_submatrix_message_t::get_result_cols_count() const
{
    return result_col_end - result_col_start;
}

bool result_submatrix_message_t::operator==(const result_submatrix_message_t &rhs) const
{
    return result_row_start == rhs.result_row_start &&
           result_row_end == rhs.result_row_end &&
           result_col_start == rhs.result_col_start &&
           result_col_end == rhs.result_col_end &&
           buffers_equal(result_buffer, rhs.result_buffer, get_result_rows_count() * get_result_cols_count());
}

bool result_submatrix_message_t::operator!=(const result_submatrix_message_t &rhs) const
{
    return !(rhs == *this);
}
