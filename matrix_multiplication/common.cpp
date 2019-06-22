//
// Created by pal on 22.6.19.
//

#include "common.hpp"
#include <iostream>

void create_submatrices_message_datatype(MPI_Datatype *submatrices_message_datatype)
{
    constexpr int count = 10;
    int block_lengths[count] = {1, 1, 1, 1, 1, 1, 1, 1, ROWS_MAX_BLOCK_SIZE * COLS_MAX_BLOCK_SIZE,
                                ROWS_MAX_BLOCK_SIZE * COLS_MAX_BLOCK_SIZE};
    MPI_Aint displacements[count] = {0, 4, 8, 12, 16, 20, 24, 28, 32, 32 + ROWS_MAX_BLOCK_SIZE * COLS_MAX_BLOCK_SIZE};
    MPI_Datatype types[count] = {MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT,
                                 MPI_INT, MPI_INT, MPI_INT, MPI_FLOAT, MPI_FLOAT};
    CHECK(MPI_Type_create_struct(count, block_lengths, displacements, types, submatrices_message_datatype));
    CHECK(MPI_Type_commit(submatrices_message_datatype));

    int type_size = 0;
    CHECK(MPI_Type_size(*submatrices_message_datatype, &type_size));
    if (DEBUG)
        std::cout << "Size of submatrices_message type = " << type_size << std::endl;
}
