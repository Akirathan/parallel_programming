//
// Created by pal on 21.6.19.
//

#include "worker.hpp"
#include "common.hpp"
#include "FlatMatrix.hpp"
#include <iostream>

static int receive_from_master(void *buf, int max_count, MPI_Datatype datatype);
static matrices_sizes_t receive_matrices_sizes_from_master();

void worker_task(int rank)
{
    matrices_sizes_t matrices_sizes = receive_matrices_sizes_from_master();
}

/**
 * @param buf Pointer to buffer.
 * @param max_count Maximum number of bytes to be received.
 * @param datatype
 * @return Number of bytes received from master
 */
static int receive_from_master(void *buf, int max_count, MPI_Datatype datatype)
{
    MPI_Status status{};
    CHECK(MPI_Recv(buf, max_count, datatype, MASTER_RANK, static_cast<int>(Tag::from_master), MPI_COMM_WORLD, &status));

    int received_count = 0;
    CHECK(MPI_Get_count(&status, datatype, &received_count));
    return received_count;
}

static matrices_sizes_t receive_matrices_sizes_from_master()
{
    int a_rows = 0;
    receive_from_master(&a_rows, 1, MPI_INT);
    int a_cols = 0;
    receive_from_master(&a_cols, 1, MPI_INT);
    int b_rows = 0;
    receive_from_master(&b_rows, 1, MPI_INT);
    int b_cols = 0;
    receive_from_master(&b_cols, 1, MPI_INT);

    if (DEBUG)
        std::cout << "Worker(" << get_rank() << "): received from master: a_rows=" << a_rows << ", a_cols="
                  << a_cols << ", b_rows=" << b_rows << ", b_cols=" << b_cols << std::endl;

    return matrices_sizes_t {
        static_cast<size_t>(a_rows),
        static_cast<size_t>(a_cols),
        static_cast<size_t>(b_rows),
        static_cast<size_t>(b_cols)
    };
}

