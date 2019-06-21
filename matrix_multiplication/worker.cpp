//
// Created by pal on 21.6.19.
//

#include "worker.hpp"
#include "common.hpp"
#include <iostream>

static void receive_from_master(void *buf, int count, MPI_Datatype datatype);
static stripes_t receive_stripes_from_master(const block_sizes_t &block_sizes);
static matrices_sizes_t receive_matrices_sizes_from_master();

void worker_task(int rank)
{
    matrices_sizes_t matrices_sizes = receive_matrices_sizes_from_master();
}

static void receive_from_master(void *buf, int count, MPI_Datatype datatype)
{
    MPI_Status status{};
    CHECK(MPI_Recv(buf, count, datatype, MASTER_RANK, static_cast<int>(Tag::from_master), MPI_COMM_WORLD, &status));
}

static stripes_t receive_stripes_from_master(const block_sizes_t &block_sizes)
{
    // First, row stripe from first matrix is received.
    size_t recv_row_stripe_size = block_sizes.rows_block_size * block_sizes.length;

    // Second, column strip from second matrix is received.
    size_t recv_col_stripe_size = block_sizes.cols_block_size * block_sizes.length;
}

static matrices_sizes_t receive_matrices_sizes_from_master()
{
    int a_rows = 0;
    receive_from_master(&a_rows, 1, MPI_INT);

    if (DEBUG)
        std::cout << "Worker: received from master: a_rows=" << a_rows << std::endl;
}
