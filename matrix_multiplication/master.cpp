//
// Created by pal on 21.6.19.
//

#include "master.hpp"
#include <cassert>
#include <iostream>
#include "common.hpp"
#include "MatrixReader.hpp"

static void send_to_worker(const void *buf, int count, MPI_Datatype datatype, int destination_rank);
static void send_matrices_sizes_to_workers(const matrices_sizes_t &matrices_sizes, int workers_count);
static block_sizes_t determine_block_sizes(size_t rows_count, size_t cols_count);


void master_task(char **argv, int workers_count)
{
    std::string matrix1_filename{argv[1]};
    std::string matrix2_filename{argv[2]};
    std::string result_filename{argv[3]};

    // Root reads header of matrix files and determines sizes.
    MatrixReader matrix1_reader{matrix1_filename};
    MatrixReader matrix2_reader{matrix2_filename};

    matrices_sizes_t matrices_sizes {
            matrix1_reader.getRowsCount(), // a_rows
            matrix1_reader.getColsCount(), // a_cols
            matrix2_reader.getRowsCount(), // b_rows
            matrix2_reader.getColsCount()  // b_cols
    };

    send_matrices_sizes_to_workers(matrices_sizes, workers_count);

    //block_sizes_t block_sizes = determine_block_sizes(result_rows_count, result_cols_count);

    std::cout << "Master: A.rows=" << matrix1_reader.getRowsCount()
              << ", A.cols=" << matrix1_reader.getColsCount()
              << ", B.rows=" << matrix2_reader.getRowsCount()
              << ", B.cols=" << matrix2_reader.getColsCount()
              << std::endl;
}

static void send_to_worker(const void *buf, int count, MPI_Datatype datatype, int destination_rank)
{
    CHECK(MPI_Send(buf, count, datatype, destination_rank, static_cast<int>(Tag::from_master), MPI_COMM_WORLD));
}


static void send_matrices_sizes_to_workers(const matrices_sizes_t &matrices_sizes, int workers_count)
{
    if (DEBUG)
        std::cout << "Master: sending sizes of matrices to all workers." << std::endl;

    for (int worker_rank = 1; worker_rank < workers_count + 1; worker_rank++) {
        send_to_worker(&matrices_sizes.a_rows, 1, MPI_INT, worker_rank);
        send_to_worker(&matrices_sizes.a_cols, 1, MPI_INT, worker_rank);
        send_to_worker(&matrices_sizes.b_rows, 1, MPI_INT, worker_rank);
        send_to_worker(&matrices_sizes.b_cols, 1, MPI_INT, worker_rank);
    }
}

static void send_blocks_to_workers()
{
    
}

static block_sizes_t determine_block_sizes(size_t rows_count, size_t cols_count)
{

}
