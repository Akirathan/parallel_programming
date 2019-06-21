//
// Created by pal on 21.6.19.
//

#include "master.hpp"
#include <cassert>
#include <iostream>
#include "common.hpp"
#include "MatrixReader.hpp"

static constexpr size_t ROWS_BLOCK_SIZE = 32;
static constexpr size_t COLS_BLOCK_SIZE = 32;

Master::Master(int workers_count, char **argv) :
        mMatrix1Reader{argv[1]},
        mMatrix2Reader{argv[2]},
        mMatricesSizes{0, 0, 0, 0},
        mWorkersCount{workers_count},
        mActualWorker{1}
{
    std::string result_filename{argv[3]};

    mMatricesSizes = {
            mMatrix1Reader.getRowsCount(), // a_rows
            mMatrix1Reader.getColsCount(), // a_cols
            mMatrix2Reader.getRowsCount(), // b_rows
            mMatrix2Reader.getColsCount()  // b_cols
    };

    mBlockSizes = determineBlockSizes(mMatricesSizes.a_cols);

    if (DEBUG)
        std::cout << "Master: A.rows=" << mMatricesSizes.a_rows
                  << ", A.cols=" << mMatricesSizes.a_cols
                  << ", B.rows=" << mMatricesSizes.b_rows
                  << ", B.cols=" << mMatricesSizes.b_cols
                  << std::endl;
}

void Master::run()
{
    sendMatricesSizesToAllWorkers();
}

void Master::sendMatricesSizesToAllWorkers()
{
    if (DEBUG)
        std::cout << "Master: sending sizes of matrices to all workers." << std::endl;

    for (int worker_rank = 1; worker_rank < mWorkersCount + 1; worker_rank++) {
        sendToWorker(&mMatricesSizes.a_rows, 1, MPI_INT, worker_rank);
        sendToWorker(&mMatricesSizes.a_cols, 1, MPI_INT, worker_rank);
        sendToWorker(&mMatricesSizes.b_rows, 1, MPI_INT, worker_rank);
        sendToWorker(&mMatricesSizes.b_cols, 1, MPI_INT, worker_rank);
    }
}

void Master::sendBlocksToWorkers()
{
    size_t start_row = 0;
    size_t end_row = std::min(start_row + ROWS_BLOCK_SIZE, mMatricesSizes.result_rows);
    while (start_row != mMatricesSizes.result_rows) {

    }
}

block_sizes_t Master::determineBlockSizes(size_t a_cols) const
{
    return block_sizes_t {
            ROWS_BLOCK_SIZE, // rows_block_size
            COLS_BLOCK_SIZE, // cols_block_size
            a_cols// length
    };
}

void Master::sendToWorker(const void *buf, int count, MPI_Datatype datatype, int destination_rank) const
{
    CHECK(MPI_Send(buf, count, datatype, destination_rank, static_cast<int>(Tag::from_master), MPI_COMM_WORLD));
}

