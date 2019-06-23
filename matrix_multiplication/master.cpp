//
// Created by pal on 21.6.19.
//

#include "master.hpp"
#include <cassert>
#include <iostream>
#include "common.hpp"
#include "MatrixReader.hpp"


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

    create_submatrices_message_datatype(&mSubmatricesMessageDatatype);

    if (DEBUG)
        std::cout << "Master: A.rows=" << mMatricesSizes.a_rows
                  << ", A.cols=" << mMatricesSizes.a_cols
                  << ", B.rows=" << mMatricesSizes.b_rows
                  << ", B.cols=" << mMatricesSizes.b_cols
                  << std::endl;
}

Master::~Master()
{
    CHECK(MPI_Type_free(&mSubmatricesMessageDatatype));
}


void Master::run()
{
    sendMatricesSizesToAllWorkers();
    sendBlocksToWorkers();
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
    size_t res_start_row = 0;
    size_t res_end_row = std::min(ROWS_MAX_BLOCK_SIZE, mMatricesSizes.result_rows);
    while (res_start_row < mMatricesSizes.result_rows)
    {
        size_t res_start_col = 0;
        size_t res_end_col = std::min(COLS_MAX_BLOCK_SIZE, mMatricesSizes.result_cols);
        while (res_start_col < mMatricesSizes.result_cols)
        {
            sendBlocksCorrespondingToResultBlock(res_start_row, res_end_row, res_start_col, res_end_col);
            res_start_col = res_end_col;
            res_end_col = std::min(res_end_col + COLS_MAX_BLOCK_SIZE, mMatricesSizes.result_cols);
        }

        res_start_row = res_end_row;
        res_end_row = std::min(res_end_row + ROWS_MAX_BLOCK_SIZE, mMatricesSizes.result_rows);
    }
}

void Master::receiveResultsFromWorkers()
{
    if (DEBUG)
        std::cout << "Master: Start receiving results from all workers..." << std::endl;

    for (int rank = 1; rank < mWorkersCount; rank++) {

    }
}

/**
 * Parameters represent submatrix in result matrix.
 * This method sends parts of stripes from matrix A and B to workers. When there is no worker left to send submatrices
 * to, then we receive result blocks from all the workers.
 * @param res_row_start
 * @param res_row_end
 * @param res_start_col
 * @param res_end_col
 */
void Master::sendBlocksCorrespondingToResultBlock(size_t res_row_start, size_t res_row_end, size_t res_col_start,
                                                  size_t res_col_end)
{
    if (DEBUG)
        std::cout << "Master: Start sending blocks corresponding to result block with: res_row_start=" << res_row_start
                  << ", res_row_end=" << res_row_end << ", res_col_start=" << res_col_start
                  << ", res_col_end=" << res_col_end << std::endl;

    size_t a_block_height = res_row_end - res_row_start;
    const int a_row_start = res_row_start;
    const int a_row_end = res_row_end;
    size_t b_block_width = res_col_end - res_col_start;
    const int b_col_start = res_col_start;
    const int b_col_end = res_col_end;
    assert(a_block_height <= ROWS_MAX_BLOCK_SIZE);
    assert(b_block_width <= COLS_MAX_BLOCK_SIZE);

    assert(mMatricesSizes.a_cols == mMatricesSizes.b_rows);
    for (size_t a_col = 0, b_row = 0; a_col < mMatricesSizes.a_cols && b_row < mMatricesSizes.b_rows;
         a_col = std::min(a_col + COLS_MAX_BLOCK_SIZE, mMatricesSizes.a_cols),
         b_row = std::min(b_row + ROWS_MAX_BLOCK_SIZE, mMatricesSizes.b_rows))
    {
        int a_col_start = a_col;
        int a_col_end = std::min(a_col + COLS_MAX_BLOCK_SIZE, mMatricesSizes.a_cols);
        int a_block_width = a_col_end - a_col_start;
        assert(a_block_width <= static_cast<int>(COLS_MAX_BLOCK_SIZE));

        int b_row_start = b_row;
        int b_row_end = std::min(b_row + ROWS_MAX_BLOCK_SIZE, mMatricesSizes.b_rows);
        int b_block_height = b_row_end - b_row_start;
        assert(b_block_height <= static_cast<int>(ROWS_MAX_BLOCK_SIZE));

        FlatMatrix<float> rectangle_a =
                mMatrix1Reader.loadRectangle(a_row_start, a_col_start, a_block_width, a_block_height);
        FlatMatrix<float> rectangle_b =
                mMatrix2Reader.loadRectangle(b_row_start, b_col_start, b_block_width, b_block_height);

        // Compose message to worker.
        submatrices_message_t message;
        message.a_row_start = a_row_start;
        message.a_row_end = a_row_end;
        message.a_col_start = a_col_start;
        message.a_col_end = a_col_end;
        message.b_row_start = b_row_start;
        message.b_row_end = b_row_end;
        message.b_col_start = b_col_start;
        message.b_col_end = b_col_end;
        std::copy(rectangle_a.getBuffer(), rectangle_a.getBuffer() + rectangle_a.getTotalSize(), message.a_buffer);
        std::copy(rectangle_b.getBuffer(), rectangle_b.getBuffer() + rectangle_b.getTotalSize(), message.b_buffer);
        if (DEBUG)
            std::cout << "Master::sendBlocksCorrespondingToResultBlock: Sending submatrices message "
                      << message << " to worker " << mActualWorker << std::endl;

        // Send to next available worker.
        sendToWorker(&message, 1, mSubmatricesMessageDatatype, mActualWorker);
        if (mActualWorker >= mWorkersCount) {
            receiveResultsFromWorkers();
            mActualWorker = 1;
        }
    }
}

block_sizes_t Master::determineBlockSizes(size_t a_cols) const
{
    return block_sizes_t {
            ROWS_MAX_BLOCK_SIZE, // rows_block_size
            COLS_MAX_BLOCK_SIZE, // cols_block_size
            a_cols// length
    };
}

void Master::sendToWorker(const void *buf, int count, MPI_Datatype datatype, int destination_rank) const
{
    CHECK(MPI_Send(buf, count, datatype, destination_rank, static_cast<int>(Tag::from_master), MPI_COMM_WORLD));
}

void Master::receiveFromWorker(void *buf, int count, MPI_Datatype datatype, int rank) const
{
    MPI_Status status{};
    CHECK(MPI_Recv(buf, count, datatype, rank, static_cast<int>(Tag::from_worker), MPI_COMM_WORLD, &status));

    int element_count = 0;
    CHECK(MPI_Get_elements(&status, datatype, &element_count));
    if (DEBUG)
        std::cout << "Master::receiveFromWorker: MPI_Get_elements = " << element_count << std::endl;
}

