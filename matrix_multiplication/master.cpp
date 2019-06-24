//
// Created by pal on 21.6.19.
//

#include "master.hpp"
#include <mpi.h>
#include <cassert>
#include <iostream>
#include "common.hpp"
#include "MatrixReader.hpp"


Master::Master(int workers_count, char **argv) :
        mMatrix1Reader{argv[1]},
        mMatrix2Reader{argv[2]},
        mMatricesSizes{0, 0, 0, 0},
        mWorkersCount{workers_count},
        mActualWorker{1},
        mResultFilename{argv[3]}
{
    mMatricesSizes = {
            mMatrix1Reader.getRowsCount(), // a_rows
            mMatrix1Reader.getColsCount(), // a_cols
            mMatrix2Reader.getRowsCount(), // b_rows
            mMatrix2Reader.getColsCount()  // b_cols
    };
    assert(mMatricesSizes.a_cols == mMatricesSizes.b_rows);

    mResultMatrix.resize(mMatricesSizes.result_rows);
    for (auto &&row : mResultMatrix) {
        row.resize(mMatricesSizes.result_cols);
    }

    create_submatrices_message_datatype(&mSubmatricesMessageDatatype);
    create_result_message_datatype(&mResultMessageDatatype);

    if (is_debug_level(DebugLevel::Info))
        std::cout << "Master: A.rows=" << mMatricesSizes.a_rows
                  << ", A.cols=" << mMatricesSizes.a_cols
                  << ", B.rows=" << mMatricesSizes.b_rows
                  << ", B.cols=" << mMatricesSizes.b_cols
                  << std::endl;
}

Master::~Master()
{
    CHECK(MPI_Type_free(&mSubmatricesMessageDatatype));
    CHECK(MPI_Type_free(&mResultMessageDatatype));
}

void Master::run()
{
    sendMatricesSizesToAllWorkers();
    sendAllBlocksAndReceiveResults();

    // Receive rest of results from some workers.
    for (int worker_rank = 1; worker_rank < mActualWorker; worker_rank++)
        receiveResultsFromWorker(worker_rank);

    // Signal all workers that computation is done.
    for (int worker_rank = 1; worker_rank <= mWorkersCount; worker_rank++)
        sendContinueFlagToWorker(worker_rank, false);

    writeResultMatrixToFile();
}

void Master::sendMatricesSizesToAllWorkers()
{
    if (is_debug_level(DebugLevel::Info))
        std::cout << "Master: Sending sizes of matrices to all workers." << std::endl;

    for (int worker_rank = 1; worker_rank <= mWorkersCount; worker_rank++) {
        sendToWorker(&mMatricesSizes.a_rows, 1, MPI_INT, worker_rank);
        sendToWorker(&mMatricesSizes.a_cols, 1, MPI_INT, worker_rank);
        sendToWorker(&mMatricesSizes.b_rows, 1, MPI_INT, worker_rank);
        sendToWorker(&mMatricesSizes.b_cols, 1, MPI_INT, worker_rank);
    }

    if (is_debug_level(DebugLevel::Info))
        std::cout << "Master: Sizes of all matrices sent to all workers." << std::endl;
}

/**
 * Iterates over blocks of result matrix.
 */
void Master::sendAllBlocksAndReceiveResults()
{
    size_t res_start_row = 0;
    size_t res_end_row = std::min(ROWS_BLOCK_SIZE, mMatricesSizes.result_rows);
    while (res_start_row < mMatricesSizes.result_rows)
    {
        size_t res_start_col = 0;
        size_t res_end_col = std::min(COLS_BLOCK_SIZE, mMatricesSizes.result_cols);
        while (res_start_col < mMatricesSizes.result_cols)
        {
            sendBlocksOfStripesAndReceiveResults(res_start_row, res_end_row, res_start_col, res_end_col);
            res_start_col = res_end_col;
            res_end_col = std::min(res_end_col + COLS_BLOCK_SIZE, mMatricesSizes.result_cols);
        }

        res_start_row = res_end_row;
        res_end_row = std::min(res_end_row + ROWS_BLOCK_SIZE, mMatricesSizes.result_rows);
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
void Master::sendBlocksOfStripesAndReceiveResults(size_t res_row_start, size_t res_row_end, size_t res_col_start,
                                                  size_t res_col_end)
{
    if (is_debug_level(DebugLevel::Info))
        std::cout << "Master: Start sending blocks corresponding to result block with: res_row_start=" << res_row_start
                  << ", res_row_end=" << res_row_end << ", res_col_start=" << res_col_start
                  << ", res_col_end=" << res_col_end << std::endl;

    const int a_row_start = res_row_start;
    const int a_row_end = res_row_end;
    const int b_col_start = res_col_start;
    const int b_col_end = res_col_end;

    for (size_t a_col = 0, b_row = 0; a_col < mMatricesSizes.a_cols && b_row < mMatricesSizes.b_rows;
         a_col = std::min(a_col + COLS_BLOCK_SIZE, mMatricesSizes.a_cols),
         b_row = std::min(b_row + ROWS_BLOCK_SIZE, mMatricesSizes.b_rows))
    {
        int a_col_start = a_col;
        int a_col_end = std::min(a_col + COLS_BLOCK_SIZE, mMatricesSizes.a_cols);
        int b_row_start = b_row;
        int b_row_end = std::min(b_row + ROWS_BLOCK_SIZE, mMatricesSizes.b_rows);

        sendContinueFlagToWorker(mActualWorker, true);
        sendSubmatrixToWorker(mActualWorker, a_row_start, a_row_end, a_col_start, a_col_end,
                              b_row_start, b_row_end, b_col_start, b_col_end);

        mActualWorker++;
        if (mActualWorker > mWorkersCount) {
            receiveResultsFromAllWorkers();
            mActualWorker = 1;
        }
    }
}

void Master::sendContinueFlagToWorker(int worker_rank, bool cont) const
{
    if (is_debug_level(DebugLevel::Info))
        std::cout << "Master: Sending continue flag (" << cont << ") to worker " << worker_rank << std::endl;

    sendToWorker(&cont, 1, MPI_CHAR, worker_rank);
}

void Master::sendSubmatrixToWorker(int worker_rank, int a_row_start, int a_row_end, int a_col_start, int a_col_end,
                                   int b_row_start, int b_row_end, int b_col_start, int b_col_end)
{
    const size_t a_block_height = a_row_end - a_row_start;
    const size_t a_block_width = a_col_end - a_col_start;
    assert(a_block_height <= ROWS_BLOCK_SIZE);
    assert(a_block_width <= COLS_BLOCK_SIZE);

    const size_t b_block_height = b_row_end - b_row_start;
    const size_t b_block_width = b_col_end - b_col_start;
    assert(b_block_height <= ROWS_BLOCK_SIZE);
    assert(b_block_width <= COLS_BLOCK_SIZE);

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

    if (is_debug_level(DebugLevel::Info))
        std::cout << "Master: Sending submatrices: " << message << " to worker: " << worker_rank << std::endl;

    sendToWorker(&message, 1, mSubmatricesMessageDatatype, mActualWorker);
}

void Master::receiveResultsFromAllWorkers()
{
    if (is_debug_level(DebugLevel::Info))
        std::cout << "Master: Start receiving results from all workers..." << std::endl;

    for (int rank = 1; rank <= mWorkersCount; rank++)
        receiveResultsFromWorker(rank);
}

void Master::receiveResultsFromWorker(int rank)
{
    result_submatrix_message_t message{};
    receiveFromWorker(&message, 1, mResultMessageDatatype, rank);

    FlatMatrix<float> received_matrix{message.result_buffer, message.get_result_rows_count(),
                                      message.get_result_cols_count()};
    if (is_debug_level(DebugLevel::Info))
        std::cout << "Master: From worker " << rank << " received result: " << received_matrix << std::endl;

    for (int result_row = message.result_row_start, received_matrix_row = 0;
         result_row < message.result_row_end;
         result_row++, received_matrix_row++)
    {
        for (int result_col = message.result_col_start, received_matrix_col = 0;
             result_col < message.result_col_end;
             result_col++, received_matrix_col++)
        {
            mResultMatrix[result_row][result_col] += received_matrix.at(received_matrix_row, received_matrix_col);
        }
    }
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
    if (is_debug_level(DebugLevel::Info))
        std::cout << "Master::receiveFromWorker: MPI_Get_elements = " << element_count << std::endl;
}

void Master::writeResultMatrixToFile() const
{
    std::ofstream output{mResultFilename, std::ios::binary};

    if (is_debug_level(DebugLevel::Info)) {
        std::cout << "Master: Writing matrix to filename" << mResultFilename << ", matrix:\n\t";
        for (auto &&row : mResultMatrix) {
            for (auto &&item : row)
                std::cout << item << ", ";
            std::cout << "\n\t";
        }
    }

    output.write((char *)&mMatricesSizes.result_cols, 4);
    output.write((char *)&mMatricesSizes.result_rows, 4);

    for (auto &&row : mResultMatrix)
        output.write((char *)&row[0], row.size() * sizeof(float));
}

