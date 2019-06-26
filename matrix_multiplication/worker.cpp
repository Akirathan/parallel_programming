//
// Created by pal on 21.6.19.
//

#include "worker.hpp"
#include "common.hpp"
#include "FlatMatrix.hpp"
#include <iostream>

Worker::Worker(int rank) :
    mRank{rank}
{
    create_submatrices_message_datatype(&mSubmatricesMessageDatatype);
    create_result_message_datatype(&mResultMessageDatatype);
}

Worker::~Worker()
{
    CHECK(MPI_Type_free(&mSubmatricesMessageDatatype));
    CHECK(MPI_Type_free(&mResultMessageDatatype));
}

void Worker::run()
{
    matrices_sizes_t matrices_sizes = receiveMatricesSizes();
    while (true) {
        bool continue_flag = receiveContinueFlag();
        if (!continue_flag)
            break;
        submatrices_message_t submatrices_msg = receiveSubmatrices();
        result_submatrix_message_t result_msg = multiplySubmatrices(submatrices_msg);
        sendResult(result_msg);
    }
    // ====
}

matrices_sizes_t Worker::receiveMatricesSizes() const
{
    int a_rows = 0;
    receiveFromMaster(&a_rows, 1, MPI_INT);
    int a_cols = 0;
    receiveFromMaster(&a_cols, 1, MPI_INT);
    int b_rows = 0;
    receiveFromMaster(&b_rows, 1, MPI_INT);
    int b_cols = 0;
    receiveFromMaster(&b_cols, 1, MPI_INT);

    if (is_debug_level(DebugLevel::Info))
        std::cout << "Worker(" << get_rank() << "): received matrix sizes from master: a_rows=" << a_rows << ", a_cols="
                  << a_cols << ", b_rows=" << b_rows << ", b_cols=" << b_cols << std::endl;

    return matrices_sizes_t {
            static_cast<size_t>(a_rows),
            static_cast<size_t>(a_cols),
            static_cast<size_t>(b_rows),
            static_cast<size_t>(b_cols)
    };
}

bool Worker::receiveContinueFlag() const
{
    bool flag = false;
    receiveFromMaster(&flag, 1, MPI_CHAR);
    return flag;
}

submatrices_message_t Worker::receiveSubmatrices()
{
    submatrices_message_t message{};
    receiveFromMaster(&message, 1, mSubmatricesMessageDatatype);
    if (is_debug_level(DebugLevel::Info))
        std::cout << "Worker (" << mRank << "): Received submatrices from master: " << message << std::endl;
    return message;
}

result_submatrix_message_t Worker::multiplySubmatrices(submatrices_message_t &submatrices)
{
    const size_t result_row_start = submatrices.a_row_start;
    const size_t result_row_end = submatrices.a_row_end;
    const size_t result_col_start = submatrices.b_col_start;
    const size_t result_col_end = submatrices.b_col_end;

    const size_t result_cols = result_col_end - result_col_start;
    const size_t result_rows = result_row_end - result_row_start;

    const size_t a_rows = submatrices.get_a_rows_count();
    const size_t a_cols = submatrices.get_a_cols_count();
    const size_t b_rows = submatrices.get_b_rows_count();
    const size_t b_cols = submatrices.get_b_cols_count();

    assert(submatrices.get_a_cols_count() == submatrices.get_b_rows_count());

    FlatMatrix<float> a_matrix{submatrices.a_buffer, a_rows, a_cols};
    FlatMatrix<float> b_matrix{submatrices.b_buffer, b_rows, b_cols};
    mResultBuffer.resize(result_rows * result_cols);
    FlatMatrix<float> res_matrix{&mResultBuffer[0], result_rows, result_cols};

    for (size_t result_row = 0; result_row < result_rows; result_row++) {
        for (size_t result_col = 0; result_col < result_cols; result_col++) {
            float sum = 0;
            size_t a_row = result_row;
            size_t b_col = result_col;
            // Iterate over one row in A and one column in B.
            for (size_t a_col = 0, b_row = 0;
                 a_col < a_cols && b_row < b_rows;
                 a_col++, b_row++)
            {
                float a_elem = a_matrix.at(a_row, a_col);
                float b_elem = b_matrix.at(b_row, b_col);
                sum += (a_elem * b_elem);
            }
            res_matrix.at(result_row, result_col) = sum;
        }
    }

    result_submatrix_message_t message;
    message.result_row_start = result_row_start;
    message.result_row_end = result_row_end;
    message.result_col_start = result_col_start;
    message.result_col_end = result_col_end;
    std::copy(mResultBuffer.begin(), mResultBuffer.end(), message.result_buffer);
    return message;
}

void Worker::sendResult(result_submatrix_message_t &message)
{
    sendToMaster(&message, 1, mResultMessageDatatype);
}

/**
 * @param buf Pointer to buffer.
 * @param max_count Maximum number of bytes to be received.
 * @param datatype
 * @return Number of bytes received from master
 */
int Worker::receiveFromMaster(void *buf, int max_count, MPI_Datatype datatype) const
{
    MPI_Status status{};
    CHECK(MPI_Recv(buf, max_count, datatype, MASTER_RANK, static_cast<int>(Tag::from_master), MPI_COMM_WORLD, &status));

    int received_count = 0;
    CHECK(MPI_Get_count(&status, datatype, &received_count));
    return received_count;
}

void Worker::sendToMaster(const void *buf, int count, MPI_Datatype datatype) const
{
    CHECK(MPI_Send(buf, count, datatype, MASTER_RANK, static_cast<int>(Tag::from_worker), MPI_COMM_WORLD));
}

