//
// Created by pal on 21.6.19.
//

#ifndef MATRIX_MULT_WORKER_HPP
#define MATRIX_MULT_WORKER_HPP

#include <vector>
#include <mpi.h>
#include "common.hpp"

class Worker {
public:
    Worker(int rank);
    ~Worker();
    void run();

private:
    int mRank;
    std::vector<float> mResultBuffer;
    MPI_Datatype mSubmatricesMessageDatatype;
    MPI_Datatype mResultMessageDatatype;

    matrices_sizes_t receiveMatricesSizes() const;
    bool receiveContinueFlag() const;
    submatrices_message_t receiveSubmatrices();
    result_submatrix_message_t multiplySubmatrices(submatrices_message_t &submatrices);
    void sendResult(result_submatrix_message_t &message);
    int receiveFromMaster(void *buf, int max_count, MPI_Datatype datatype) const;
    void sendToMaster(const void *buf, int count, MPI_Datatype datatype) const;
};

#endif //MATRIX_MULT_WORKER_HPP
