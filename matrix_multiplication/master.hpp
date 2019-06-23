//
// Created by pal on 21.6.19.
//

#ifndef MATRIX_MULT_MASTER_HPP
#define MATRIX_MULT_MASTER_HPP

#include "common.hpp"
#include <mpi.h>
#include <vector>
#include "MatrixReader.hpp"

class Master {
public:
    Master(int workers_count, char **argv);
    ~Master();
    void run();

private:
    MatrixReader mMatrix1Reader;
    MatrixReader mMatrix2Reader;
    matrices_sizes_t mMatricesSizes;
    block_sizes_t mBlockSizes;
    int mWorkersCount;
    int mActualWorker;
    std::vector<std::vector<float>> mResultMatrix;
    MPI_Datatype mSubmatricesMessageDatatype;
    MPI_Datatype mResultMessageDatatype;

    void sendMatricesSizesToAllWorkers();
    void sendBlocksToWorkers();
    void receiveResultsFromWorkers();
    block_sizes_t determineBlockSizes(size_t a_cols) const;

    void sendBlocksCorrespondingToResultBlock(size_t res_row_start, size_t res_row_end, size_t res_start_col,
                                              size_t res_end_col);

    void sendToWorker(const void *buf, int count, MPI_Datatype datatype, int destination_rank) const;
    void receiveFromWorker(void *buf, int count, MPI_Datatype datatype, int rank) const;
};

#endif //MATRIX_MULT_MASTER_HPP
