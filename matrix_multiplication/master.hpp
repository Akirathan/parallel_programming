//
// Created by pal on 21.6.19.
//

#ifndef MATRIX_MULT_MASTER_HPP
#define MATRIX_MULT_MASTER_HPP

#include "common.hpp"
#include "MatrixReader.hpp"

class Master {
public:
    Master(int workers_count, char **argv);
    void run();

private:
    MatrixReader mMatrix1Reader;
    MatrixReader mMatrix2Reader;
    matrices_sizes_t mMatricesSizes;
    block_sizes_t mBlockSizes;
    int mWorkersCount;
    int mActualWorker;

    void sendMatricesSizesToAllWorkers();
    void sendBlocksToWorkers();
    block_sizes_t determineBlockSizes(size_t a_cols) const;
    void sendToWorker(const void *buf, int count, MPI_Datatype datatype, int destination_rank) const;

    void sendBlocksCorrespondingToResultBlock(size_t res_start_row, size_t res_end_row, size_t res_start_col,
                                              size_t res_end_col);
};

#endif //MATRIX_MULT_MASTER_HPP
