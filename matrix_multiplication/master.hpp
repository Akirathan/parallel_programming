//
// Created by pal on 21.6.19.
//

#ifndef MATRIX_MULT_MASTER_HPP
#define MATRIX_MULT_MASTER_HPP

#include "common.hpp"

class Master {
public:
    Master(int workers_count, char **argv);
    void run();

private:
    matrices_sizes_t mMatricesSizes;
    block_sizes_t mBlockSizes;
    int mWorkersCount;

    void sendMatricesSizesToWorkers();
    void sendBlocksToWorkers();
    block_sizes_t determineBlockSizes(size_t a_cols);
    void sendToWorker(const void *buf, int count, MPI_Datatype datatype, int destination_rank) const;
};

#endif //MATRIX_MULT_MASTER_HPP
