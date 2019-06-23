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
    int mWorkersCount;
    int mActualWorker;
    std::vector<std::vector<float>> mResultMatrix;
    std::string mResultFilename;
    MPI_Datatype mSubmatricesMessageDatatype;
    MPI_Datatype mResultMessageDatatype;

    void sendMatricesSizesToAllWorkers();
    void sendAllBlocksAndReceiveResults();
    void sendBlocksOfStripesAndReceiveResults(size_t res_row_start, size_t res_row_end, size_t res_col_start,
                                              size_t res_col_end);
    void sendContinueFlagToWorker(int worker_rank, bool cont) const;
    void sendSubmatrixToWorker(int worker_rank, int a_row_start, int a_row_end, int a_col_start, int a_col_end,
                               int b_row_start, int b_row_end, int b_col_start, int b_col_end);
    void receiveResultsFromAllWorkers();
    void receiveResultsFromWorker(int rank);
    void sendToWorker(const void *buf, int count, MPI_Datatype datatype, int destination_rank) const;
    void receiveFromWorker(void *buf, int count, MPI_Datatype datatype, int rank) const;
    void writeResultMatrixToFile() const;
};

#endif //MATRIX_MULT_MASTER_HPP
