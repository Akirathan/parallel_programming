//
// Created by pal on 17.6.19.
//

#include <cassert>
#include <iostream>
#include <string>
#include <vector>
#include <mpi.h>
#include "exception.hpp"
#include "common.hpp"
#include "MatrixReader.hpp"
#include "master.hpp"
#include "worker.hpp"


static void usage(char **argv)
{
    std::cout << "Usage: " << std::endl;
    std::cout << argv[0] << " <matrix_filename1> <matrix_filename2> <result_filename>" << std::endl;
    std::cerr << "Exiting..." << std::endl;
    exit(1);
}

int main(int argc, char **argv)
{
    if (argc != 4)
        usage(argv);

    CHECK(MPI_Init(&argc, &argv));

    int tasks_count = 0;
    CHECK(MPI_Comm_size(MPI_COMM_WORLD, &tasks_count));
    if (tasks_count < 2) {
        std::cout << "Need at least 2 MPI tasks, exiting..." << std::endl;
        exit(1);
    }

    int workers_count = tasks_count - 1;
    int rank = get_rank();

    if (rank == 0)
        master_task(argv, workers_count);
    else
        worker_task(rank);

    CHECK(MPI_Finalize());
}