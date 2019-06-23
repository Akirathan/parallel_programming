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

static void _main(int argc, char **argv)
{
    CHECK(MPI_Init(&argc, &argv));

    if (get_rank() == 0) {
        int arr[3] = {1, 2, 3};
        CHECK(MPI_Send(arr, 3 * sizeof(int), MPI_INT, 1, (int)Tag::from_master, MPI_COMM_WORLD));
    }
    else if (get_rank() == 1) {
        constexpr int MAX = 50;
        int input_buff[MAX] = {};

        MPI_Status status{};
        MPI_Recv(input_buff, MAX, MPI_INT, 0, (int)Tag::from_master, MPI_COMM_WORLD, &status);

        int received_count = 0;
        MPI_Get_count(&status, MPI_INT, &received_count);

        std::cout << "Real received number = " << received_count << std::endl;
    }

    CHECK(MPI_Finalize());
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

    if (rank == 0) {
        Master master{workers_count, argv};
        master.run();
    }
    else {
        Worker worker{rank};
        worker.run();
    }

    CHECK(MPI_Finalize());
}