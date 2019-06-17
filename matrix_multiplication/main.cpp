//
// Created by pal on 17.6.19.
//

#include <iostream>
#include <string>
#include "exception.hpp"
#include <mpi.h>

static void _mpi_check(int err, int line, const char *src_file, const char *err_msg = nullptr)
{
    if (err != 0)
        throw RuntimeError() << "MPI Error (" << err << ")\n"
            << "at " << src_file << "[" << line << "]: " << err_msg;
}

#define CHECK(cmd) _mpi_check(cmd, __LINE__, __FILE__, #cmd)

static void usage(char **argv)
{
    std::cout << "Usage: " << std::endl;
    std::cout << argv[0] << " <matrix_filename1> <matrix_filename2> <result_filename>" << std::endl;
    std::cerr << "Exiting..." << std::endl;
    exit(1);
}

static int get_rank()
{
    int rank = 0;
    CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    return rank;
}

int main(int argc, char **argv)
{
    if (argc != 4)
        usage(argv);

    CHECK(MPI_Init(&argc, &argv));

    std::string matrix1_filename{argv[1]};
    std::string matrix2_filename{argv[2]};
    std::string result_filename{argv[3]};

    CHECK(MPI_Finalize());
}