//
// Created by pal on 17.6.19.
//

#include <iostream>
#include <string>
#include <vector>
#include <mpi.h>
#include "exception.hpp"
#include "MatrixReader.hpp"


static void _mpi_check(int err, int line, const char *src_file, const char *err_msg = nullptr)
{
    if (err != 0)
        throw RuntimeError() << "MPI Error (" << err << ")\n"
            << "at " << src_file << "[" << line << "]: " << err_msg;
}

#define CHECK(cmd)  _mpi_check(cmd, __LINE__, __FILE__, #cmd)

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

struct stripes_t {
    std::vector<std::vector<float>> row_stripe;
    std::vector<std::vector<float>> column_stripe;
};

struct block_sizes_t {
    /// Size of rows block from first matrix.
    size_t rows_block_size = 0;
    /// Size of columns block from second matrix.
    size_t cols_block_size = 0;
    /// Length of the block.
    size_t length = 0;
};

static stripes_t receive_stripes_from_root(const block_sizes_t &block_sizes)
{
    // First, row stripe from first matrix is received.
    size_t recv_row_stripe_size = block_sizes.rows_block_size * block_sizes.length;

    // Second, column strip from second matrix is received.
    size_t recv_col_stripe_size = block_sizes.cols_block_size * block_sizes.length;
}

static block_sizes_t determine_block_sizes(size_t rows_count, size_t cols_count)
{

}

int main(int argc, char **argv)
{
    if (argc != 4)
        usage(argv);

    CHECK(MPI_Init(&argc, &argv));

    std::string matrix1_filename{argv[1]};
    std::string matrix2_filename{argv[2]};
    std::string result_filename{argv[3]};

    // Root reads header of matrix files and determines sizes.
    MatrixReader matrix1_reader{matrix1_filename};
    MatrixReader matrix2_reader{matrix2_filename};
    size_t result_rows_count = matrix1_reader.getRowsCount();
    size_t result_cols_count = matrix2_reader.getColsCount();
    block_sizes_t block_sizes = determine_block_sizes(result_rows_count, result_cols_count);

    std::cout << "A.rows=" << matrix1_reader.getRowsCount() << ", A.cols=" << matrix1_reader.getColsCount()
              << ", B.rows=" << matrix2_reader.getRowsCount() << ", B.cols=" << matrix2_reader.getColsCount()
              << std::endl;

    stripes_t stripes = receive_stripes_from_root(block_sizes);

    CHECK(MPI_Finalize());
}