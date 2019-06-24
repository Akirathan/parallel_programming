//
// Created by pal on 21.6.19.
//

#include <cstdio>
#include <cstring>
#include <functional>
#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <unistd.h>
#include "exception.hpp"
#include "FlatMatrix.hpp"
#include "MatrixReader.hpp"

#define REMOTE
#ifdef REMOTE
#include <mpi.h>
#include "common.hpp"
#endif // REMOTE

using test_t = std::pair<std::string, std::function<void(void)>>;

#define _assert_msg(condition, msg) \
    if (!(condition)) \
        throw RuntimeError() << __FILE__ << ": " << __LINE__ << ": " << msg;

#define _assert(condition) _assert_msg(condition, "")


static bool is_local()
{
    char hostname[50] = {};
    int err = gethostname(hostname, 50);
    _assert_msg(err == 0, "gethostname call failed");
    return std::strcmp(hostname, "mayfa-PC") == 0;
}

static void test_flat_matrix()
{
}

static void write_matrix(const std::string &file_name, const std::vector<std::vector<float>> &matrix)
{
    std::ofstream output{file_name, std::ios::binary};
    int32_t rows = matrix.size();
    int32_t cols = matrix[0].size();

    output.write((char *)&cols, sizeof(int32_t));
    output.write((char *)&rows, sizeof(int32_t));
    for (int32_t i = 0; i < rows; ++i) {
        for (int32_t j = 0; j < cols; ++j) {
            output.write((char *)&matrix[i][j], sizeof(float));
        }
    }
}

static void matrix_reader_rectangle_test(const std::vector<std::vector<float>> &matrix, MatrixReader &reader,
                                         size_t upper_left_row, size_t upper_left_col, size_t width, size_t height)
{
    FlatMatrix<float> rectangle = reader.loadRectangle(upper_left_row, upper_left_col, width, height);
    for (size_t rect_i = 0; rect_i < height; rect_i++) {
        for (size_t rect_j = 0; rect_j < width; rect_j++) {
            size_t total_i = rect_i + upper_left_row;
            size_t total_j = rect_j + upper_left_col;
            _assert(rectangle.at(rect_i, rect_j) == matrix[total_i][total_j]);
        }
    }
}

static void hmatrix_stripes_fit_in_memory()
{
    if (is_local())
        return;

    std::string matrix_a_file_name = "/mnt/home/_teaching/para/03-matrixmul-mpi/data/hmatrix.a";
    std::string matrix_b_file_name = "/mnt/home/_teaching/para/03-matrixmul-mpi/data/hmatrix.b";

    //std::string lmatrix_file_name = "/mnt/home/_teaching/para/03-matrixmul-mpi/data/lmatrix.a";

    MatrixReader matrix_a_reader{matrix_a_file_name};
    std::cout << "\thmatrix.a rows=" << matrix_a_reader.getRowsCount() << ", cols=" << matrix_a_reader.getColsCount()
              << std::endl;

    // Try loading just one line.
    size_t a_stripe_width = matrix_a_reader.getColsCount();
    size_t a_stripe_height = 64;
    FlatMatrix<float> a_stripe = matrix_a_reader.loadRectangle(0, 0, a_stripe_width, a_stripe_height);
    std::cout << "\tLoaded "<< a_stripe_height << " height stripe of hmatrix.a" << std::endl;

    MatrixReader matrix_b_reader{matrix_b_file_name};
    std::cout << "\thmatrix.b rows=" << matrix_b_reader.getRowsCount() << ", cols=" << matrix_b_reader.getColsCount()
              << std::endl;

    size_t b_stripe_width = 64;
    size_t b_stripe_height = matrix_b_reader.getRowsCount();
    FlatMatrix<float> b_stripe = matrix_b_reader.loadRectangle(0, 0, b_stripe_width, b_stripe_height);
    std::cout << "\tLoaded " << b_stripe_width << " width stripe of hmatrix.b" << std::endl;
}

#ifdef REMOTE

struct buffer_t {
    int size;
    int content[10];
    float f_content[10];
};

static void test_create_data_structure()
{
    // Create point data structure
    int count = 3;
    int block_lengths[] = {1, 10, 10};
    MPI_Aint displacements[] = {0, 4, 44};
    MPI_Datatype types[] = {MPI_INT, MPI_INT, MPI_FLOAT};
    MPI_Datatype new_type;
    CHECK(MPI_Type_create_struct(count, block_lengths, displacements, types, &new_type));
    CHECK(MPI_Type_commit(&new_type));
    std::cout << "Type commited" << std::endl;

    int new_type_size = 0;
    CHECK(MPI_Type_size(new_type, &new_type_size));
    std::cout << "Size of new type = " << new_type_size << std::endl;

    int rank = get_rank();
    if (rank == 0) {
        buffer_t buffer{3, {1,2,3}, {1.1, 2.2, 3.3}};
        CHECK(MPI_Send(&buffer, 1, new_type, 1, (int)Tag::from_master, MPI_COMM_WORLD));
    }
    else {
        buffer_t buffer{};
        MPI_Status status{};
        CHECK(MPI_Recv(&buffer, 1, new_type, 0, (int)Tag::from_master, MPI_COMM_WORLD, &status));

        int element_count = 0;
        CHECK(MPI_Get_elements(&status, new_type, &element_count));
        std::cout << "MPI_Get_elements = " << element_count << std::endl;

        std::cout << "Received buffer_t.size=" << buffer.size << ", buffer_t.content=[";
        for (int i = 0; i < buffer.size; i++)
            std::cout << buffer.content[i] << ", ";
        std::cout << "], buffer_t.f_content=[";
        for (int i = 0; i < buffer.size; i++)
            std::cout << std::setw(3) << buffer.f_content[i] << ", ";
        std::cout << "]" << std::endl;
    }

    CHECK(MPI_Type_free(&new_type));
}

static void test_create_submatrices_message_datatype()
{
    CHECK(MPI_Barrier(MPI_COMM_WORLD));

    MPI_Datatype submatrices_message_dt{};
    create_submatrices_message_datatype(&submatrices_message_dt);

    // Assemble message.
    std::vector<float> a_buffer(32*32);
    std::vector<float> b_buffer(32*32);
    for (size_t i = 0; i < 32*32; i++) {
        a_buffer[i] = (float)(i % 45);
        b_buffer[i] = (float)(i % 38);
    }
    submatrices_message_t message;
    message.a_col_start = 0;
    message.a_col_end = 32;
    message.a_row_start = 0;
    message.a_row_end = 32;
    message.b_col_start = 0;
    message.b_col_end = 32;
    message.b_row_start = 0;
    message.b_row_end = 32;
    std::copy(a_buffer.begin(), a_buffer.end(), message.a_buffer);
    std::copy(b_buffer.begin(), b_buffer.end(), message.b_buffer);

    CHECK(MPI_Barrier(MPI_COMM_WORLD));

    if (get_rank() == 0) {
        CHECK(MPI_Send(&message, 1, submatrices_message_dt, 1, (int)Tag::from_master, MPI_COMM_WORLD));
    }
    else {
        submatrices_message_t received_message{};
        MPI_Status status{};
        CHECK(MPI_Recv(&received_message, 1, submatrices_message_dt, 0, (int)Tag::from_master, MPI_COMM_WORLD, &status));

        // Compare buffers of message and received_message.
        for (size_t i = 0; i < 32*32; i++) {
            if (message.a_buffer[i] != received_message.a_buffer[i])
                throw RuntimeError() << "message.a_buffer[i] != received_message.a_buffer[i] where i=" << i;
            if (message.b_buffer[i] != received_message.b_buffer[i])
                throw RuntimeError() << "message.b_buffer[i] != received_message.b_buffer[i] where i=" << i;
        }
        // Compare whole messages.
        _assert_msg(received_message == message, "Received message is different than sent message.");
    }

    CHECK(MPI_Type_free(&submatrices_message_dt));
    CHECK(MPI_Barrier(MPI_COMM_WORLD));
}

static void test_create_result_message_datatype()
{
    CHECK(MPI_Barrier(MPI_COMM_WORLD));

    MPI_Datatype result_message_dt{};
    create_result_message_datatype(&result_message_dt);
    result_submatrix_message_t message {
        1, 2, 3, 4, {1.1, 2.2, 3.3, 4.4, 5.5}
    };

    CHECK(MPI_Barrier(MPI_COMM_WORLD));

    if (get_rank() == 0) {
        CHECK(MPI_Send(&message, 1, result_message_dt, 1, (int)Tag::from_master, MPI_COMM_WORLD));
    }
    else {
        result_submatrix_message_t received_message{};
        MPI_Status status{};
        CHECK(MPI_Recv(&received_message, 1, result_message_dt, 0, (int)Tag::from_master, MPI_COMM_WORLD, &status));

        _assert_msg(received_message == message, "Received and sent messages should equal");
    }

    CHECK(MPI_Type_free(&result_message_dt));
    CHECK(MPI_Barrier(MPI_COMM_WORLD));
}

#endif // REMOTE


static void test_matrix_reader()
{
#ifdef REMOTE
    if (get_rank() != 0)
        return;
#endif

    const std::string file_name = "./tmp_matrix.bin";

    std::vector<std::vector<float>> matrix = {
            {1, 2, 3, 23},
            {4, 5, 6, 23},
            {7, 8, 9, 23}
    };
    const size_t rows = matrix.size();
    const size_t cols = matrix[0].size();
    write_matrix(file_name, matrix);

    MatrixReader reader{file_name};

    _assert(reader.getColsCount() == cols);
    _assert(reader.getRowsCount() == rows);
    // Test full matrix load.
    matrix_reader_rectangle_test(matrix, reader, 0, 0, reader.getColsCount(), reader.getRowsCount());

    // Test partial matrix read - read just submatrix.
    matrix_reader_rectangle_test(matrix, reader, 1, 1, 2, 2);
    matrix_reader_rectangle_test(matrix, reader, 1, 2, 2, 1);

    std::remove(file_name.c_str());
}

static void run_one_test(const test_t &test)
{
    std::cout << "==========================" << std::endl;
    std::cout << "Running test: " << test.first << std::endl;
    try {
        test.second();
    }
    catch (StreamException &exception) {
        std::cout << "Test " << test.first << " failed with " << exception.what() << std::endl;
        std::cout << "Exiting ..." << std::endl;
        exit(1);
    }
    std::cout << "Test " << test.first << " passed" << std::endl;
    std::cout << "==========================" << std::endl;
}

static std::vector<test_t> tests = {
        {"Flat matrix tests", test_flat_matrix},
        {"Matrix reader tests", test_matrix_reader}
        //{"Stripe fits in memory test", hmatrix_stripes_fit_in_memory}

#ifdef REMOTE
        ,
        //{"Create new data structure", test_create_data_structure},
        {"Create submatrices message datatype", test_create_submatrices_message_datatype},
        {"Create result message datatype", test_create_result_message_datatype}
#endif // REMOTE
};

int main(int argc, char **argv)
{
#ifdef REMOTE
    CHECK(MPI_Init(&argc, &argv));
#endif

    std::cout << "Running all tests..." << std::endl;
    for (auto &&test : tests) {
        run_one_test(test);
    }
    std::cout << "All tests passed..." << std::endl;

#ifdef REMOTE
    CHECK(MPI_Finalize());
#endif
}