//
// Created by pal on 12.5.19.
//

#include <cstdlib>
#include <iostream>

template <typename T>
static void print_matrix(T **matrix, size_t matrix_size)
{
    for (size_t i = 0; i < matrix_size; i++) {
        for (size_t j = 0; j < matrix_size; j++)
            std::cout << matrix[i][j] << " ";
        std::cout << std::endl;
    }
}

static void alloc_array()
{
    int *arr = new int[5];
    
    delete[] arr;
}

static void alloc_matrix()
{
    const size_t matrix_size = 3;
    int **matrix = new int*[matrix_size];
    for (size_t i = 0; i < matrix_size; i++) {
        matrix[i] = new int[matrix_size];
    }

    for (size_t i = 0; i < matrix_size; i++)
        for (size_t j = 0; j < matrix_size; j++)
            matrix[i][j] = i + j;

    print_matrix(matrix, matrix_size);

    for (size_t i = 0; i < matrix_size; i++)
        delete[] matrix[i];
    delete[] matrix;
}

int main()
{
    alloc_array();
}
