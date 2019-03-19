cmake_minimum_required(VERSION 3.13)
project(serial_k_means)

set(CMAKE_CXX_STANDARD 17)
add_compile_options("-Wall")

include_directories(.)

set(SERIAL_SOURCES
        test.cpp
        #k-means.cpp
        )

add_executable(serial_k_means ${SERIAL_SOURCES})
