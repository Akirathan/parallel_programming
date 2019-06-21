cmake_minimum_required(VERSION 3.13)
set(PROJECT_NAME matrix_mult)
project(${PROJECT_NAME})

add_compile_options("-Wall" "-Wextra")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE " -O2")
set(CMAKE_CXX_FLAGS_DEBUG " -g ")

add_compile_definitions("BOOST_ALL_DYN_LINK" "LOCAL")
find_package(Boost COMPONENTS log system unit_test_framework REQUIRED)
include_directories(${Boost_INCLUDE_DIR})

set(SOURCES
        main.cpp
        MatrixReader.cpp
        MatrixReader.hpp
        exception.hpp
        common.hpp
        worker.hpp
        master.hpp
        worker.cpp
        master.cpp
        FlatMatrix.hpp)

add_executable(${PROJECT_NAME}
        ${SOURCES}
        )

add_executable(try try.cpp)

target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES})

