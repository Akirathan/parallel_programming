cmake_minimum_required(VERSION 3.13)
set(PROJECT_NAME my_project)
project(potential)

add_compile_options("-Wall" "-Wextra")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE " -O2")
set(CMAKE_CXX_FLAGS_DEBUG " -g ")

add_compile_definitions("BOOST_ALL_DYN_LINK" "LOCAL")
find_package(Boost COMPONENTS log system unit_test_framework REQUIRED)
include_directories(internal)

set(SOURCES
        internal/args.hpp
        internal/data.hpp
        internal/exception.hpp
        internal/interface.hpp
        internal/serial.hpp
        internal/stopwatch.hpp
        internal/verifier.hpp
        internal/interface.hpp
        implementation.hpp
        kernels.h
        kernels.cu
        )

add_executable(potential
        ${SOURCES}
        try.cpp
        potential.cpp
        )

target_link_libraries(potential)