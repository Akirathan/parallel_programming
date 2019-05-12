cmake_minimum_required(VERSION 3.14)
set(PROJECT_NAME my_project)
project(levenstein)

add_compile_options("-Wall" "-Wextra" -fopenmp)
add_link_options(-fopenmp)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE " -O3")
set(CMAKE_CXX_FLAGS_DEBUG " -g ")

add_compile_definitions("BOOST_ALL_DYN_LINK")
find_package(Boost COMPONENTS log system unit_test_framework REQUIRED)
include_directories(internal)

set(SOURCES
        implementation.hpp
        levenshtein.cpp
        internal/exception.hpp
        internal/interface.hpp
        internal/stopwatch.hpp
        )

add_executable(levenstein
        ${SOURCES}
        )

target_link_libraries(levenstein)

add_executable(my_main my_main.cpp)