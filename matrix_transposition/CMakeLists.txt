cmake_minimum_required(VERSION 3.13)

project(tbb-matrix-tran)

set(TBB_ROOT_DIR tbb)

set(SOURCES
        tbb-matrix-tran.cpp
        try.cpp
        )

include_directories(${TBB_ROOT_DIR}/include)

add_executable(tbb-matrix-tran ${SOURCES})

set(CMAKE_EXE_LINKER_FLAGS "-Wl,-rpath /usr/local/lib")

target_link_libraries(tbb-matrix-tran
        tbb
        tbbmalloc)

