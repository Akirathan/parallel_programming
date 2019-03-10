cmake_minimum_required(VERSION 3.13)
project(k_means)


set(TBB_ROOT_DIR tbb)
set(FRAMEWORK_DIR framework)

# Compile options
set(CMAKE_CXX_STANDARD 17)
add_compile_options("-Wall")

include_directories(${TBB_ROOT_DIR}/include)
include_directories(${FRAMEWORK_DIR})
include_directories(${FRAMEWORK_DIR}/internal)

set(SOURCES
        #${FRAMEWORK_DIR}/k-means.cpp
        ${FRAMEWORK_DIR}/test.cpp
        )

add_executable(k_means ${SOURCES})

set(CMAKE_EXE_LINKER_FLAGS "-Wl,-rpath /usr/local/lib")

target_link_libraries(k_means
        tbb
        tbbmalloc)

######## SERIAL #######
add_subdirectory(serial)


####### Testing part ######
add_subdirectory(tests)

