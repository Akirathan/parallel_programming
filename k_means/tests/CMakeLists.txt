project(tests)

#find_package(Boost COMPONENTS unit_test_framework REQUIRED)

#include_directories(/usr/include/boost/test)

set(TEST_SOURCES
        sample_test.cpp
        )

add_executable(tests
        ${TEST_SOURCES}
        )

#target_link_libraries(tests ${Booost_UNIT_TEST_FRAMEWORK_LIBRARY})
