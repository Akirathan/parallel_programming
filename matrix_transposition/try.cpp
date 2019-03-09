#include <iostream>
#include <tbb/tbb.h>
#include "try.hpp"

void sample_for_loop()
{
    int counter = 0;
    tbb::parallel_for(tbb::blocked_range<size_t>(0, 3),
        [&](const tbb::blocked_range<size_t>& r) {
                for(size_t i=r.begin(); i!=r.end(); ++i){
                    counter++;
                }
        }
    );

    std::cout << "counter: " << counter << std::endl;
}