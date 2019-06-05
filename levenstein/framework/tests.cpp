
#include <vector>
#include <iostream>
#include <algorithm>
#include <random>
#include "../serial/implementation.hpp"
#include "../serial/dummy_levenstein.hpp"
#include "implementation.hpp"
#include "internal/exception.hpp"


static size_t compute_serial(const std::vector<char> &array_1, const std::vector<char> &array_2)
{
    SerialEditDistance<char, size_t, false> serial_impl;
    serial_impl.init(array_1.size(), array_2.size());
    return serial_impl.compute(array_1, array_2);
}

template <bool DEBUG = false>
static size_t compute_mine(const std::vector<char> &array_1, const std::vector<char> &array_2)
{
    EditDistance<char, size_t, DEBUG> my_impl;
    my_impl.init(array_1.size(), array_2.size());
    return my_impl.compute(array_1, array_2);
}

static size_t compute_dummy(const std::vector<char> &array_1, const std::vector<char> &array_2)
{
    dummy_levenstein dummy_impl{array_1.begin(), array_1.end(), array_2.begin(), array_2.end()};
    return dummy_impl.compute();
}

template <bool DEBUG = false>
static void compare_both(const std::vector<char> &array_1, const std::vector<char> &array_2)
{
    size_t serial_res = compute_serial(array_1, array_2);
    size_t my_res = compute_mine<DEBUG>(array_1, array_2);
    size_t dummy_res = compute_dummy(array_1, array_2);
    if (dummy_res != serial_res)
        throw bpp::RuntimeError() << "Serial and dummy results differ!! Serial result = " << serial_res
                                  << ", Dummy result = " << dummy_res;
    if (serial_res != my_res)
        throw bpp::RuntimeError() << "Different results: Mine=" << my_res << ", Serial=" << serial_res;
}

static std::vector<char> get_random_vector(size_t size)
{
    std::uniform_int_distribution<int> ui{0, 255};
    std::mt19937 engine;

    std::vector<char> vec(size);
    std::generate(vec.begin(), vec.end(), [&ui, &engine] {
        return ui(engine);
    });

    return vec;
}

static std::pair<std::vector<char>, std::vector<char>> get_random_vectors(size_t size_1, size_t size_2)
{
    std::uniform_int_distribution<int> ui{0, 255};
    std::mt19937 engine;

    std::vector<char> vec_1(size_1);
    std::generate(vec_1.begin(), vec_1.end(), [&ui, &engine] {
        return ui(engine);
    });

    std::vector<char> vec_2(size_2);
    std::generate(vec_2.begin(), vec_2.end(), [&ui, &engine] {
        return ui(engine);
    });

    return std::make_pair(vec_1, vec_2);
}

static void simple_test()
{
    std::vector<char> array_1 = {'a', 'b', 'c', 'd'};
    std::vector<char> array_2 = {'e', 'f', 'g', 'h'};
    compare_both(array_1, array_2);
}

static void simple_one_char_same_test()
{
    std::vector<char> array_1 = {'a', 'b', 'c', 'd'};
    std::vector<char> array_2 = {'e', 'b', 'g', 'h'};
    compare_both(array_1, array_2);
}

static void different_sizes_test()
{
    std::cout << "Running different_sizes_test..." << std::endl;
    std::vector<char> array_1 = {'a', 'b', 'c'};
    std::vector<char> array_2 = {'d', 'e', 'f', 'g'};
    std::cout << "\tcompare_both(array_1, array_2)" << std::endl;
    compare_both(array_1, array_2);
    std::cout << "\tcompare_both(array_2, array_1)" << std::endl;
    compare_both(array_2, array_1);
}

static void different_sizes_complex_tests()
{
    std::cout << "Running different_sizes_complex_tests..." << std::endl;

    auto vectors = get_random_vectors(512, 64);
    compare_both(vectors.first, vectors.second);

    std::cout << "different_sizes_complex_tests passed" << std::endl;
}

static void bednarek_random_tests()
{
    std::cout << "Bednarek's random tests started" << std::endl;

    const std::vector<std::pair<size_t, size_t>> sizes = {
            {512,   64},
            /*{4096,  64},
            {4096,  512},
            {32768, 64},
            {32768, 512},
            {32768, 4096}*/
    };

    const size_t iters = 2;

    std::uniform_int_distribution<int> ui{0, 255};
    std::mt19937 engine;

    for (auto &&size : sizes) {
        // Reset engine.
        engine.seed();

        std::vector<char> vec_a(size.first);
        std::vector<char> vec_b(size.second);
        std::generate(vec_a.begin(), vec_a.end(), [&ui, &engine] {
            return ui(engine);
        });
        std::generate(vec_b.begin(), vec_b.end(), [&ui, &engine] {
            return ui(engine);
        });

        EditDistance my_impl;
        my_impl.init(size.first, size.second);
        SerialEditDistance serial_impl;
        serial_impl.init(size.first, size.second);
        dummy_levenstein dummy_impl{vec_a.begin(), vec_a.end(), vec_b.begin(), vec_b.end()};
        for (size_t i = 0; i < iters; ++i) {
            size_t my_res = my_impl.compute(vec_a, vec_b);
            size_t serial_res = serial_impl.compute(vec_a, vec_b);
            size_t dummy_res = dummy_impl.compute();
            if (serial_res != dummy_res)
                throw bpp::RuntimeError() << "Serial and dummy results differ! Serial res = " << serial_res
                                          << ", Dummy res = " << dummy_res;
            if (my_res != serial_res)
                throw bpp::RuntimeError() << "Iteration=" << i << ", serial result=" << serial_res
                                          << ", My result=" << my_res;
        }
    }

    std::cout << "Bednarek's random tests passed" << std::endl;
}

static void run_all_tests()
{
    std::cout << "Running all tests..." << std::endl;
    simple_test();
    simple_one_char_same_test();
    different_sizes_test();
    different_sizes_complex_tests();
    bednarek_random_tests();
    std::cout << "All tests passed" << std::endl;
}

int main()
{
    run_all_tests();
}