#!/usr/bin/python

import subprocess
import time
import os
import copy

TEST_DIR = './tests'

COMPARATOR_EXE = './exec/comparator'
GENERATOR_EXE = './exec/generator'
SERIAL_EXE = './exec/multiply'
PARALLEL_EXE = './matrix_mult'

SERIAL_SRUN = ['srun', '-p', 'small-lp']
PARTITION = 'small-hp'
N_PROCS = 3
N_HOSTS = 1
CORES = 2
TIMEOUT = '20:00'
PARALLEL_SRUN = ['srun', '-p', PARTITION, '-n', str(N_PROCS), '-N', str(N_HOSTS), '-c', str(CORES),
                 '--distribution', 'cyclic', '--time='+TIMEOUT]


class TestInput:
    def __init__(self, a_rows_count, a_cols_count, b_rows_count, b_cols_count, seed):
        assert(a_cols_count == b_rows_count)
        self.a_rows_count = a_rows_count
        self.a_cols_count = a_cols_count
        self.b_rows_count = b_rows_count
        self.b_cols_count = b_cols_count
        self.seed = seed

    def __str__(self):
        return 'a_rows_count=%d, a_cols_count=%d, b_rows_count=%d, b_cols_count=%d, seed=%d' \
               % (self.a_rows_count, self.a_cols_count, self.b_rows_count, self.b_cols_count, self.seed)


INPUTS = [
    TestInput(5, 5, 5, 5, 42),
    #TestInput(5, 12, 12, 7, 42),
    #TestInput(50, 50, 50, 50, 42),
    #TestInput(250, 340, 340, 100, 42),
]

BIG_TESTS = True
BIG_INPUTS = [
    TestInput(4*1024, 7*1024, 7*1024, 6*1024, 42),
]

# Note that release tests may run for long time.
RELEASE_TESTS = True
HMATRIX_A = '/mnt/home/_teaching/para/03-matrixmul-mpi/data/hmatrix.a'
HMATRIX_B = '/mnt/home/_teaching/para/03-matrixmul-mpi/data/hmatrix.b'
HMATRIX_SERIAL_TIME_MS = 1308.9 * 1000
LMATRIX_A = '/mnt/home/_teaching/para/03-matrixmul-mpi/data/lmatrix.a'
LMATRIX_B = '/mnt/home/_teaching/para/03-matrixmul-mpi/data/lmatrix.b'
LMATRIX_SERIAL_TIME_MS = 1255 * 1000


def file_exists(fname):
    try:
        open(fname, mode='r')
    except Exception:
        return False
    return True


def check_files_and_dirs():
    for executable in [COMPARATOR_EXE, GENERATOR_EXE, SERIAL_EXE, PARALLEL_EXE]:
        try:
            open(executable, mode='r')
        except OSError:
            print('Cannot find executable file %s, exiting...' % executable)
            exit(1)
    print('Executables and dirs checked.')


def compare(matrix_1_fname, matrix_2_fname):
    exit_status = subprocess.call(SERIAL_SRUN + [COMPARATOR_EXE, matrix_1_fname, matrix_2_fname])
    return exit_status == 0


def generate(rows_count, cols_count, seed, fname):
    exit_status = subprocess.call(SERIAL_SRUN + [GENERATOR_EXE, str(rows_count), str(cols_count), str(seed), fname])
    if exit_status != 0:
        raise Exception('Generate call failed')


def serial_multiply(matrix_1_fname, matrix_2_fname, out_fname):
    """ Returns elapse time in ms. """
    time_start = time.time()
    exit_status = subprocess.call(SERIAL_SRUN + [SERIAL_EXE, matrix_1_fname, matrix_2_fname, out_fname])
    time_end = time.time()
    if exit_status != 0:
        raise Exception('Serial multiplication call failed with status %d' % exit_status)
    return (time_end - time_start) * 1000


def parallel_multiply(matrix_1_fname, matrix_2_fname, out_fname):
    cmd = PARALLEL_SRUN + [PARALLEL_EXE, matrix_1_fname, matrix_2_fname, out_fname]
    print('Command: %s' % cmd)
    time_start = time.time()
    exit_status = subprocess.call(cmd)
    time_end = time.time()
    if exit_status != 0:
        raise Exception('Parallel multiplication call failed with status %d' % exit_status)
    return (time_end - time_start) * 1000


def test(matrix_a_fname, matrix_b_fname):
    """
    Returns speedup.
    """
    def generate_serial_time_fname():
        items = matrix_a_fname.split('/')
        a_fname = items[len(items) - 1]
        items = matrix_b_fname.split('/')
        b_fname = items[len(items) - 1]
        return TEST_DIR + '/%s---%s---serial_time_ms.txt' % (a_fname, b_fname)

    def serial_was_executed():
        serial_time_fname = generate_serial_time_fname()
        if file_exists(serial_time_fname):
            return True
        else:
            return False

    def read_serial_time_from_file():
        serial_time_fname = generate_serial_time_fname()
        with open(serial_time_fname, 'r') as file:
            return float(file.read())

    def write_serial_time_to_file(time_ms):
        my_time_ms = copy.copy(time_ms)
        serial_time_fname = generate_serial_time_fname()
        with open(serial_time_fname, 'w') as file:
            file.write(str(my_time_ms))

    print('===========================================')
    print('Running test for %s and %s' % (matrix_a_fname, matrix_b_fname))
    tmp_out_serial = TEST_DIR + '/matrix_serial_result.bin'
    tmp_out_parallel = TEST_DIR + '/matrix_parallel_result.bin'

    serial_time_ms = serial_multiply(matrix_a_fname, matrix_b_fname, tmp_out_serial)
    parallel_time_ms = parallel_multiply(matrix_a_fname, matrix_b_fname, tmp_out_parallel)

    print('Test multiplication of matrices %s and %s, with Serial time=%f ms, Parallel time=%f ms'
          % (matrix_a_fname, matrix_b_fname, serial_time_ms, parallel_time_ms))

    if not compare(tmp_out_serial, tmp_out_parallel):
        print('Test FAILED')
    else:
        print('Test SUCCEEDED')

    print('===========================================')
    return serial_time_ms / parallel_time_ms


def functional_tests():
    print('Running tests on small inputs...')
    for test_input in INPUTS:
        matrix_a_fname = TEST_DIR + '/matrix-r%d-c%d-s%d.a' % \
                         (test_input.a_rows_count, test_input.a_cols_count, test_input.seed)
        matrix_b_fname = TEST_DIR + '/matrix-r%d-c%d-s%d.b' % \
                         (test_input.b_rows_count, test_input.b_cols_count, test_input.seed)

        generate(test_input.a_rows_count, test_input.a_cols_count, test_input.seed, matrix_a_fname)
        generate(test_input.b_rows_count, test_input.b_cols_count, test_input.seed, matrix_b_fname)
        speedup = test(matrix_a_fname, matrix_b_fname)
        print('Speedup is %f' % speedup)


if __name__ == '__main__':
    check_files_and_dirs()
    functional_tests()
    exit(0)

    parallel_result_fname = TEST_DIR + '/release_tests_parallel_result.bin'

    if BIG_TESTS:
        for test_input in BIG_INPUTS:
            matrix_a_fname = TEST_DIR + '/matrix-r%d-c%d-s%d.a' % \
                             (test_input.a_rows_count, test_input.a_cols_count, test_input.seed)
            matrix_b_fname = TEST_DIR + '/matrix-r%d-c%d-s%d.b' % \
                             (test_input.b_rows_count, test_input.b_cols_count, test_input.seed)

            generate(test_input.a_rows_count, test_input.a_cols_count, test_input.seed, matrix_a_fname)
            generate(test_input.b_rows_count, test_input.b_cols_count, test_input.seed, matrix_b_fname)

            time_ms = parallel_multiply(matrix_a_fname, matrix_b_fname, parallel_result_fname)
            print('%s test run in %f ms' % (test_input, time_ms))

    if RELEASE_TESTS:

        hmatrix_time_ms = parallel_multiply(HMATRIX_A, HMATRIX_B, parallel_result_fname)
        print('hmatrix serial time is %f, parallel time is %f ms' % (HMATRIX_SERIAL_TIME_MS, hmatrix_time_ms))
        print('hmatrix speedup is %f' % (HMATRIX_SERIAL_TIME_MS/hmatrix_time_ms))

        lmatrix_time_ms = parallel_multiply(LMATRIX_A, LMATRIX_B, parallel_result_fname)
        print('lmatrix serial time is %f, parallel time is %f ms' % (LMATRIX_SERIAL_TIME_MS, lmatrix_time_ms))
        print('lmatrix speedup is %f' % (LMATRIX_SERIAL_TIME_MS/lmatrix_time_ms))




