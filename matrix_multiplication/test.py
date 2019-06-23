#!/usr/bin/python

import subprocess
import time
import os

TEST_DIR = './tests'

COMPARATOR_EXE = './exec/comparator'
GENERATOR_EXE = './exec/generator'
SERIAL_EXE = './exec/multiply'
PARALLEL_EXE = './matrix_mult'

SERIAL_SRUN = ['srun', '-p', 'small-lp']
PARTITION = 'small-hp'
N_PROCS = 2
N_HOSTS = 1
CORES = 2
TIMEOUT = '20:00'
PARALLEL_SRUN = ['srun', '-p', PARTITION, '-n', str(N_PROCS), '-N', str(N_HOSTS), '-c', str(CORES),
                 '--distribution', 'cyclic', '--time='+TIMEOUT]


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
    print('===========================================')
    print('Running test for %s and %s' % (matrix_a_fname, matrix_b_fname))
    tmp_out_serial = TEST_DIR + '/matrix_serial_result.bin'
    tmp_out_parallel = TEST_DIR + '/matrix_parallel_result.bin'

    serial_time_ms = serial_multiply(matrix_a_fname, matrix_b_fname, tmp_out_serial)
    parallel_time_ms = parallel_multiply(matrix_a_fname, matrix_b_fname, tmp_out_parallel)

    print('Test multiplication of matrices %s and %s, with Serial time=%f, Parallel time=%f'
          % (matrix_a_fname, matrix_b_fname, serial_time_ms, parallel_time_ms))

    if not compare(tmp_out_serial, tmp_out_parallel):
        print('Test FAILED')
    else:
        print('Test SUCCEEDED')

    print('===========================================')
    return serial_time_ms / parallel_time_ms


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
    TestInput(5, 5, 5, 5, 42)
]

if __name__ == '__main__':
    check_files_and_dirs()
    input_idx = 0
    for test_input in INPUTS:
        matrix_a_fname = TEST_DIR + '/%d-matrix-r%d-c%d-s%d.a' % \
                         (input_idx, test_input.a_rows_count, test_input.a_cols_count, test_input.seed)
        matrix_b_fname = TEST_DIR + '/%d-matrix-r%d-c%d-s%d.b' % \
                         (input_idx, test_input.b_rows_count, test_input.b_cols_count, test_input.seed)

        generate(test_input.a_rows_count, test_input.a_cols_count, test_input.seed, matrix_a_fname)
        generate(test_input.b_rows_count, test_input.b_cols_count, test_input.seed, matrix_b_fname)
        speedup = test(matrix_a_fname, matrix_b_fname)
        print('Speedup is %f' % speedup)

        input_idx += 1


