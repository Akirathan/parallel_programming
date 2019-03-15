#!/usr/bin/python3

import subprocess
import filecmp

SERIAL_EXE = "serial/k-means_serial"
PARALLEL_EXE = "cmake-build-debug/k_means"
DEBUG = False
EXIT_AFTER_FIRST_FAIL = False

TEST_INPUTS = [
{
	"points_file": "data/01-256k",
	"k": "4",
	"iters": "4"
},
{
	"points_file": "data/01-256k",
	"k": "32",
	"iters": "32"
},
{
	"points_file": "data/01-256k",
	"k": "128",
	"iters": "64"
},
{
	"points_file": "data/02-1M",
	"k": "32",
	"iters": "32"
}]

SERIAL_CENTROIDS_FILE = "data/serial_centroids"
SERIAL_ASSIGN_FILE = "data/serial_assignments"
PARALLEL_CENTROIDS_FILE = "data/parallel_centroids"
PARALLEL_ASSIGN_FILE = "data/parallel_assignments"

def check_exes():
	try:
		open(SERIAL_EXE, mode="r")
	except OSError:
		print(f"Cannot open {SERIAL_EXE}")
	
	try:
		open(PARALLEL_EXE, mode="r")
	except OSError:
		print(f"Cannot open {PARALLEL_EXE}")


def compare_files(file_1, file_2) -> bool:
	return filecmp.cmp(file_1, file_2, shallow=False)
	
def fail(test_input):
	print(f"Test input {test_input} FAILED, exitting...")
	exit(1)

def success(test_input):
	print(f"Test input {test_input} SUCEEDED")

def run_serial(test_input):
	args = [SERIAL_EXE, test_input["points_file"], test_input["k"], test_input["iters"], SERIAL_CENTROIDS_FILE, SERIAL_ASSIGN_FILE]
	if DEBUG:
		args = [SERIAL_EXE, "-debug", test_input["points_file"], test_input["k"], test_input["iters"], SERIAL_CENTROIDS_FILE, SERIAL_ASSIGN_FILE]

	print(f"Running serial...")
	subprocess.run(args)


def run_parallel(test_input):
	args = [PARALLEL_EXE, test_input["points_file"], test_input["k"], test_input["iters"], PARALLEL_CENTROIDS_FILE, PARALLEL_ASSIGN_FILE]
	if DEBUG:
		args = [PARALLEL_EXE, "-debug", test_input["points_file"], test_input["k"], test_input["iters"], PARALLEL_CENTROIDS_FILE, PARALLEL_ASSIGN_FILE]

	print(f"Running parallel...")
	subprocess.run(args)


def print_heading(heading: str):
	print("================================")
	print(f"======  {heading} =====")
	print("================================")

def run_test(test_input):
	print_heading(f"Running test {test_input} ...")
	run_serial(test_input)
	run_parallel(test_input)
	test_succ = True
	if not compare_files(PARALLEL_CENTROIDS_FILE, SERIAL_CENTROIDS_FILE):
		test_succ = False
	else:
		test_succ = True

	if EXIT_AFTER_FIRST_FAIL and not test_succ:
		fail(test_input)
	elif not test_succ:
		print(f"Test {test_input} FAILED")
	elif test_succ:
		success(test_input)


	
check_exes()
for test_input in TEST_INPUTS:
	run_test(test_input)
	
