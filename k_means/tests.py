#!/usr/bin/python3

import subprocess
import filecmp

SERIAL_EXE = "serial/k-means_serial"
PARALLEL_EXE = "cmake-build-debug/k_means"
DEBUG = False
EXIT_AFTER_FIRST_FAIL = False
RUN_JUST_PARALLEL = False

TEST_INPUTS = [
{
	"points_file": "data/02-1M",
	"k": "128",
	"iters": "10"
},
{
	"points_file": "data/02-1M",
	"k": "128",
	"iters": "32"
},
{
	"points_file": "data/02-1M",
	"k": "128",
	"iters": "128"
},
{
	"points_file": "data/02-1M",
	"k": "128",
	"iters": "256"
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
	print("Test FAILED, exitting...")
	exit(1)

def success(test_input):
	print("Test SUCEEDED")

def run_serial(test_input):
	args = [SERIAL_EXE, test_input["points_file"], test_input["k"], test_input["iters"], SERIAL_CENTROIDS_FILE, SERIAL_ASSIGN_FILE]
	if DEBUG:
		args = [SERIAL_EXE, "-debug", test_input["points_file"], test_input["k"], test_input["iters"], SERIAL_CENTROIDS_FILE, SERIAL_ASSIGN_FILE]

	subprocess.run(args)


def run_parallel(test_input):
	args = [PARALLEL_EXE, test_input["points_file"], test_input["k"], test_input["iters"], PARALLEL_CENTROIDS_FILE, PARALLEL_ASSIGN_FILE]
	if DEBUG:
		args = [PARALLEL_EXE, "-debug", test_input["points_file"], test_input["k"], test_input["iters"], PARALLEL_CENTROIDS_FILE, PARALLEL_ASSIGN_FILE]

	subprocess.run(args)

def run_just_parallel(test_input):
	print_heading(f"Running parallel {test_input} ...")
	run_parallel(test_input)

def print_heading(heading: str):
	print("================================")
	print(f"======  {heading} =====")
	print("================================")

def run_test(test_input):
	print_heading(f"Running test {test_input} ...")
	print("Running serial...")
	run_serial(test_input)
	print("Running parallel...")
	run_parallel(test_input)
	test_succ = True
	if not compare_files(PARALLEL_CENTROIDS_FILE, SERIAL_CENTROIDS_FILE):
		test_succ = False
	else:
		test_succ = True

	if EXIT_AFTER_FIRST_FAIL and not test_succ:
		fail(test_input)
	elif not test_succ:
		print("Test FAILED")
	elif test_succ:
		success(test_input)


	
check_exes()
for test_input in TEST_INPUTS:
	if RUN_JUST_PARALLEL:
		run_just_parallel(test_input)
	else:
		run_test(test_input)
	
