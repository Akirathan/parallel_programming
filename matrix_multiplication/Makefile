# $@ ... name of file being generated
# $^ ... all prerequisities
# $< ... first prerequisite
# Substitution references:
#   $(VARIABLE:OLD_SUFFIX=NEW_SUFFIX)

MAKE = make -j 4
CPP = mpicxx
CFLAGS = -Wall -g -O3 -std=c++17
INCLUDE = .
LDFLAGS=
LIBS=
LIBDIRS=
SOURCES = \
	MatrixReader.cpp \
	master.cpp \
	common.cpp \
	worker.cpp \
	main.cpp

UNIT_TEST_SOURCES = \
	MatrixReader.cpp \
	common.cpp \
	unit_tests.cpp

OBJECTS = $(SOURCES:.cpp=.o)
UNIT_TEST_OBJECTS = $(UNIT_TEST_SOURCES:.cpp=.o)

HEADERS = $(shell find . -name '*.hpp')
EXECUTABLE = ./du3

COMPILE_CMD = $(CPP) $(CFLAGS) $(addprefix -I,$(INCLUDE)) $(LDFLAGS) $(addprefix -L,$(LIBDIRS)) $(addprefix -l,$(LIBS)) $< -c -o $@
LINK_CMD = $(CPP) $(CFLAGS) $(addprefix -I,$(INCLUDE)) $(LDFLAGS) $(addprefix -L,$(LIBDIRS)) $(addprefix -l,$(LIBS)) $^ -o $@

.PHONY: all clear clean purge run run_hmatrix run_unit_tests send_files

all: $(EXECUTABLE)

# Building Targets

$(EXECUTABLE): $(OBJECTS)
	$(LINK_CMD)

$(OBJECTS): $(SOURCES) $(HEADERS)

unit_tests: $(UNIT_TEST_OBJECTS)
	$(LINK_CMD)

$(UNIT_TEST_OBJECTS): $(UNIT_TEST_SOURCES) $(HEADERS)

%.o: %.cpp
	$(COMPILE_CMD)

#######################################

FILES = \
	Makefile \
	main.cpp \
	FlatMatrix.hpp \
	exception.hpp \
	MatrixReader.hpp \
	MatrixReader.cpp \
	common.hpp \
	common.cpp \
	worker.hpp \
	worker.cpp \
	master.hpp \
	master.cpp \
	unit_tests.cpp \
	stopwatch.hpp

SSH_CMD = ssh parlab

DATA_DIR = /mnt/home/_teaching/para/03-matrixmul-mpi/data
DEBUG_DATA_DIR = ~/matrix_mult_para/tests
PROJECT_DIR = /mnt/home/s_bruchpa/matrix_mult_para

COMPARATOR_EXE = $(PROJECT_DIR)/exec/comparator

PARTITION = small-hp
RELEASE_N_PROCS = 256
RELEASE_N_HOSTS = 8
RELEASE_CORES = 2
N_PROCS = 256
N_HOSTS = 8
CORES = 2
TIMEOUT = 5:00
SRUN_CMD = srun -n $(N_PROCS) -N $(N_HOSTS) -p $(PARTITION) -c $(CORES) --distribution cyclic --time=$(TIMEOUT)
UNIT_TEST_SRUN_CMD = srun -n 2 -N 1 -p $(PARTITION) -c 2 --distribution cyclic --time=$(TIMEOUT)

DEBUG_MATRIX_A = $(DEBUG_DATA_DIR)/matrix-r5-c5-s42.a
DEBUG_MATRIX_B = $(DEBUG_DATA_DIR)/matrix-r5-c5-s42.b
HMATRIX_A = /mnt/home/_teaching/para/03-matrixmul-mpi/data/hmatrix.a
HMATRIX_B = /mnt/home/_teaching/para/03-matrixmul-mpi/data/hmatrix.b
HMATRIX_R = /mnt/home/_teaching/para/03-matrixmul-mpi/data/hmatrix.r
LMATRIX_A = /mnt/home/_teaching/para/03-matrixmul-mpi/data/lmatrix.a
LMATRIX_B = /mnt/home/_teaching/para/03-matrixmul-mpi/data/lmatrix.b
LMATRIX_R = /mnt/home/_teaching/para/03-matrixmul-mpi/data/lmatrix.r
DEBUG_OUT = $(DEBUG_DATA_DIR)/result-parallel.bin
ARGS = $(DEBUG_MATRIX_A) $(DEBUG_MATRIX_B) $(DEBUG_OUT)

send_files:
	@echo "Sending files..."
	@scp -r -P 42222 $(FILES) "s_bruchpa@parlab.ms.mff.cuni.cz:$(PROJECT_DIR)"

compile: send_files
	@echo "Compiling..."
	@$(SSH_CMD) "cd $(PROJECT_DIR); srun $(MAKE)"

compile_unit_tests: send_files
	@echo "Compiling unit tests..."
	@$(SSH_CMD) "cd $(PROJECT_DIR); srun $(MAKE) unit_tests"

run_unit_tests:
	@echo "Running $(UNIT_TEST_SRUN_CMD) unit_tests"
	@$(SSH_CMD) "cd $(PROJECT_DIR); $(UNIT_TEST_SRUN_CMD) ./unit_tests"

test: test.py
	@echo "Running testing script..."
	@scp -r -P 42222 test.py "s_bruchpa@parlab.ms.mff.cuni.cz:$(PROJECT_DIR)"
	@$(SSH_CMD) "cd $(PROJECT_DIR); ./test.py"

run:
	@echo "Running $(SRUN_CMD) $(EXECUTABLE) $(ARGS)"
	@$(SSH_CMD) "cd $(PROJECT_DIR); $(SRUN_CMD) $(EXECUTABLE) $(ARGS)"

HMATRIX_CMD = $(SRUN_CMD) $(EXECUTABLE) $(HMATRIX_A) $(HMATRIX_B) $(DEBUG_OUT)
run_hmatrix:
	@echo "Running $(HMATRIX_CMD)"
	@$(SSH_CMD) "cd $(PROJECT_DIR); $(HMATRIX_CMD); echo Comparing...; srun $(COMPARATOR_EXE) $(DEBUG_OUT) $(HMATRIX_R)"

LMATRIX_CMD = $(SRUN_CMD) $(EXECUTABLE) $(LMATRIX_A) $(LMATRIX_B) $(DEBUG_OUT)
run_lmatrix:
	@echo "Running $(LMATRIX_CMD)"
	@$(SSH_CMD) "cd $(PROJECT_DIR); $(LMATRIX_CMD); echo Comparing...; srun $(COMPARATOR_EXE) $(DEBUG_OUT) $(LMATRIX_R)"


# Cleaning Stuff

clear:
	@echo Removing all generated files...
	-@rm -f $(EXECUTABLE)

clean: clear

purge: clear

