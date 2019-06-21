# $@ ... name of file being generated
# $^ ... all prerequisities
# $< ... first prerequisite
# Substitution references:
#   $(VARIABLE:OLD_SUFFIX=NEW_SUFFIX)

CPP = mpicxx
CFLAGS = -Wall -O3 -std=c++17
INCLUDE = .
LDFLAGS=
LIBS=
LIBDIRS=
SOURCES = \
	MatrixReader.cpp \
	master.cpp \
	worker.cpp \
	main.cpp

OBJECTS = $(SOURCES:.cpp=.o)

HEADERS = $(shell find . -name '*.hpp')
EXECUTABLE = ./matrix_mult

COMPILE_CMD = $(CPP) $(CFLAGS) $(addprefix -I,$(INCLUDE)) $(LDFLAGS) $(addprefix -L,$(LIBDIRS)) $(addprefix -l,$(LIBS)) $< -c -o $@
LINK_CMD = $(CPP) $(CFLAGS) $(addprefix -I,$(INCLUDE)) $(LDFLAGS) $(addprefix -L,$(LIBDIRS)) $(addprefix -l,$(LIBS)) $^ -o $@

.PHONY: all clear clean purge

all: $(EXECUTABLE)

# Building Targets

$(EXECUTABLE): $(OBJECTS)
	$(LINK_CMD)

$(OBJECTS): $(SOURCES) $(HEADERS)

%.o: %.cpp
	$(COMPILE_CMD)

#######################################

FILES = \
	Makefile \
	main.cpp \
	exception.hpp \
	MatrixReader.hpp \
	MatrixReader.cpp \
	common.hpp \
	worker.hpp \
	worker.cpp \
	master.hpp \
	master.cpp

SSH_CMD = ssh parlab

DATA_DIR = /mnt/home/_teaching/para/03-matrixmul-mpi/data
DEBUG_DATA_DIR = ~/matrix_mult_para/data
PROJECT_DIR = /mnt/home/s_bruchpa/matrix_mult_para

PARTITION = small-hp
N_PROCS = 256
N_HOSTS = 8
CORES = 2
TIMEOUT = 20:00
SRUN_CMD = srun -n $(N_PROCS) -N $(N_HOSTS) -p $(PARTITION) -c $(CORES) -m cyclic --time=$(TIMEOUT)

DEBUG_MATRIX_A = $(DEBUG_DATA_DIR)/matrix-r5-c5-s42.a
DEBUG_MATRIX_B = $(DEBUG_DATA_DIR)/matrix-r5-c5-s42.b
DEBUG_OUT = $(DEBUG_DATA_DIR)/result-parallel-debug
ARGS = $(DEBUG_MATRIX_A) $(DEBUG_MATRIX_B) $(DEBUG_OUT)

send_files:
	@echo "Sending files..."
	@scp -r -P 42222 $(FILES) "s_bruchpa@parlab.ms.mff.cuni.cz:$(PROJECT_DIR)"

compile: send_files
	@echo "Compiling..."
	@$(SSH_CMD) "cd $(PROJECT_DIR); srun make"

test:
	@echo "Running testing script..."
	@$(SSH_CMD) "cd $(PROJECT_DIR); ./test.sh"

run:
	@echo "Running $(EXECUTABLE) $(ARGS)"
	@$(SSH_CMD) "cd $(PROJECT_DIR); $(SRUN_CMD) $(EXECUTABLE) $(ARGS)"

# Cleaning Stuff

clear:
	@echo Removing all generated files...
	-@rm -f $(EXECUTABLE)

clean: clear

purge: clear

