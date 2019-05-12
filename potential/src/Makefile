CPP=g++17
NVCC=/usr/local/cuda/bin/nvcc
CFLAGS=-Wall -O3 -std=c++17
INCLUDE=/usr/local/cuda/include ./internal .
LDFLAGS=
LIBS=
LIBDIRS=/usr/local/cuda/lib64
HEADERS=$(shell find . -name '*.hpp')
TARGET=potential
#TARGET=try


.PHONY: all clear clean purge

all: $(TARGET)



# Building Targets

$(TARGET): $(TARGET).cpp $(HEADERS) kernels.obj
	@echo Compiling and linking executable "$@" ...
	@$(CPP) $(CFLAGS) $(addprefix -I,$(INCLUDE)) $(LDFLAGS) $(addprefix -L,$(LIBDIRS)) $(addprefix -l,$(LIBS)) -lcudart kernels.obj $< -o $@


kernels.obj: kernels.cu
	@echo Compiling CUDA kernels "$@" ...
	@$(NVCC) $(addprefix -I,$(INCLUDE)) --compile -cudart static $< -o $@


FILES = \
	Makefile \
	implementation.hpp \
	kernels.cu \
	kernels.h \
	potential.cpp \
	try.cpp

PROJECT_DIR = "/mnt/home/s_bruchpa/potential/src"
PARTITION = volta-lp
SSH_CMD = ssh gpulab
ARGS = -verify -verbose ~/potential/data/debug-v64-e256.gbf

send_files:
	@echo "Sending files..."
	@scp -r -P 42222 $(FILES) "s_bruchpa@gpulab.ms.mff.cuni.cz:$(PROJECT_DIR)"

compile: send_files
	@$(SSH_CMD) "cd $(PROJECT_DIR); srun -p $(PARTITION) --gres=gpu:1 make"

run:
	@$(SSH_CMD) "cd $(PROJECT_DIR); srun -p $(PARTITION) --gres=gpu:1 $(TARGET) $(ARGS)"
	


# Cleaning Stuff

clear:
	@echo Removing object files ...
	-@rm -f *.obj

clean: clear

purge: clear
	@echo Removing executable ...
	-@rm -f $(TARGETS)
