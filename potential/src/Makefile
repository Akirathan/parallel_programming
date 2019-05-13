CPP=g++17
NVCC=/usr/local/cuda/bin/nvcc
OPTIMALIZATIONS =
CFLAGS = -Wall -std=c++17 -g $(OPTIMALIZATIONS)
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
	@$(NVCC) $(addprefix -I,$(INCLUDE)) -arch sm_70 --compile -cudart static $< -o $@


FILES = \
	Makefile \
	implementation.hpp \
	kernels.cu \
	kernels.h \
	potential.cpp \
	try.cpp

PROJECT_DIR = "/mnt/home/s_bruchpa/potential/src"
PARTITION = volta-lp
SRUN_CMD = srun -p $(PARTITION) --gres=gpu:1
SSH_CMD = ssh gpulab
ARGS = -verify -verbose ~/potential/data/debug-v64-e256.gbf

send_files:
	@echo "Sending files..."
	@scp -r -P 42222 $(FILES) "s_bruchpa@gpulab.ms.mff.cuni.cz:$(PROJECT_DIR)"

compile: send_files
	@$(SSH_CMD) "cd $(PROJECT_DIR); $(SRUN_CMD) make"

run:
	@$(SSH_CMD) "cd $(PROJECT_DIR); $(SRUN_CMD) $(TARGET) $(ARGS)"

debug:
	@$(SSH_CMD) "cd $(PROJECT_DIR); $(SRUN_CMD) gdb --args $(TARGET) $(ARGS)"



# Cleaning Stuff

clear:
	@echo Removing object files ...
	-@rm -f *.obj

clean: clear

purge: clear
	@echo Removing executable ...
	-@rm -f $(TARGETS)
