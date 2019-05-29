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
	@$(NVCC) $(addprefix -I,$(INCLUDE)) -arch sm_70 -lineinfo --compile -cudart static $< -o $@


FILES = \
	Makefile \
	implementation.hpp \
	kernels.cu \
	kernels.h \
	potential.cpp \
	try.cpp

INTERNAL_FILES = \
	internal/serial.hpp


PROJECT_DIR = "/mnt/home/s_bruchpa/potential/src"
PARTITION = volta-lp
SRUN_CMD = srun -p $(PARTITION) --gres=gpu:1
SSH_CMD = ssh gpulab
ARGS = -verify -verbose ~/potential/data/debug-v64-e256.gbf
CUCA_MEMCHECK = /usr/local/cuda/bin/cuda-memcheck
DATA_1 = ~/potential/data/v16k-e64k.gbf
DATA_2 = ~/potential/data/v32k-e256k.gbf
DATA_3 = ~/potential/data/v64k-e1024k.gbf

send_files:
	@echo "Sending files..."
	@scp -r -P 42222 $(FILES) "s_bruchpa@gpulab.ms.mff.cuni.cz:$(PROJECT_DIR)"
	@echo "Sending internal files..."
	@scp -r -P 42222 $(INTERNAL_FILES) "s_bruchpa@gpulab.ms.mff.cuni.cz:$(PROJECT_DIR)/internal"

compile: send_files
	@$(SSH_CMD) "cd $(PROJECT_DIR); $(SRUN_CMD) make"

run:
	@$(SSH_CMD) "cd $(PROJECT_DIR); $(SRUN_CMD) $(TARGET) $(ARGS)"

run_all:
	@echo "Running $(DATA_1)..."
	@$(SSH_CMD) "cd $(PROJECT_DIR); $(SRUN_CMD) $(TARGET) $(DATA_1)"
	@echo "Running $(DATA_2)..."
	@$(SSH_CMD) "cd $(PROJECT_DIR); $(SRUN_CMD) $(TARGET) $(DATA_2)"
	@echo "Running $(DATA_3)..."
	@$(SSH_CMD) "cd $(PROJECT_DIR); $(SRUN_CMD) $(TARGET) $(DATA_3)"

debug:
	@$(SSH_CMD) "cd $(PROJECT_DIR); $(SRUN_CMD) gdb --args $(TARGET) $(ARGS)"

memcheck:
	@$(SSH_CMD) "cd $(PROJECT_DIR); $(SRUN_CMD) $(CUDA_MEMCHECK) $(TARGET) $(ARGS)"



# Cleaning Stuff

clear:
	@echo Removing object files ...
	-@rm -f *.obj

clean: clear

purge: clear
	@echo Removing executable ...
	-@rm -f $(TARGETS)
