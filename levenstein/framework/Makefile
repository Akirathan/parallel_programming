CPP=g++17
CFLAGS=-Wall -O3 -std=c++17 -mavx2 -fopenmp
INCLUDE=. ./internal
LDFLAGS=
LIBS=
LIBDIRS=
SOURCE=levenshtein.cpp
HEADERS=$(shell find . -name '*.hpp')
EXECUTABLE=./levenshtein


.PHONY: all clear clean purge

all: $(EXECUTABLE)



# Building Targets

$(EXECUTABLE): $(SOURCE) $(HEADERS)
	@$(CPP) $(CFLAGS) $(addprefix -I,$(INCLUDE)) $(LDFLAGS) $(addprefix -L,$(LIBDIRS)) $(addprefix -l,$(LIBS)) $< -o $@



FILES = \
	Makefile \
	implementation.hpp \
	levenshtein.cpp

PROJECT_DIR = "/mnt/home/s_bruchpa/levenstein_para/framework"
DATA_DIR = ~/levenstein_para/data
PARTITION = small-lp
SRUN_CMD = srun -p $(PARTITION)
SSH_CMD = ssh parlab
ARGS = $(DATA_DIR)/debug-4k-A $(DATA_DIR)/debug-4k-B
DATA_02_A = $(DATA_DIR)/02-16k-A
DATA_02_B = $(DATA_DIR)/02-16k-B
DATA_03_A = $(DATA_DIR)/03-64k-A
DATA_03_B = $(DATA_DIR)/03-64k-B
DATA_04_A = $(DATA_DIR)/04-48k-A
DATA_04_B = $(DATA_DIR)/04-64k-B
DATA_05_A = $(DATA_DIR)/05-64k-A
DATA_05_B = $(DATA_DIR)/05-48k-B

send_files:
	@echo "Sending files..."
	@scp -r -P 42222 $(FILES) "s_bruchpa@parlab.ms.mff.cuni.cz:$(PROJECT_DIR)"

compile: send_files
	@echo "Compiling..."
	@$(SSH_CMD) "cd $(PROJECT_DIR); $(SRUN_CMD) make"

run:
	@echo "Running $(EXECUTABLE) $(ARGS)"
	@$(SSH_CMD) "cd $(PROJECT_DIR); $(SRUN_CMD) $(EXECUTABLE) $(ARGS)"

run_all:
	@echo "Running $(EXECUTABLE) $(DATA_02_A) $(DATA_02_B)"
	@$(SSH_CMD) "cd $(PROJECT_DIR); $(SRUN_CMD) $(EXECUTABLE) $(DATA_02_A) $(DATA_02_B)"

	@echo "Running $(EXECUTABLE) $(DATA_03_A) $(DATA_03_B)"
	@$(SSH_CMD) "cd $(PROJECT_DIR); $(SRUN_CMD) $(EXECUTABLE) $(DATA_03_A) $(DATA_03_B)"

	@echo "Running $(EXECUTABLE) $(DATA_04_A) $(DATA_04_B)"
	@$(SSH_CMD) "cd $(PROJECT_DIR); $(SRUN_CMD) $(EXECUTABLE) $(DATA_04_A) $(DATA_04_B)"

	@echo "Running $(EXECUTABLE) $(DATA_05_A) $(DATA_05_B)"
	@$(SSH_CMD) "cd $(PROJECT_DIR); $(SRUN_CMD) $(EXECUTABLE) $(DATA_05_A) $(DATA_05_B)"

# Cleaning Stuff

clear:
	@echo Removing all generated files...
	-@rm -f $(EXECUTABLE)

clean: clear

purge: clear
