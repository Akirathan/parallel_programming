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
PARTITION = small-lp
SRUN_CMD = srun -p $(PARTITION)
SSH_CMD = ssh parlab
ARGS = ~/levenstein_para/data/...

send_files:
	@echo "Sending files..."
	@scp -r -P 42222 $(FILES) "s_bruchpa@parlab.ms.mff.cuni.cz:$(PROJECT_DIR)"

compile: send_files
	@echo "Compiling..."
	@$(SSH_CMD) "cd $(PROJECT_DIR); $(SRUN_CMD) make"

run:
	@$(SSH_CMD) "cd $(PROJECT_DIR); $(SRUN_CMD) $(EXECUTABLE) $(ARGS)"

# Cleaning Stuff

clear:
	@echo Removing all generated files...
	-@rm -f $(EXECUTABLE)

clean: clear

purge: clear
