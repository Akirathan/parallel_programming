TBBROOT=/usr/local/tbb
CPP=g++17
CFLAGS=-Wall -O3 -msse4.2 -std=c++17
INCLUDE=$(TBBROOT)/include
LDFLAGS=
LIBS=tbb
LIBDIRS=$(TBBROOT)/lib
SOURCES=tbb-matrix-tran.cpp
HEADERS=$(shell find . -name '*.hpp')
EXECUTABLE=./tbb-matrix-tran


.PHONY: all clear clean purge

all: $(EXECUTABLE)



# Building Targets

$(EXECUTABLE): $(SOURCES) $(HEADERS)
	@echo Compiling and linking executable "$@" ...
	@echo $(TBBROOT)
	@$(CPP) $(CFLAGS) $(addprefix -I,$(INCLUDE)) $(LDFLAGS) $(addprefix -L,$(LIBDIRS)) $(addprefix -l,$(LIBS)) $(SOURCES) -o $@

# Cleaning Stuff

clear:
	@echo Removing all generated files...
	-@rm -f $(EXECUTABLE)

clean: clear

purge: clear
