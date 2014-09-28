SRC_DIR = ./src
BUILD_DIR = ./build
INCLUDE_DIR = ./include
ifndef INSTALL_PREFIX
  INSTALL_PREFIX=/usr/local
endif

CC = g++
NVCC = nvcc
SRCS = $(SRC_DIR)/common.cpp
CUDA_SRCS = $(SRC_DIR)/elementwise.cu \
            $(SRC_DIR)/reduction.cu \
            $(SRC_DIR)/blas.cu \
            $(SRC_DIR)/random.cu \
            $(SRC_DIR)/image/img2win.cu

OBJS = $(SRCS:.cpp=.o) $(CUDA_SRCS:.cu=.o)
LIBCUDARRAY = $(BUILD_DIR)/libcudarray.so

INCLUDES = -I$(INCLUDE_DIR)
C_FLAGS = -O3 -fPIC -Wall -Wfatal-errors
NVCC_FLAGS = -arch=sm_35 --use_fast_math -O3 --compiler-options '$(C_FLAGS)'
LDFLAGS = -lcudart -lcublas -lcufft -lcurand


$(LIBCUDARRAY) : $(OBJS)
	mkdir -p $(BUILD_DIR)
	$(CC) -shared $(C_FLAGS) -o $@ $^ $(LDFLAGS)

%.o : %.cpp
	$(CC) $(C_FLAGS) $(INCLUDES) -c -o $@ $<

%.o : %.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c -o $@ $<

all: $(LIBCUDARRAY)

.PHONY: install
install: $(LIBCUDARRAY)
	cp $(LIBCUDARRAY) $(INSTALL_PREFIX)/lib

uninstall:
	rm $(INSTALL_PREFIX)/lib/libcudarray.so

.PHONY: clean
clean:
	rm -f $(OBJS) $(LIBCUDARRAY)
