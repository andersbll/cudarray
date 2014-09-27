SRC_DIR = ./src
BUILD_DIR = ./build
INCLUDE_DIR = ./include
INSTALL_PREFIX = /usr/local

CC = g++
NVCC = nvcc
SRCS = $(SRC_DIR)/common.cpp
CUDA_SRCS = $(SRC_DIR)/elementwise.cu \
            $(SRC_DIR)/reduction.cu \
            $(SRC_DIR)/blas.cu \
            $(SRC_DIR)/image/img2win.cu

OBJS = $(SRCS:.cpp=.o) $(CUDA_SRCS:.cu=.o)


INCLUDES = -I$(INCLUDE_DIR)
C_FLAGS = -O3 -fPIC -Wall
NVCC_FLAGS = -arch=sm_35 --use_fast_math -O3 --compiler-options '-fPIC -Wall'
LDFLAGS = -lcudart -lcublas -lcufft -lcurand


libcudarray : $(OBJS)
	mkdir -p $(BUILD_DIR)
	$(CC) -shared $(C_FLAGS) -o $@ $^ $(LDFLAGS)

%.o : %.cpp
	$(CC) $(C_FLAGS) $(INCLUDES) -c -o $@ $<

%.o : %.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c -o $@ $<

all: libcudarray

.PHONY: install
install: libcudarray
	cp $(BUILD_DIR)/libcudarray.so $(INSTALL_PREFIX)/lib

uninstall:
	rm $(INSTALL_PREFIX)/lib/libcudarray.so

.PHONY: clean
clean:
	rm -f $(OBJS)
