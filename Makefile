SRC_DIR = ./src
BUILD_DIR = ./build
INCLUDE_DIR = ./include

CC = g++
NVCC = nvcc
SRCS = $(SRC_DIR)/common.cpp
CUDA_SRCS = $(SRC_DIR)/elementwise.cu \
            $(SRC_DIR)/reduction.cu \
            $(SRC_DIR)/image/img2win.cu

OBJS = $(SRCS:.cpp=.o) $(CUDA_SRCS:.cu=.o)


INCLUDES = -I$(INCLUDE_DIR)
C_FLAGS = -O3 -fPIC
NVCC_FLAGS = -arch=sm_35 --use_fast_math -O --ptxas-options=-v --compiler-options '-fPIC'
LIBS = -lcudart -lcublas -lcufft -lcurand


$(BUILD_DIR)/libcudarray.so : $(OBJS)
	mkdir -p $(BUILD_DIR)
	$(CC) -shared $(C_FLAGS) -o $@ $^ $(LIBS)

%.o : %.cpp
	$(CC) $(C_FLAGS) $(INCLUDES) -c -o $@ $<

%.o : %.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c -o $@ $<

all: cudarray

.PHONY: clean

clean:
	rm -f $(OBJS)
