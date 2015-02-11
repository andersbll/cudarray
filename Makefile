ifndef CUDA_PREFIX
  CUDA_PREFIX = /appl/cuda/6.5
endif
ifndef INSTALL_PREFIX
  INSTALL_PREFIX=/zhome/b9/d/42756/anaconda/envs/segment-GPU
endif


SRC_DIR = ./src

SRCS = $(SRC_DIR)/nnet/conv_bc01_matmul.cpp \
       $(SRC_DIR)/nnet/pool_b01.cpp \
       $(SRC_DIR)/nnet/cudnn.cpp \
       $(SRC_DIR)/nsnet/pool_seg_b01.cpp 


CUDA_SRCS = $(SRC_DIR)/elementwise.cu \
            $(SRC_DIR)/reduction.cu \
            $(SRC_DIR)/blas.cu \
            $(SRC_DIR)/random.cu \
            $(SRC_DIR)/image/img2win.cu \
            $(SRC_DIR)/nnet/one_hot.cu \
            $(SRC_DIR)/nsnet/one_hot.cu


INCLUDE_DIRS = ./include
INCLUDE_DIRS += $(CUDA_PREFIX)/include

ifneq ($(wildcard $(CUDA_PREFIX)/lib64),)
  # Use lib64 if it exists
  LIB_DIRS += $(CUDA_PREFIX)/lib64
endif
LIB_DIRS += $(CUDA_PREFIX)/lib
LIBS += cudart cublas cufft curand

ifeq ($(CUDNN_ENABLED), 1)
  C_FLAGS += -DCUDNN_ENABLED
  LIBS += cudnn
endif

CXX = g++
NVCC = $(CUDA_PREFIX)/bin/nvcc
BUILD_DIR = ./build
OBJS = $(SRCS:.cpp=.o) $(CUDA_SRCS:.cu=.o)
LIBCUDARRAY = libcudarray.so
LIBCUDARRAY_BUILD = $(BUILD_DIR)/$(LIBCUDARRAY)
LIBCUDARRAY_INSTALL = $(INSTALL_PREFIX)/lib/$(LIBCUDARRAY)

INCLUDES += $(foreach include_dir,$(INCLUDE_DIRS),-I$(include_dir))
C_FLAGS += -O3 -fPIC -Wall -Wfatal-errors
NVCC_FLAGS = -arch=sm_35 -O3 --compiler-options '$(C_FLAGS)' \
             --ftz=true --prec-div=false -prec-sqrt=false --fmad=true
LDFLAGS += $(foreach lib_dir,$(LIB_DIRS),-L$(lib_dir)) \
	       $(foreach lib,$(LIBS),-l$(lib))


$(LIBCUDARRAY_BUILD) : $(OBJS)
	mkdir -p $(BUILD_DIR)
	$(CXX) -shared $(C_FLAGS) -o $@ $^ $(LDFLAGS)

%.o : %.cpp
	$(CXX) $(C_FLAGS) $(INCLUDES) -c -o $@ $<

%.o : %.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c -o $@ $<

all: $(LIBCUDARRAY_BUILD)

$(LIBCUDARRAY_INSTALL) : $(LIBCUDARRAY_BUILD)
	cp $(LIBCUDARRAY_BUILD) $(LIBCUDARRAY_INSTALL)

install: $(INSTALL_PREFIX)/lib/$(LIBCUDARRAY)

uninstall:
	rm $(LIBCUDARRAY_INSTALL)

.PHONY: clean
clean:
	rm -f $(OBJS) $(LIBCUDARRAY)
