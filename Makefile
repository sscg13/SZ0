EXE := SZ0
EVALFILE := sz0_small.onnx
ARCH := native
TUNE := native
DEBUG := no
GPU := yes
SUFFIX :=

ifeq ($(OS), Windows_NT)
	SUFFIX := .exe
endif

rwildcard = $(foreach d,$(wildcard $(1:=/*)),$(call rwildcard,$d,$2) $(filter $(subst *,%,$2),$d))

C_SRCS := $(call rwildcard,src,*.c)
CPP_SRCS := $(call rwildcard,src,*.cpp)

CPP_OBJS := $(patsubst %.cpp,%.o,$(CPP_SRCS))
C_OBJS := $(patsubst %.c,%.o,$(C_SRCS))
OBJS := $(CPP_OBJS) $(C_OBJS)

CXX := clang++
CC := clang

ifeq ($(CXX), g++)
	CC := gcc
endif

ifeq ($(GPU), yes)
	EVALFILE := sz0_small_batched.onnx
endif

ONNX_DIR := onnx
INCLUDES := -I$(ONNX_DIR)/include
LDFLAGS  := -L$(ONNX_DIR)/lib 
LDLIBS   := -lonnxruntime


ifeq ($(DEBUG), no)
	CXXFLAGS := -O3 -march=$(ARCH) -mtune=$(TUNE) -std=c++23 -pthread -DNNFILE=\"$(EVALFILE)\"
	CFLAGS := -O3 -march=$(ARCH) -mtune=$(TUNE)
else
	CXXFLAGS := -g -march=$(ARCH) -mtune=$(TUNE) -std=c++23 -pthread -DNNFILE=\"$(EVALFILE)\"
	CFLAGS := -g -march=$(ARCH) -mtune=$(TUNE)
endif

ifeq ($(GPU), yes)
    CXXFLAGS += -DUSE_CUDA
    INCLUDES += -I$(CUDNN_DIR)/include
    LDFLAGS  += -L$(CUDNN_DIR)/lib64 -Wl,-rpath,$(ONNX_DIR)/lib -Wl,-rpath,$(CONDA_PREFIX)/lib
endif

OUT := $(EXE)$(SUFFIX)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(EXE): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(OUT) $^ $(LDFLAGS) $(LDLIBS)
	@echo "Build complete. Run with ./$(OUT)"

clean:
	rm -f $(OBJS)