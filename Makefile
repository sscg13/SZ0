EXE := SZ0
ARCH := native
TUNE := native
DEBUG := no
GPU := yes
# BATCH=yes builds the batched (in-flight rollout) search pipeline without
# CUDA, so GPU-level batching behavior can be tested with CPU inference.
BATCH := no
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

ONNX_DIR := onnx
INCLUDES := -I$(ONNX_DIR)/include
LDFLAGS  := -L$(ONNX_DIR)/lib 
LDLIBS   := -lonnxruntime


ifeq ($(DEBUG), no)
	CXXFLAGS := -O3 -march=$(ARCH) -mtune=$(TUNE) -std=c++23 -pthread
	CFLAGS := -O3 -march=$(ARCH) -mtune=$(TUNE)
else
	CXXFLAGS := -g -march=$(ARCH) -mtune=$(TUNE) -std=c++23 -pthread
	CFLAGS := -g -march=$(ARCH) -mtune=$(TUNE)
endif

# Track header dependencies so edits to consts.h etc. trigger recompiles;
# without this a header change silently links stale objects.
CXXFLAGS += -MMD -MP
CFLAGS += -MMD -MP
DEPS := $(OBJS:.o=.d)

ifeq ($(GPU), yes)
    CXXFLAGS += -DUSE_CUDA
    INCLUDES += -I$(CUDNN_DIR)/include
    LDFLAGS  += -L$(CUDNN_DIR)/lib64 -Wl,-rpath,$(ONNX_DIR)/lib -Wl,-rpath,$(CONDA_PREFIX)/lib
    # cudaMemcpy for the CUDA-graph path (SZ0_CUDA_GRAPH=1), which needs
    # device-resident IO. CUDA_HOME is set by the cuda module.
    INCLUDES += -I$(CUDA_HOME)/include
    LDFLAGS  += -L$(CUDA_HOME)/lib64 -Wl,-rpath,$(CUDA_HOME)/lib64
    LDLIBS   += -lcudart
endif

ifeq ($(BATCH), yes)
    CXXFLAGS += -DUSE_BATCHED_SEARCH
endif

# Rebuild all objects when the compile flags change (GPU / BATCH / DEBUG
# flips) — same .cpp, different code. Must sit below every CXXFLAGS +=.
FLAGSTAMP := .buildflags
$(shell [ "$$(cat $(FLAGSTAMP) 2>/dev/null)" = "$(CXXFLAGS)" ] || echo "$(CXXFLAGS)" > $(FLAGSTAMP))

OUT := $(EXE)$(SUFFIX)

%.o: %.cpp $(FLAGSTAMP)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

%.o: %.c $(FLAGSTAMP)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(EXE): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(OUT) $^ $(LDFLAGS) $(LDLIBS)
	@echo "Build complete. Run with ./$(OUT)"

clean:
	rm -f $(OBJS) $(DEPS) $(FLAGSTAMP)

-include $(DEPS)