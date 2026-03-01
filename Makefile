EXE := SZ0
ARCH := native
TUNE := native
DEBUG := no

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

ONNX_DIR := src/onnx/win
INCLUDES := -I$(ONNX_DIR)/include
LDFLAGS  := -L$(ONNX_DIR)/lib
LDLIBS   := -lonnxruntime

ifeq ($(DEBUG), no)
	CXXFLAGS := -O3 -march=$(ARCH) -mtune=$(TUNE) -std=c++23 
	CFLAGS := -O3 -march=$(ARCH) -mtune=$(TUNE)
else
	CXXFLAGS := -g -march=$(ARCH) -mtune=$(TUNE) -std=c++23
	CFLAGS := -g -march=$(ARCH) -mtune=$(TUNE)
endif

SUFFIX :=

ifeq ($(OS), Windows_NT)
	SUFFIX := .exe
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