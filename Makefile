NVCC := /usr/local/cuda-12.9/bin/nvcc

# temp
CPPFLAGS := -ccbin $(CONDA_PREFIX)/bin/x86_64-conda-linux-gnu-g++ -Wno-deprecated-gpu-targets

SRCDIR := src
OBJDIR := obj
TARGET := hull-white-mc

SRCS := $(shell find $(SRCDIR) -name "*.cu")
OBJS := $(patsubst $(SRCDIR)/%.cu,$(OBJDIR)/%.o,$(SRCS))
COREOBJS := $(filter-out $(OBJDIR)/main.o,$(OBJS))

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(CPPFLAGS) -o $@ $^

$(OBJDIR)/%.o: $(SRCDIR)/%.cu
	mkdir -p $(dir $@)
	$(NVCC) $(CPPFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJDIR)
	rm -f $(TARGET)
