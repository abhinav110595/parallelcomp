# Location of the CUDA Toolkit
NVCC := /usr/local/cuda/bin/nvcc
CCFLAGS := -O2

build: squbitsim squbitsim2

squbitsim.o:squbitsim.cu
	$(NVCC) $(INCLUDES) $(CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

squbitsim: squbitsim.o
	$(NVCC) $(LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

run: build
	$(EXEC) ./squbitsim

clean:
	rm -f squbitsim *.o

squbitsim2.o:squbitsim2.cu
	$(NVCC) $(INCLUDES) $(CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

squbitsim2: squbitsim2.o
	$(NVCC) $(LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

run2: build
	$(EXEC) ./squbitsim2

clean2:
	rm -f squbitsim2 *.o
