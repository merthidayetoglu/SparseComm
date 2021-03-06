# ----- Make Macros -----

CXX = mpicxx
CXXFLAGS = -std=c++11 -fopenmp -I/usr/local/cuda/include
OPTFLAGS = -O3 

NVCC = nvcc
NVCCFLAGS = -lineinfo -O3 -std=c++11 -gencode arch=compute_80,code=sm_80 -ccbin=mpicxx -Xcompiler -fopenmp -Xptxas="-v"

LD_FLAGS = -ccbin=mpicxx -Xcompiler -fopenmp 

TARGETS = SparseComm
OBJECTS = main.o 

# ----- Make Rules -----

all:	$(TARGETS)

%.o: %.cpp
	${CXX} ${CXXFLAGS} ${OPTFLAGS} $< -c -o $@

%.o : %.cu
	${NVCC} ${NVCCFLAGS} $< -c -o $@

SparseComm: $(OBJECTS)
	$(NVCC) -o $@ $(OBJECTS) $(LD_FLAGS)

clean:
	rm -f $(TARGETS) *.o *.o.* *.txt *.bin core *.html *.xml
