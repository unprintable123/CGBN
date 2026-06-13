ifdef GMP_HOME
  INC := -I$(GMP_HOME)/include
  LIB := -L$(GMP_HOME)/lib
endif
ifndef GMP_HOME
  INC :=
  LIB :=
endif

SM ?= 86
NVCCFLAGS ?= -O3 -I../../include -arch=sm_$(SM)

pick:
	@echo
	@echo Please run one of the following:
	@echo "   make kepler"
	@echo "   make maxwell"
	@echo "   make pascal"
	@echo "   make volta"
	@echo "   make turing"
	@echo "   make ampere"
	@echo

clean:
	rm -f ecc_test kangaroo_test add_test kangaroo_gpu kangaroo_bench

ampere: 
	g++ $(INC) $(LIB) -O3 ecc_test.cpp -o ecc_test -lgmp
	g++ $(INC) $(LIB) -O3 kangaroo.cpp -o kangaroo_test -lgmp
	nvcc $(INC) $(LIB) -O3 -I../../include -arch=sm_86 ecc_test_gpu2.cu -o add_test -lgmp
	nvcc $(INC) $(LIB) -O3 -I../../include -arch=sm_86 kangaroo_gpu.cu -o kangaroo_gpu -lgmp

ada:
	nvcc $(INC) $(LIB) -O3 -I../../include -arch=sm_89 kangaroo_gpu.cu -o kangaroo_gpu -lgmp

kangaroo_bench:
	nvcc $(INC) $(LIB) $(NVCCFLAGS) -DKANGAROO_BENCH kangaroo_gpu.cu -o kangaroo_bench -lgmp
