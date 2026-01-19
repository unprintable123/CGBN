ifdef GMP_HOME
  INC := -I$(GMP_HOME)/include
  LIB := -L$(GMP_HOME)/lib
endif
ifndef GMP_HOME
  INC :=
  LIB :=
endif

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
	rm -f ecc_test kangaroo_test add_test kangaroo_gpu

ampere: 
	nvcc $(INC) $(LIB) -O3 -I../../include -arch=sm_86 kangaroo_gpu.cu -o kangaroo_gpu -lgmp

ada:
	nvcc $(INC) $(LIB) -O3 -I../../include -arch=sm_89 kangaroo_gpu.cu -o kangaroo_gpu -lgmp
