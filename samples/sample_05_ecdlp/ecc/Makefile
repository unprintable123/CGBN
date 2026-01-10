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



ampere: 
	g++ $(INC) $(LIB) -O3 ecc.h ecc_test.cpp -o ecc_test -lgmp
	g++ $(INC) $(LIB) -O3 ecc.h kangaroo.cpp -o kangaroo_test -lgmp
	nvcc $(INC) $(LIB) -O3 -I../../../include -arch=sm_80 ecc_test_gpu2.cu -o add_test -lgmp
	nvcc $(INC) $(LIB) -O3 -I../../../include -arch=sm_80 kangaroo_gpu.cu -o kangaroo_gpu -lgmp

