
# This is the path where your AMD OpenCL SDK is located
#AMDSDK_PATH=/opt/AMDAPPSDK-3.0
# Choose the path that matches you. The follwoing lines are some examples:
#AMDSDK_PATH=/opt/AMDAPP
AMDSDK_PATH=/opt/intel-opencl-sdk
#AMDSDK_PATH=/opt/AMDAPPSDK
#AMDSDK_PATH=/opt/AMDAPP
#AMDSDK_PATH=/usr/local/cuda

# ==== COMPILER FLAGS ==== #
#O_FLAGS=-Wall -Wextra -pedantic -g -O0
O_FLAGS=-Wall -Wextra -pedantic -O3 -g
CXX=g++
CXX_FLAGS=$(O_FLAGS) -std=c++11


# ==== INCLUDE AND LIB ==== #
INCLUDE=-I$(AMDSDK_PATH)/include -L$(AMDSDK_PATH)/lib/x86_64 -I../Matrix -I../util -L../Matrix/Linux-amd64 -L../util/Linux-amd64
OCL_LIBS=-lOpenCL
# GLUT libs, needed if the library is compiled with OpenGL support
GLUT_LIBS=-lm -lGLU -lglut -lGLU -lGL -lGLEW
OBJS=FlexCL.o FlexCLMatrix.o grid_manager.o boundaryhandler.o bicgstab_cl.o
# bicgstab.o
LIBS=$(OCL_LIBS)#$(GLUT_LIBS)

# ==== User configuration ===== #
USER_LIBS=-lmatrix_mt




# ==== Add you own programs here ==== #
BINS=CL_Matrix matrix_mul performance bicgstab_cl





# ==== Makefile section ==== #

default:	all
all:	$(OBJS) $(BINS)
clean:
	rm -f *.o


# ==== PROGRAM SECTION ===== #
# Insert your program instructuions here


CL_Matrix:	CL_Matrix.cpp $(OBJS)
	$(CXX) $(CXX_FLAGS) -o matrix_cl $< $(OBJS) $(INCLUDE) $(LIBS) -pthread $(USER_LIBS)

matrix_mul:	matrix_mul.cpp $(OBJS)
	$(CXX) $(CXX_FLAGS) -o $@ $< $(OBJS) $(INCLUDE) $(LIBS) -pthread $(USER_LIBS)

performance:	performance.cpp $(OBJS)
	$(CXX) $(CXX_FLAGS) -o $@ $< $(OBJS) $(INCLUDE) $(LIBS) -pthread $(USER_LIBS) 

bicgstab_cl:	main.cpp main.hpp $(OBJS)
	$(CXX) $(CXX_FLAGS) -o $@ $< $(OBJS) $(INCLUDE) $(LIBS) -pthread $(USER_LIBS) 

# ==== Additional objects ==== #

FlexCL.o:	FlexCL.cpp FlexCL.hpp
	$(CXX) $(CXX_FLAGS) -c -o $@ $< $(INCLUDE)
FlexCLMatrix.o:	FlexCLMatrix.cpp FlexCLMatrix.hpp FlexCL.cpp FlexCL.hpp 
	$(CXX) $(CXX_FLAGS) -c -o $@ $< $(INCLUDE)

grid_manager.o: grid_manager.C grid_manager.H
	$(CXX) $(CXX_FLAGS) -c $< -o $@ $(INCLUDE)
	
boundaryhandler.o: BoundaryHandler.C BoundaryHandler.H
	$(CXX) $(CXX_FLAGS) -c $< -o $@ $(INCLUDE)

bicgstab_cl.o: BiCGStabCL.cpp BiCGStabCL.hpp
	$(CXX) $(CXX_FLAGS) -c $< -o $@ $(INCLUDE)
	
	
	
#bicgstab.o:	solveLin_BICGStab.C solveLin_BICGStab.H
#	$(CXX) $(CXX_FLAGS) -c $< -o $@ $(INCLUDE)
