include $(HOME)/src/makeinclude/include.mak

OFLAGS  =  $(X_OFLAGS) -Wall -fexceptions 

CXXFLAGS = $(OFLAGS) -I.. $(X_INC) $(X_NO_AUTO_INCLUDE)

FFLAGS = $(OFLAGS)

LDFLAGS = $(X_LIB)

LOADLIBES = -lutil_mt -lmatrix_mt -lm\
	$(X_COMPLEX_LIB) $(X_HDF5_LIBS) 

TARGETS = PrimBICG PrimBICG_mpi

all:	$(TARGETS)

grid_manager.o: grid_manager.C grid_manager.H
	$(CXX) $(CXXFLAGS) -c $< -o $@

BoundaryHandler.o: BoundaryHandler.C BoundaryHandler.H
	$(CXX) $(CXXFLAGS) -c $< -o $@

BoundaryHandler_mpi.o: BoundaryHandler.C BoundaryHandler.H
	$(MPICXX) $(CXXFLAGS) -D parallel -c $< -o $@

Hdf5File.o: Hdf5File_cbase.C Hdf5File_cbase.H
	$(CXX) $(CXXFLAGS) -c $< -o $@

Hdf5File_mpi.o: Hdf5File_cbase.C Hdf5File_cbase.H
	$(MPICXX) $(CXXFLAGS) -D parallel -c $< -o $@

solveLin_BICGStab.o: solveLin_BICGStab.C
	$(CXX) $(CXXFLAGS) -c $< -o $@

solveLin_BICGStab_mpi.o: solveLin_BICGStab.C
	$(MPICXX) $(CXXFLAGS) -D parallel -c $< -o $@

main.o: main.C
	$(CXX) $(CXXFLAGS) -c $< -o $@

main_mpi.o: main.C
	$(MPICXX) $(CXXFLAGS) -D parallel -c $< -o $@

mpi_manager.o: mpi_manager.C mpi_manager.H
	$(MPICXX) $(CXXFLAGS) -c $< -o $@

PrimBICG: main.o  grid_manager.o BoundaryHandler.o Hdf5File.o solveLin_BICGStab.o
	$(CXX) $^ $(LDFLAGS) $(LOADLIBES) -o $@

PrimBICG_mpi: main_mpi.o  grid_manager.o BoundaryHandler_mpi.o Hdf5File_mpi.o solveLin_BICGStab_mpi.o mpi_manager.o
	$(MPICXX) $^ $(LDFLAGS) $(LOADLIBES) -o $@

clean:
	$(X_CLEAN) PrimBICG
