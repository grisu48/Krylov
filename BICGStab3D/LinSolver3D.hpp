/*
 * LinSolver3D.hpp
 * Interface class for any 3D linear solver
 */

#ifndef _LINEAR_SOLVER3D_HPP_
#define _LINEAR_SOLVER3D_HPP_

#include "matrix.H"
#include "grid_manager.H"
#include "BoundaryHandler.H"
#include "LinSolver3D.hpp"

/**
 * This class is the general interface class for all 3D linear solvers
 */
class Linsolver3D {
public:
	Linsolver3D() {dim=3; debug=0;}
	virtual ~Linsolver3D(){}

	virtual void setup(grid_1D &xGrid, grid_1D &yGrid, grid_1D &zGrid, double epsilon_,
			NumArray<int> solverPars,
			bool spatial_diffusion, bool allow_offDiagDiffusion,
#ifdef parallel
			 mpi_manager_3D &MyMPI,
#endif
			int maxIter=0) = 0;

	/** Solve with diagonal diffusion matrix */
	virtual void solve(BoundaryHandler3D &bounds, NumMatrix<double,3> &phi,
			NumMatrix<double,3> &rhs, NumMatrix<double,3> &lambda,
			double D_xx, double D_yy, double D_zz, int debug=0, double delt=0., bool evolve_time=false) = 0;
	/** Solve with diffusion matrix without off-diagonal elements*/
	virtual void solve(BoundaryHandler3D &bounds, NumMatrix<double,3> &phi,
			NumMatrix<double,3> &rhs, NumMatrix<double,3> &lambda,
			NumMatrix<double,3> &Dxx, NumMatrix<double,3> &Dyy,
			NumMatrix<double,3> &Dzz,
			int debug=0, double delt=0., bool evolve_time=false) = 0;
	/** Solve with arbitrary diffusion matrix */
	virtual void solve(BoundaryHandler3D &bounds, NumMatrix<double,3> &phi,
			NumMatrix<double,3> &rhs, NumMatrix<double,3> &lambda,
			NumMatrix<double,3> &Dxx, NumMatrix<double,3> &Dyy,
			NumMatrix<double,3> &Dzz, NumMatrix<double,3> &Dxy,
			int debug=0, bool use_offDiagDiffusion=false,
			double delt=0., bool evolve_time=false) = 0;

protected:
	virtual void solve_int(BoundaryHandler3D &bounds, NumMatrix<double,3> &phi,
			NumMatrix<double,3> &rhs, NumMatrix<double,3> &lambda,
			NumMatrix<double,3> &Dxx, NumMatrix<double,3> &Dyy,
			NumMatrix<double,3> &Dzz, NumMatrix<double,3> &Dxy, int debug=0,
			double delt=0., bool evolve_time=false) = 0;
	// Internal grid storage routine
	virtual void set_grid(grid_1D &xGrid, grid_1D &yGrid, grid_1D &zGrid) = 0;
#ifdef parallel
	MPI_Comm comm3d;
#endif
	int dim, rank;
	int debug;
};





#endif /* LINSOLVER3D_HPP_ */
