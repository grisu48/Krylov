
#ifndef UBIK_BICGSTAB_H
#define UBIK_BICGSTAB_H

#include "matrix.H"
#include "grid_manager.H"
#include "BoundaryHandler.H"
#include "LinSolver3D.hpp"
#include "Krylov.H"

class BICGStab : public Krylov
{
public:
	BICGStab(grid_manager &TheGrid, double tol, int LValue,
#ifdef parallel
	         mpi_manager_3D &MyMPI,
#endif
	         bool spatial_diffusion=false, bool allow_offDiagDiffusion=false);
	// BICGStab(double epsilon, int LValue,
	//          grid_manager &TheGrid, bool spatial_diffusion);
	// solve(BoundaryHandler3D &bounds,
	//       NumMatrix<double,3> &psi,
	//       NumMatrix<double,3> &rhs,
	//       NumMatrix<double,3> &lambda,
	//       NumMatrix<double,3> &D_xx,
	//       NumMatrix<double,3> &D_yy,
	//       NumMatrix<double,3> &D_zz, int debug);
	virtual void solve(BoundaryHandler3D &bounds, NumMatrix<double,3> &phi,
	           NumMatrix<double,3> &rhs, NumMatrix<double,3> &lambda,
	           double D_xx, double D_yy, double D_zz, int debug=0);
	virtual void solve(BoundaryHandler3D &bounds, NumMatrix<double,3> &phi,
	           NumMatrix<double,3> &rhs, NumMatrix<double,3> &lambda,
	           NumMatrix<double,3> &Dxx, NumMatrix<double,3> &Dyy,
	           NumMatrix<double,3> &Dzz, NumMatrix<double,3> &Dxy,
	           int debug=0, bool use_offDiagDiffusion=false);
	virtual void solve_int(BoundaryHandler3D &bounds, NumMatrix<double,3> &phi,
	               NumMatrix<double,3> &rhs, NumMatrix<double,3> &lambda,
	               NumMatrix<double,3> &Dxx, NumMatrix<double,3> &Dyy,
	               NumMatrix<double,3> &Dzz, NumMatrix<double,3> &Dxy);
private:
	void make_Arrays(int mx, int my, int mz);

	void add_MatTimesVec(NumMatrix<double,3> &result,
	                     NumMatrix<double,3> &vec);

#ifdef parallel
	MPI_Comm comm3d;
#endif
	NumMatrix<double,3> *residuals, *uMat;
	NumMatrix<double,3> resTilde;
	//	NumMatrix<double,3> storage;

	int rank;
	bool use_spatialDiffusion, use_offDiagDiffusion;


};


#endif
