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
	Linsolver3D() {}
	virtual ~Linsolver3D() {}

	virtual void solve(BoundaryHandler3D &bounds, NumMatrix<double,3> &phi,
			NumMatrix<double,3> &rhs, NumMatrix<double,3> &lambda,
			double D_xx, double D_yy, double D_zz, int debug=0) = 0;
	virtual void solve(BoundaryHandler3D &bounds, NumMatrix<double,3> &phi,
			NumMatrix<double,3> &rhs, NumMatrix<double,3> &lambda,
			NumMatrix<double,3> &Dxx, NumMatrix<double,3> &Dyy,
			NumMatrix<double,3> &Dzz, NumMatrix<double,3> &Dxy,
			int debug=0, bool use_offDiagDiffusion=false) = 0;
	virtual void solve_int(BoundaryHandler3D &bounds, NumMatrix<double,3> &phi,
			NumMatrix<double,3> &rhs, NumMatrix<double,3> &lambda,
			NumMatrix<double,3> &Dxx, NumMatrix<double,3> &Dyy,
			NumMatrix<double,3> &Dzz, NumMatrix<double,3> &Dxy) = 0;

	virtual void test(void) = 0;
};





#endif /* LINSOLVER3D_HPP_ */
