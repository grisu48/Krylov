/*
 * Krylov.hpp
 *
 *  Created on: Sep 10, 2015
 *      Author: phoenix
 */

#ifndef KRYLOV_SOLVER_HPP_
#define KRYLOV_SOLVER_HPP_

#include "LinSolver3D.hpp"
#include "matrix.H"

/**
 * Abstract class containing the methods for a Krylov linear solver.
 * Extends the virtual class Linsolver3D and adds some methods that are used for the Krylov methods
 */
class Krylov : public Linsolver3D {
public:
	Krylov();
	virtual ~Krylov();


protected:
	/**  Dimension of the solver. For the 3d solver it is always 3 */
	double delx[3];
	double DiffDiag[3];
	double breaktol, eps;

	/**
	 * Calculate the Residual out of the given matrices
	 * @param bounds Boundary handler for parallelisation
	 * @param phi
	 * @param rhs
	 * @param lambda
	 * @param residual
	 * @param Dxx
	 * @param Dyy
	 * @param Dzz
	 * @param Dxy
	 */
	void get_Residual(BoundaryHandler3D &bounds,
			NumMatrix<double,3> &phi, NumMatrix<double,3> &rhs,
			NumMatrix<double,3> &lambda,
			NumMatrix<double,3> &residual,
			NumMatrix<double,3> &Dxx, NumMatrix<double,3> &Dyy,
			NumMatrix<double,3> &Dzz, NumMatrix<double,3> &Dxy);

	/**
	 * Calculate the Residual out of the given matrices
	 * @param bounds Boundary handler for parallelisation
	 * @param phi
	 * @param rhs
	 * @param lambda
	 * @param residual
	 */
	void get_Residual(BoundaryHandler3D &bounds,
			NumMatrix<double,3> &phi, NumMatrix<double,3> &rhs,
			NumMatrix<double,3> &lambda,
			NumMatrix<double,3> &residual);
	/**
	 * Calculate the dot product of two given matrices
	 * @param vecA Matrix A
	 * @param vecB Matrix B
	 * @return the dot-product of matrix B and matrix B
	 */
	double dot_product(NumMatrix<double,3> &vecA,
			NumMatrix<double,3> &vecB);
//	static double dot_product(NumMatrix<double,3> &vecA,
//			NumMatrix<double,3> &vecB);
	/**
	 * Calculate the l2-norm of the given vector. The l2-norm or Eucledian Norm is the square root of the sum of the squares of the indices.
	 * (l2 = sqrt( sum( (x^2) ) )
	 * @param vec Vector of whom the l2 norm is calculated from
	 * @return the l2-norm of the given vector
	 */
//	static double get_l2Norm(NumMatrix<double,3> &vec);
	double get_l2Norm(NumMatrix<double,3> &vec);

	/**
	 * Multiply
	 * @param bounds
	 * @param vecIn
	 * @param lambda
	 * @param Dxx
	 * @param Dyy
	 * @param Dzz
	 * @param Dxy
	 * @param vecOut
	 * @param apply_bcs
	 */
	void multiply_withMat(BoundaryHandler3D &bounds,
			NumMatrix<double,3> &vecIn,
			NumMatrix<double,3> &lambda,
			NumMatrix<double,3> &Dxx,
			NumMatrix<double,3> &Dyy,
			NumMatrix<double,3> &Dzz,
			NumMatrix<double,3> &Dxy,
			NumMatrix<double,3> &vecOut,
			bool apply_bcs=true);

	/**
	 * Multiply
	 * @param bounds
	 * @param vecIn
	 * @param lambda
	 * @param vecOut
	 * @param apply_bcs
	 */
	void multiply_withMat(BoundaryHandler3D &bounds,
			NumMatrix<double,3> &vecIn,
			NumMatrix<double,3> &lambda,
			NumMatrix<double,3> &vecOut,
			bool apply_bcs=true);
};

#endif /* KRYLOV_HPP_ */
