/*
 * BiCGStabCL.hpp
 *
 *  Created on: Mar 23, 2015
 *      Author: phoenix
 */

#ifndef BICGSTAB_SOLVER_OPENCL_HPP_
#define BICGSTAB_SOLVER_OPENCL_HPP_

#define COMPARE_SOLVER 0

// Additional checks in the BiCGStab solver used for debugging
#ifndef BICGSTAB_SOLVER_ADDITIONAL_CHECKS
#define BICGSTAB_SOLVER_ADDITIONAL_CHECKS 1
#endif


#include "LinSolver3D.hpp"
#include "grid_manager.H"
#include "FlexCL.hpp"
#include "FlexCLMatrix.hpp"

#include <exception>
#include <string>

#if COMPARE_SOLVER == 1
#include "solveLin_BICGStab.H"
#endif


class NumException : public std::exception {
private:
	std::string _message;
public:
	NumException() {}
	NumException(std::string message) { this->_message = message; }
	NumException(const char* msg) { this->_message = std::string(msg); }
	virtual ~NumException() {}

	virtual const char* what() const _GLIBCXX_USE_NOEXCEPT { return this->_message.c_str(); }
};

class BiCGStabSolver: public Linsolver3D {
private:

#if COMPARE_SOLVER == 1
	BICGStab *compare_solver;
#endif

protected:
	/** OpenCL context for this accelerated solver */
	flexCL::Context *_context = NULL;

	grid_manager *grid;

	/** Status code of the current solver*/
	int status = 0;

	int tolerance;
	int lValue;

	size_t mx[3];
	double deltaX[3];

	// XXX: Used from old solver. Either document it or remove it!
	bool use_spatialDiffusion;
	bool use_offDiagDiffusion;
	int debug;

	/** Diffusion diagonals (D_xx, D_yy, D_zz) */
	double diffDiag[3];

	/* ==== OpenCL variables ==== */

	flexCL::CLMatrix3d *_matrix_rhs, *_matrix_lambda;
	flexCL::CLMatrix3d **_matrix_residuals;
	flexCL::CLMatrix3d **_uMat;
	flexCL::Program *_clProgram;
	flexCL::Kernel *_clKernelGenerateAx_Full;
	flexCL::Kernel *_clKernelGenerateAx_NoSpatial;
	flexCL::Kernel *_clKernelBoundary;


	bool checkMatrix(flexCL::CLMatrix3d *matrix);
	bool checkMatrix(flexCL::CLMatrix3d &matrix);


	void applyBoundary(flexCL::CLMatrix3d* matrix);
	/**
	 * Generate the Matrix A*x by using the input matrices phi, lambda and the diffusion matrices Dxx, Dyy, Dzz, Dxy
	 * The resulting is written in rhs (right-hand-side)
	 */
	void generateAx(flexCL::CLMatrix3d* phi, flexCL::CLMatrix3d* rhs, flexCL::CLMatrix3d* lambda, flexCL::CLMatrix3d* Dxx, flexCL::CLMatrix3d* Dyy, flexCL::CLMatrix3d* Dzz, flexCL::CLMatrix3d* Dxy);
	void generateAx(flexCL::CLMatrix3d* phi, flexCL::CLMatrix3d* rhs, flexCL::CLMatrix3d* lambda);

	void calculateResidual(flexCL::CLMatrix3d* residual, flexCL::CLMatrix3d* phi, flexCL::CLMatrix3d* rhs, flexCL::CLMatrix3d* lambda, flexCL::CLMatrix3d* Dxx, flexCL::CLMatrix3d* Dyy, flexCL::CLMatrix3d* Dzz, flexCL::CLMatrix3d* Dxy);
	void calculateResidual(flexCL::CLMatrix3d* residual, flexCL::CLMatrix3d* phi, flexCL::CLMatrix3d* rhs, flexCL::CLMatrix3d* lambda);

public:
	BiCGStabSolver(grid_manager &grid, double tolerance, int lValue, flexCL::Context* context);
	virtual ~BiCGStabSolver();

	/** This method initializes and sets the OpenCL context up */
	void setupContext(void);
	/** Cleanup the OpenCL context*/
	void cleanupContext(void);

	/** Checks if the context is initialized */
	bool isInitialized(void);


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

	virtual void test(void);

};

#endif /* BICGSTABCL_HPP_ */
