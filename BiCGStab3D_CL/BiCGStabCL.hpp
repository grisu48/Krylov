/*
 * BiCGStabCL.hpp
 *
 *  Created on: Mar 23, 2015
 *      Author: phoenix
 */

#ifndef BICGSTAB_SOLVER_OPENCL_HPP_
#define BICGSTAB_SOLVER_OPENCL_HPP_

#include <exception>
#include <string>
#include <vector>

#include "LinSolver3D.hpp"
#include "grid_manager.H"
#include "FlexCL.hpp"
#include "FlexCLMatrix.hpp"


#if COMPARE_SOLVER == 1
#include "solveLin_BICGStab.H"
#endif


/**
 * Numerical exception
 */
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


/**
 * OpenCL driven BiCGStab solver
 * It is a extension of Linsolver3D, so it can be implemented in the PICARD code
 */
class BiCGStabSolver: public Linsolver3D {
private:

#if COMPARE_SOLVER == 1
	BICGStab *compare_solver;
#endif

protected:
	/** OpenCL context for this accelerated solver */
	flexCL::Context *_context = NULL;

	/** Grid manager, not used any more */
	grid_manager *grid;

	/** Status code of the current solver */
	int status = 0;

	/** Numerical tolerance when the solver exits */
	double tolerance;

	/** lValue*/
	int lValue;

	/** Verbosity flag*/
	bool verbose;

	/** Stored number of iterations used for solve */
	long _iterations;

	/** Maximum number of iterations or 0, if no limit exists */
	long _maxIterations = 0;

	/** Dimension of the problem */
	size_t mx[3];
	/** dx values for each dimension */
	double deltaX[3];

	// XXX: Used from old solver. Internal matrix status flags
	bool use_spatialDiffusion;
	bool use_offDiagDiffusion;
	int debug;

	/** Diffusion diagonals (D_xx, D_yy, D_zz) */
	double diffDiag[3];

	/** Minimum step time */
	long _steptime_min;
	/** Maximum step time */
	long _steptime_max;

	/** Step times */
	std::vector<long> stepTimes;


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

	void setVerbose(bool);
	long iterations(void);

	/** Get the iteration runtimes of all steps */
	std::vector<long> stepRuntimes(void);
	/** Get the iteration runtimes of all steps
	 * @param vector vector where the runtimes are written to
	 * */
	void stepRuntimes(std::vector<long> &vector);


	/**
	 * Setup method.
	 * This method is used instead of a constructor to allow the solver instance to be escaped.
	 * This is ugly but needed since it is planned to be implemented in code, that works this way
	 */
	virtual void setup(grid_1D &xGrid, grid_1D &yGrid, grid_1D &zGrid, double epsilon_,
			NumArray<int> &solverPars,
			bool spatial_diffusion, bool allow_offDiagDiffusion,
#ifdef parallel
			 mpi_manager_3D &MyMPI,
#endif
			int maxIter=0);

#if 0
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
#endif


	/** Set the advection matrix */
	virtual void set_Advection(NumMatrix<double,3> &ux_fine, BoundaryHandler3D &bounds, int dir);

	/** Solve with diagonal diffusion matrix */
	virtual void solve(BoundaryHandler3D &bounds, NumMatrix<double,3> &phi,
			NumMatrix<double,3> &rhs, NumMatrix<double,3> &lambda,
			double D_xx, double D_yy, double D_zz, int debug=0, double delt=0., bool evolve_time=false);
	/** Solve with diffusion matrix without off-diagonal elements*/
	virtual void solve(BoundaryHandler3D &bounds, NumMatrix<double,3> &phi,
			NumMatrix<double,3> &rhs, NumMatrix<double,3> &lambda,
			NumMatrix<double,3> &Dxx, NumMatrix<double,3> &Dyy,
			NumMatrix<double,3> &Dzz,
			int debug=0, double delt=0., bool evolve_time=false);
	/** Solve with arbitrary diffusion matrix */
	virtual void solve(BoundaryHandler3D &bounds, NumMatrix<double,3> &phi,
			NumMatrix<double,3> &rhs, NumMatrix<double,3> &lambda,
			NumMatrix<double,3> &Dxx, NumMatrix<double,3> &Dyy,
			NumMatrix<double,3> &Dzz, NumMatrix<double,3> &Dxy,
			int debug=0, bool use_offDiagDiffusion=false,
			double delt=0., bool evolve_time=false);

	/** Minimum step time for the calculation (Milliseconds) */
	long steptimeMin(void);

	/** Maximum step time for the calculation (Milliseconds) */
	long steptimeMax(void);

protected:
	/** Internal used solve method. This is the actual solver */
	virtual void solve_int(BoundaryHandler3D &bounds, NumMatrix<double,3> &phi,
			NumMatrix<double,3> &rhs, NumMatrix<double,3> &lambda,
			NumMatrix<double,3> &Dxx, NumMatrix<double,3> &Dyy,
			NumMatrix<double,3> &Dzz, NumMatrix<double,3> &Dxy, int debug=0);

	/** Set the grid. Inherited from Linsolver3D */
	virtual void set_Grid(grid_1D &xGrid, grid_1D &yGrid, grid_1D &zGrid);


};

#endif /* BICGSTABCL_HPP_ */
