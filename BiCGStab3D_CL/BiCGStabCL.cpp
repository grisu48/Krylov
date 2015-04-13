/*
 * BiCGStabSolver.cpp
 *
 *  Created on: Mar 23, 2015
 *      Author: phoenix
 */

#include "BiCGStabCL.hpp"
#include <iostream>




using namespace std;
using namespace flexCL;


/* Status codes */
#define _STATUS_UNINITIALIZED 0
#define _STATUS_READY 128

/* Kernel filename */
#define KERNEL_FILENAME "bicgstab_kernel.cl"

#define VERBOSE 1
#define PROFILING 1




#define DELETE(x) { if(x!=NULL) delete x; x = NULL; }
#define epsilon 1e-3
#define REAL_EQUAL(x,y) { fabs(x-y)<epsilon*(y/x) }


using namespace std;
using namespace flexCL;


BiCGStabSolver::BiCGStabSolver(grid_manager &grid, double tolerance, int lValue, flexCL::Context* context) {
	this->_context = context;
	this->status = _STATUS_UNINITIALIZED;

	this->grid = &grid;
	this->lValue = lValue;
	this->tolerance = tolerance;
	this->debug = 0;
	this->use_offDiagDiffusion = false;
	this->use_spatialDiffusion = false;
	this->_matrix_rhs = NULL;
	this->_matrix_lambda = NULL;
	this->_matrix_residuals = NULL;
	this->_uMat = NULL;

	for(int i=0;i<3;i++) {
		this->mx[i] = (size_t)grid.get_mx(i);
		this->deltaX[i] = grid.get_delx(i);
	}

	this->_clProgram = NULL;
	this->_clKernelGenerateAx_Full = NULL;
	this->_clKernelGenerateAx_NoSpatial = NULL;
	this->_clKernelBoundary = NULL;

#if COMPARE_SOLVER == 1
	this->compare_solver = new BICGStab(grid, tolerance, lValue);
#endif

}

BiCGStabSolver::~BiCGStabSolver() {
	this->cleanupContext();

#if COMPARE_SOLVER == 1
	DELETE(this->compare_solver);
#endif
}
bool BiCGStabSolver::isInitialized(void) { return this->status == _STATUS_READY; }

void BiCGStabSolver::setupContext(void) {
	//cleanupContext();
#if VERBOSE == 1
	cout << "BiCGStabSolver::setupContext(...)" << endl;
#endif
#if PROFILING == 1
	this->_context->createProfilingCommandQueue();
#endif

	const size_t mx = (size_t)this->mx[0];
	const size_t my = (size_t)this->mx[1];
	const size_t mz = (size_t)this->mx[2];

	this->_matrix_rhs = new CLMatrix3d(this->_context, mx,my,mz);
	this->_matrix_rhs->initializeContext();
	this->_matrix_lambda = new CLMatrix3d(this->_context, mx,my,mz);
	this->_matrix_lambda->initializeContext();

	this->_matrix_residuals = new CLMatrix3d*[this->lValue+1];
	this->_uMat = new CLMatrix3d*[this->lValue+1];
	for(int i=0;i<this->lValue+1;i++) {
		this->_matrix_residuals[i] = new CLMatrix3d(this->_context, mx,my,mz);
		this->_uMat[i] = new CLMatrix3d(this->_context, mx,my,mz);

		this->_matrix_residuals[i]->initializeContext();
		this->_uMat[i]->initializeContext();

		this->_matrix_residuals[i]->clear();
		this->_uMat[i]->clear();
	}

	this->_clProgram = this->_context->createProgramFromSourceFile(KERNEL_FILENAME);
	this->_clKernelGenerateAx_Full = this->_clProgram->createKernel("generateAx_Full");
	this->_clKernelGenerateAx_NoSpatial = this->_clProgram->createKernel("generateAx_NoSpatial");
	this->_clKernelBoundary = this->_clProgram->createKernel("boundary");

	this->status = _STATUS_READY;
}

void BiCGStabSolver::cleanupContext(void) {
#if VERBOSE == 1
	cout << "BiCGStabSolver::cleanupContext(...)" << endl;
#endif
	DELETE(this->_matrix_rhs);
	DELETE(this->_matrix_lambda);

	if(this->_matrix_residuals != NULL) {
		for(int i=0;i<this->lValue+1;i++) {
			DELETE(this->_matrix_residuals[i]);
		}
		delete[] this->_matrix_residuals;
		this->_matrix_residuals = NULL;
	}
	if(this->_uMat != NULL) {
		for(int i=0;i<this->lValue+1;i++) {
			DELETE(this->_uMat[i]);
		}
		delete[] this->_uMat;
		this->_uMat = NULL;
	}
}



void BiCGStabSolver::solve(BoundaryHandler3D &bounds, NumMatrix<double,3> &phi,
		NumMatrix<double,3> &rhs, NumMatrix<double,3> &lambda,
		double D_xx, double D_yy, double D_zz, int debug) {

	this->debug = debug;
	this->use_spatialDiffusion = false;
	diffDiag[0] = D_xx;
	diffDiag[1] = D_yy;
	diffDiag[2] = D_zz;

	solve_int(bounds, phi, rhs, lambda, rhs, rhs, rhs, rhs);

}
void BiCGStabSolver::solve(BoundaryHandler3D &bounds, NumMatrix<double,3> &phi,
		NumMatrix<double,3> &rhs, NumMatrix<double,3> &lambda,
		NumMatrix<double,3> &Dxx, NumMatrix<double,3> &Dyy,
		NumMatrix<double,3> &Dzz, NumMatrix<double,3> &Dxy,
		int debug, bool use_offDiagDiffusion) {

	this->use_spatialDiffusion = true;
	this->use_offDiagDiffusion = use_offDiagDiffusion;
	this->debug = debug;
	if(!use_offDiagDiffusion) {
		Dxy.clear();
	}

	solve_int(bounds, phi, rhs, lambda, Dxx, Dyy, Dzz, Dxy);

}


/** Transfers the given NumMatrix to the given OpenCL context */
static CLMatrix3d* transferMatrix(Context *context, NumMatrix<double,3> &matrix, CLMatrix3d *copyContextFrom, size_t* size) {
	size_t _size[3] = { size[0], size[1], size[2] };

	// +2 because we want 1 ghost cell in each dimension
	Matrix3d temp(_size[0]+2, _size[1]+2, _size[2]+2);
	temp.clear();


	for(size_t ix=0; ix<size[0]; ix++)
		for(size_t iy=0; iy<size[0]; iy++)
			for(size_t iz=0; iz<size[0]; iz++)
				temp(ix+1,iy+1,iz+1) = matrix(ix,iy,iz);




	CLMatrix3d* result = temp.transferToDevice(context);
	//result->initializeContext();
	return result;
}

void BiCGStabSolver::applyBoundary(CLMatrix3d* matrix) {
	size_t size[3] = {matrix->mx(0),matrix->mx(1), matrix->mx(2)};

	this->_clKernelBoundary->setArgument(0, matrix->clMem());
	this->_clKernelBoundary->setArgument(1, size[0]+2);
	this->_clKernelBoundary->setArgument(2, size[0]+2);
	this->_clKernelBoundary->setArgument(3, size[0]+2);
	this->_clKernelBoundary->enqueueNDRange(size[0]+2, size[1]+2, size[2]+2);

}

void BiCGStabSolver::generateAx(flexCL::CLMatrix3d* phi, flexCL::CLMatrix3d* dst, flexCL::CLMatrix3d* lambda, flexCL::CLMatrix3d* Dxx, flexCL::CLMatrix3d* Dyy, flexCL::CLMatrix3d* Dzz, flexCL::CLMatrix3d* Dxy) {
	this->_clKernelGenerateAx_Full->setArgument(0, phi->clMem());
	this->_clKernelGenerateAx_Full->setArgument(1, lambda->clMem());
	this->_clKernelGenerateAx_Full->setArgument(2, Dxx->clMem());
	this->_clKernelGenerateAx_Full->setArgument(3, Dyy->clMem());
	this->_clKernelGenerateAx_Full->setArgument(4, Dzz->clMem());
	this->_clKernelGenerateAx_Full->setArgument(5, Dxy->clMem());
	this->_clKernelGenerateAx_Full->setArgument(6, dst->clMem());

	this->_clKernelGenerateAx_Full->setArgument(7, this->mx[0]+2);
	this->_clKernelGenerateAx_Full->setArgument(8, this->mx[1]+2);
	this->_clKernelGenerateAx_Full->setArgument(9, this->mx[2]+2);

	this->_clKernelGenerateAx_Full->setArgument(10, this->deltaX[0]);
	this->_clKernelGenerateAx_Full->setArgument(11, this->deltaX[1]);
	this->_clKernelGenerateAx_Full->setArgument(12, this->deltaX[2]);

	this->_clKernelGenerateAx_NoSpatial->enqueueNDRange(this->mx[0], this->mx[1], this->mx[2]);

#if PROFILING == 1
	this->_context->join();
	const unsigned long runtime_ms = this->_clKernelGenerateAx_Full->runtime() * 1e-6;
	cerr << "generateAx -- " << runtime_ms << " ms" << endl;
#endif

	applyBoundary(dst);
}

void BiCGStabSolver::generateAx(flexCL::CLMatrix3d* phi, flexCL::CLMatrix3d* dst, flexCL::CLMatrix3d* lambda) {
	this->_clKernelGenerateAx_NoSpatial->setArgument(0, phi->clMem());
	this->_clKernelGenerateAx_NoSpatial->setArgument(1, lambda->clMem());
	this->_clKernelGenerateAx_NoSpatial->setArgument(2, dst->clMem());

	this->_clKernelGenerateAx_NoSpatial->setArgument(3, this->mx[0]+2);
	this->_clKernelGenerateAx_NoSpatial->setArgument(4, this->mx[1]+2);
	this->_clKernelGenerateAx_NoSpatial->setArgument(5, this->mx[2]+2);

	this->_clKernelGenerateAx_NoSpatial->setArgument(6, this->deltaX[0]);
	this->_clKernelGenerateAx_NoSpatial->setArgument(7, this->deltaX[1]);
	this->_clKernelGenerateAx_NoSpatial->setArgument(8, this->deltaX[2]);

	this->_clKernelGenerateAx_NoSpatial->enqueueNDRange(this->mx[0], this->mx[1], this->mx[2]);

#if PROFILING == 1
	this->_context->join();
	const unsigned long runtime_ms = this->_clKernelGenerateAx_NoSpatial->runtime() * 1e-6;
	cerr << "PROFILING: generateAx -- " << runtime_ms << " ms" << endl;
#endif

	applyBoundary(dst);
}

bool BiCGStabSolver::checkMatrix(flexCL::CLMatrix3d &matrix) {
	return this->checkMatrix(&matrix);
}

bool BiCGStabSolver::checkMatrix(flexCL::CLMatrix3d *matrix) {
	flexCL::Matrix3d *m = matrix->transferToHost();
	bool result = true;
	if(m->hasNanValues()) result = false;
	delete m;
	return result;
}



void BiCGStabSolver::calculateResidual(flexCL::CLMatrix3d* residual, flexCL::CLMatrix3d* phi, flexCL::CLMatrix3d* rhs, flexCL::CLMatrix3d* lambda, flexCL::CLMatrix3d* Dxx, flexCL::CLMatrix3d* Dyy, flexCL::CLMatrix3d* Dzz, flexCL::CLMatrix3d* Dxy) {
	this->generateAx(phi, residual, lambda, Dxx, Dyy, Dzz, Dxy);
	residual->add(rhs);
	applyBoundary(residual);

#if PROFILING == 1
	this->_context->join();
	cerr << "PROFILING: calculated Residual" << endl;
#endif
}

void BiCGStabSolver::calculateResidual(flexCL::CLMatrix3d* residual, flexCL::CLMatrix3d* phi, flexCL::CLMatrix3d* rhs, flexCL::CLMatrix3d* lambda) {
	this->generateAx(phi, residual, lambda);
	residual->add(rhs);
	applyBoundary(residual);

#if PROFILING == 1
	this->_context->join();
	cerr << "PROFILING: calculated Residual" << endl;
#endif
}

void BiCGStabSolver::solve_int(BoundaryHandler3D &bounds,
		NumMatrix<double,3> &phi,
		NumMatrix<double,3> &rhs,
		NumMatrix<double,3> &lambda,
		NumMatrix<double,3> &Dxx,
		NumMatrix<double,3> &Dyy,
		NumMatrix<double,3> &Dzz,
		NumMatrix<double,3> &Dxy) {
#if VERBOSE == 1
	cout << "BiCGStabSolver::solve_int(...)" << endl;
#endif
	if(!isInitialized()) this->setupContext();

	/*
	 * -- Problem description: --
	 *
	 */

	// Transfer matrices to OpenCL context
	CLMatrix3d *cl_phi = transferMatrix(this->_context, phi, this->_matrix_rhs, this->mx);
	CLMatrix3d *cl_rhs = transferMatrix(this->_context, rhs, this->_matrix_rhs, this->mx);
	CLMatrix3d *cl_lambda = transferMatrix(this->_context, lambda, this->_matrix_rhs, this->mx);
	CLMatrix3d *cl_Dxx = NULL;
	CLMatrix3d *cl_Dyy = NULL;
	CLMatrix3d *cl_Dzz = NULL;
	CLMatrix3d *cl_Dxy = NULL;
	if(use_spatialDiffusion) {
		cl_Dxx = transferMatrix(this->_context, Dxx, this->_matrix_rhs, this->mx);
		cl_Dyy = transferMatrix(this->_context, Dyy, this->_matrix_rhs, this->mx);
		cl_Dzz = transferMatrix(this->_context, Dzz, this->_matrix_rhs, this->mx);
		cl_Dxy = transferMatrix(this->_context, Dxy, this->_matrix_rhs, this->mx);
	}
	CLMatrix3d *cl_resTilde = new CLMatrix3d(this->_context, cl_phi->mx(0), cl_phi->mx(1), cl_phi->mx(2));
	cl_resTilde->initializeContext();
	cl_resTilde->clear();

	this->_context->join();
	cout << "Matrices transferred to host" << endl;

#if BICGSTAB_SOLVER_ADDITIONAL_CHECKS == 1
	if(!this->checkMatrix(cl_phi)) throw "Initial matrix check failed (phi)";
	if(!this->checkMatrix(cl_rhs)) throw "Initial matrix check failed (rhs)";
	if(!this->checkMatrix(cl_lambda)) throw "Initial matrix check failed (lambda)";
	if(!this->checkMatrix(cl_resTilde)) throw "Initial matrix check failed (res Tilde)";
	if(use_spatialDiffusion) {
		if(!this->checkMatrix(cl_Dxx)) throw "Initial matrix check failed (Dxx)";
		if(!this->checkMatrix(cl_Dyy)) throw "Initial matrix check failed (Dyy)";
		if(!this->checkMatrix(cl_Dzz)) throw "Initial matrix check failed (Dzz)";
		if(!this->checkMatrix(cl_Dxy)) throw "Initial matrix check failed (Dxy)";
	}
#endif

	try {
		unsigned long iterations = 0;
		double normRhs = cl_rhs->l2Norm();

		if(normRhs < 1e-9) normRhs = 1.0;

		cout << "  normRHS = " << normRhs << endl;

		// Mathematical variables
		double rho0 = 1.0;
		double rho1 = 1.0;;
		double alpha = 0.0;
		double omega = 1.0;
		double norm = 0.0;

		double tau[lValue+1][lValue+1];
		double sigma[lValue+1], gammap[lValue+1], gammapp[lValue+1];
		double gamma[lValue+1];

		// Calculate r_0

		if(use_spatialDiffusion) {
			this->calculateResidual(this->_matrix_residuals[0], cl_phi, cl_rhs, cl_lambda, cl_Dxx, cl_Dyy, cl_Dzz, cl_Dxy);
		} else {
			this->calculateResidual(this->_matrix_residuals[0], cl_phi, cl_rhs, cl_lambda);
		}
#if BICGSTAB_SOLVER_ADDITIONAL_CHECKS == 1
		if(!this->checkMatrix(this->_matrix_residuals[0])) throw "matrix_residual check 0 failed";
#endif

		do {
			iterations++;
			cout << "Starting iteration " << iterations << " ... " << endl;

			rho0 *= -omega;

			// ==== BI-CG PART ============================================= //
			cout << "BI-CG part of the solver ... " << endl;

			for(int jj=0; jj<lValue; ++jj) {

#if BICGSTAB_SOLVER_ADDITIONAL_CHECKS == 1
		if(!this->checkMatrix(this->_matrix_residuals[jj])) throw "matrix_residual check 1 failed";
#endif
				rho1 = this->_matrix_residuals[jj]->dotProduct(cl_resTilde);
				//rho1 = dot_product(residuals[jj], resTilde);
				double beta = alpha*rho1/rho0;
				rho0 = rho1;

				for(int ii=0; ii<=jj; ++ii) {
					_uMat[ii]->mul(-beta);
					_uMat[ii]->add(_matrix_residuals[ii]);
#if BICGSTAB_SOLVER_ADDITIONAL_CHECKS == 1
		if(!this->checkMatrix(this->_uMat[ii])) throw "_uMat check 2 failed";
#endif
				}

				// u_(j+1) = A*u_j
				if(use_spatialDiffusion)
					generateAx(_uMat[jj], _uMat[jj+1], cl_lambda, cl_Dxx, cl_Dyy, cl_Dzz, cl_Dxy);
				else
					generateAx(_uMat[jj], _uMat[jj+1], cl_lambda);

#if BICGSTAB_SOLVER_ADDITIONAL_CHECKS == 1
		if(!this->checkMatrix(this->_uMat[jj+1])) throw "_uMat check 3 failed";
#endif

				alpha = rho0/(_uMat[jj+1]->dotProduct(cl_resTilde));

				for(int ii=0; ii<=jj; ++ii) {
					_matrix_residuals[ii]->subMultiplied(_uMat[ii+1], alpha);
#if BICGSTAB_SOLVER_ADDITIONAL_CHECKS == 1
		if(!this->checkMatrix(this->_matrix_residuals[ii])) throw "matrix_residual check 4 failed";
#endif
				}


				if(use_spatialDiffusion) {
					generateAx(_matrix_residuals[jj], _matrix_residuals[jj+1], cl_lambda, cl_Dxx, cl_Dyy, cl_Dzz, cl_Dxy);
				} else {
					generateAx(_matrix_residuals[jj], _matrix_residuals[jj+1], cl_lambda);
				}
#if BICGSTAB_SOLVER_ADDITIONAL_CHECKS == 1
		if(!this->checkMatrix(this->_matrix_residuals[jj])) throw "matrix_residual check 5 failed";
#endif

				cl_phi->addMultiplied(_uMat[0], alpha);

#if BICGSTAB_SOLVER_ADDITIONAL_CHECKS == 1
		if(!this->checkMatrix(this->_uMat[0])) throw "_uMat check 6 failed";
#endif
			}			// END BiCG PART


			// ==== MR PART ================================================ //
			cout << "MR part of the solver ... " << endl;
			for(int jj=1; jj<=lValue; ++jj) {
				for(int ii=1; ii<jj; ++ii) {
					tau[ii][jj] = _matrix_residuals[jj]->dotProduct(_matrix_residuals[ii]) / sigma[ii];
					_matrix_residuals[jj]->subMultiplied(_matrix_residuals[ii], tau[ii][jj]);
				}


				sigma[jj] = _matrix_residuals[jj]->dotProduct(_matrix_residuals[jj]);
				gammap[jj] = _matrix_residuals[0]->dotProduct(_matrix_residuals[jj])/sigma[jj];
			}

			omega = gamma[lValue] = gammap[lValue];
			cout << omega << endl;

			for(int jj=lValue-1; jj>=1; --jj) {
				gamma[jj] = gammap[jj];
				for(int ii=jj+1; ii<=lValue; ++ii) {
					gamma[jj] -= tau[jj][ii]*gamma[ii];
				}
			}

			for(int jj=1; jj<lValue; ++jj) {
				gammapp[jj] = gamma[jj+1];
				for(int ii=jj+1; ii<lValue; ++ii) {
					gammapp[jj] += tau[jj][ii]*gamma[ii+1];
				}
			}


			// Check break conditions:
			cl_phi->addMultiplied(_matrix_residuals[0], gamma[1]);
			_matrix_residuals[0]->subMultiplied(_matrix_residuals[lValue], gammap[lValue]);
			_uMat[0]->subMultiplied(_uMat[lValue], gamma[lValue]);

			for(int jj=1; jj<lValue; ++jj) {
				_uMat[0]->subMultiplied(_uMat[jj],gamma[jj]);
				cl_phi->addMultiplied(_matrix_residuals[jj],gammapp[jj]);
				_matrix_residuals[0]->subMultiplied(_matrix_residuals[jj],gammap[jj]);
			}

			norm = _matrix_residuals[0]->l2Norm();

			cout << "Iteration " << iterations << ": NORM = " << norm << endl;

			if(iterations > 100) {
				cout << "EMERGENCY BREAK (no_iteration = " << iterations << ")" << endl;
				break;
			}
		} while(norm > tolerance*normRhs);

		cout << "Completed after " << iterations << " iterations" << endl;



	} catch (...) {
		// Emergency cleanup
		DELETE(cl_phi);
		DELETE(cl_rhs);
		DELETE(cl_lambda);
		DELETE(cl_Dxx);
		DELETE(cl_Dyy);
		DELETE(cl_Dzz);
		DELETE(cl_Dxy);
		throw;
	}

	// Cleanup
	DELETE(cl_phi);
	DELETE(cl_rhs);
	DELETE(cl_lambda);
	DELETE(cl_Dxx);
	DELETE(cl_Dyy);
	DELETE(cl_Dzz);
	DELETE(cl_Dxy);
}
