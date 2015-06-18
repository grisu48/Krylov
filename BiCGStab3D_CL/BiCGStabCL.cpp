/*
 * BiCGStabSolver.cpp
 *
 *  Created on: Mar 23, 2015
 *      Author: phoenix
 */

#include "BiCGStabCL.hpp"
#include <iostream>
#include <iomanip>



using namespace std;
using namespace flexCL;


/* Status codes */
#define _STATUS_UNINITIALIZED 0
#define _STATUS_READY 128

/* Kernel filename */
#define KERNEL_FILENAME "bicgstab_kernel.cl"

// Verbose output
#ifndef VERBOSE
#define VERBOSE 1
#endif
// Profiling mode on or off
#ifndef PROFILING
#define PROFILING 1
#endif

#ifndef TESTING
#define TESTING 1
#endif


// Delete routine including a NULL check and assignment
#define DELETE(x) { if(x!=NULL) delete x; x = NULL; }
// Tolerance for numerical operations
#define epsilon 1e-3
// Equal comparison for two floating point values
#define REAL_EQUAL(x,y) { fabs(x-y)<epsilon*(y/x) }
// Number of ghost cells
#define RIM 1


using namespace std;
using namespace flexCL;


double dot_product(NumMatrix<double,3> &vecA, NumMatrix<double,3> &vecB) {
	const int dim = 3;

	//! Compute the scalar product of two quantities
	int mx[dim];
	for(int dir=0; dir<dim; ++dir) {
		mx[dir] = vecA.getHigh(dir)-1;
	}

	double result=0.;


	for(int iz = 1; iz < mx[2]; iz += 1) {
		for(int iy = 1; iy < mx[1]; iy += 1) {
			for(int ix = 1; ix < mx[0]; ix += 1) {
				result += vecA(ix,iy,iz)*vecB(ix,iy,iz);
			}
		}
	}

	// Now add boundary values
	// at x-boundaries
	for(int iz = 1; iz < mx[2]; iz += 1) {
		for(int iy = 1; iy < mx[1]; iy += 1) {
			result +=  vecA(0,iy,iz)*vecB(0,iy,iz);
			result +=  vecA(mx[0],iy,iz)*vecB(mx[0],iy,iz);
		}
	}
	// at y-boundaries
	for(int iz = 1; iz < mx[2]; iz += 1) {
		for(int ix = 1; ix < mx[0]; ix += 1) {
			result += vecA(ix,0,iz)*vecB(ix,0,iz);
			result += vecA(ix,mx[1],iz)*vecB(ix,mx[1],iz);
		}
	}
	// at z-boundaries:
	for(int iy = 1; iy < mx[1]; iy += 1) {
		for(int ix = 1; ix < mx[0]; ix += 1) {
			result += vecA(ix,iy,0)*vecB(ix,iy,0);
			result += vecA(ix,iy,mx[2])*vecB(ix,iy,mx[2]);
		}
	}

	// Now add boundary values with weight
	for(int iz = 1; iz < mx[2]; iz += 1) {
		result += vecA(0,0,iz)*vecB(0,0,iz);
		result += vecA(mx[0],0,iz)*vecB(mx[0],0,iz);
		result += vecA(0,mx[1],iz)*vecB(0,mx[1],iz);
		result += vecA(mx[0],mx[1],iz)*vecB(mx[0],mx[1],iz);
	}

	for(int iy = 1; iy < mx[1]; iy += 1) {
		result += vecA(0,iy,0)*vecB(0,iy,0);
		result += vecA(mx[0],iy,0)*vecB(mx[0],iy,0);
		result += vecA(0,iy,mx[2])*vecB(0,iy,mx[2]);
		result += vecA(mx[0],iy,mx[2])*vecB(mx[0],iy,mx[2]);
	}

	for(int ix = 1; ix < mx[0]; ix += 1) {
		result += vecA(ix,0,0)*vecB(ix,0,0);
		result += vecA(ix,mx[1],0)*vecB(ix,mx[1],0);
		result += vecA(ix,0,mx[2])*vecB(ix,0,mx[2]);
		result += vecA(ix,mx[1],mx[2])*vecB(ix,mx[1],mx[2]);
	}

	// Finally add boundary values with weight 1/8
	result += vecA(0,0,0)*vecB(0,0,0);
	result += vecA(mx[0],0,0)*vecB(mx[0],0,0);
	result += vecA(0,mx[1],0)*vecB(0,mx[1],0);
	result += vecA(mx[0],mx[1],0)*vecB(mx[0],mx[1],0);
	result += vecA(0,0,mx[2])*vecB(0,0,mx[2]);
	result += vecA(mx[0],0,mx[2])*vecB(mx[0],0,mx[2]);
	result += vecA(0,mx[1],mx[2])*vecB(0,mx[1],mx[2]);
	result += vecA(mx[0],mx[1],mx[2])*vecB(mx[0],mx[1],mx[2]);
	return result;
}

inline double dot_product(NumMatrix<double,3> &vec) {
	return dot_product(vec,vec);
}

/**
 * Print slice of the matrix as z/2
 */
void print(NumMatrix<double,3> &matrix) {
	ssize_t mx[3];
	ssize_t low[3];
	ssize_t high[3];
	for(int i=0;i<3;i++) {
		low[i] = matrix.getLow(i);
		high[i] = matrix.getHigh(i);
		mx[i] = high[i] - low[i];
	}

	// Print slice at z/2
	const int z = mx[2]/2;
	int row = 0;
	for(ssize_t x = low[0]; x <= high[0]; x++) {
		cout << setw(4) << ++row << "|\t";
		for(ssize_t y = low[1]; y <= high[1]; y++) {
			cout << '\t' << setw(8) << matrix(x,y,z);
			//cout << "matrix[" << x << "," << y << "," << z << "] = " << matrix(x,y,z) << endl;
		}

		cout << endl;
	}
}


/**
 * Print slice of the matrix as z/2
 */
void print(CLMatrix3d *matrix) {
	ssize_t mx[3];
	ssize_t rim = matrix->rim();
	for(int i=0;i<3;i++) mx[i] = matrix->mx(i);
	// const size_t rim = matrix->rim();

	Matrix3d *locMatrix = matrix->transferToHost();

	// Print slice at z/2
	const int z = mx[2]/2;
	int row = 0;
	for(ssize_t x = -rim; x < mx[0]+rim ; x++) {
		cout << setw(4) << ++row << "|\t";
		for(ssize_t y = -rim; y < mx[1]+rim ; y++) {
			cout << '\t' << setw(8) << locMatrix->get(x,y,z);
		}
		cout << endl;
	}

	delete locMatrix;

}

/**
 * Print slice of the matrix as z/2
 */
inline void print(CLMatrix3d &matrix) { print(&matrix); }


bool compareMatrixSize(NumMatrix<double,3> &matA, NumMatrix<double,3> &matB) {
	for(int i=0;i<3;i++) {
		if(matA.getLow(i) != matB.getLow(i)) return false;
		if(matA.getHigh(i) != matB.getHigh(i)) return false;
	}
	return true;
}

/**
 * @brief Compare two matrices. If a cell is varying, the method prints an error message to the given ostream instance
 * @return 0 if the matrices match each other, -1 if the size of the matrices are different or otherwise it returns the number of cells that vary from each other (numerical float comparison)
 */
size_t compareMatrices(NumMatrix<double,3> mat1, CLMatrix3d* mat2, ostream &out = cerr) {
	size_t mx[3];
	size_t rim = mat2->rim();
	size_t result = 0;

	for(int i=0;i<3;i++) {
		mx[i] = (mat1.getHigh(i) - mat1.getLow(i))-2*rim;
		if(mx[i] != mat2->mx(i)) {
			out << "compareMatrices - mx[" << i << "] failed (" << mx[i] << " != " << mat2->mx(i) << ")" << endl;
			return -1;
		}
	}

	Matrix3d *matLocal = mat2->transferToHost();

	for(size_t x = 0; x < mx[0]; x++) {
		for(size_t y = 0; y < mx[1]; y++) {
			for(size_t z = 0; z < mx[2]; z++) {
				const double v1 = mat1(x,y,z);
				const double v2 = matLocal->get(x,y,z);
				const double delta = fabs(v1-v2);
				if(delta > epsilon) {
					out << "compareMatrices[" << x << "," << y << "," << z << "] - " << v1 << " != " << v2 << " (DELTA = " << delta << ")" << endl;
					result++;
				}

			}
		}
	}
	out.flush();

	delete matLocal;
	return result;


}


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

	const size_t mx = (size_t)this->mx[0]+2;
	const size_t my = (size_t)this->mx[1]+2;
	const size_t mz = (size_t)this->mx[2]+2;

	this->_matrix_rhs = new CLMatrix3d(this->_context, mx,my,mz);
	this->_matrix_rhs->initializeContext();
	this->_matrix_rhs->clear();
	this->_matrix_lambda = new CLMatrix3d(this->_context, mx,my,mz);
	this->_matrix_lambda->initializeContext();
	this->_matrix_lambda->clear();

	this->_matrix_residuals = new CLMatrix3d*[this->lValue+1];
	this->_uMat = new CLMatrix3d*[this->lValue+1];
	for(int i=0;i<this->lValue+1;i++) {
		this->_matrix_residuals[i] = new CLMatrix3d(this->_context, mx,my,mz, NULL, RIM);
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

// XXX: Remove parameter size
static CLMatrix3d* transferMatrix(Context *context, NumMatrix<double,3> &matrix, CLMatrix3d *copyContextFrom, size_t* size) {
	const ssize_t rim = RIM;		// Number of ghost cells
	ssize_t _size[3];
	ssize_t _low[3];
	ssize_t _high[3];

	for(int i=0;i<3;i++) {
		_low[i] = matrix.getLow(i);
		_high[i] = matrix.getHigh(i);
		_size[i] = _high[i] - _low[i];
	}

	// +2 because we want at least 1 ghost cell in each dimension
	//Matrix3d temp(_size[0]+2*rim, _size[1]+2*rim, _size[2]+2*rim);
	Matrix3d temp(_size[0], _size[1], _size[2], rim);
	temp.clear();

	// Copy the whole matrix
	for(ssize_t ix=_low[0]; ix<_high[0]; ix++)
		for(ssize_t iy=_low[1]; iy<_high[1]; iy++)
			for(ssize_t iz=_low[2]; iz<_high[2]; iz++) {
				const double value = matrix(ix,iy,iz);
#if BICGSTAB_SOLVER_ADDITIONAL_CHECKS == 1
				if(::isnan(value) || ::isinf(value))
					cerr << "transferMatrix - NAN or INF value at " << ix << "," << iy << "," << iz << endl;
#endif
				temp.set(ix,iy,iz, value);
			}


	CLMatrix3d* result = temp.transferToDevice(context);
#if BICGSTAB_SOLVER_ADDITIONAL_CHECKS == 1

	// Transfer back to host and check the two matrices
	Matrix3d* copy = result->transferToHost();
	size_t deltaCells = copy->compare(temp, true);

	delete copy;
	if(deltaCells > 0) {
		cerr << "transferring matrix to device: " << deltaCells << " cells varying !! " << endl;
		throw NumException("Transfer to device failed (deltaCells > 0)");
	}

	/*

	// Compare just non-null matrices deep
	if(!temp.isNull(true)) {
		size_t cells =compareMatrices(matrix, result);
		if(cells != 0) {
			cerr << "Transfer to host failed. Matrices varying in " << cells << " out of " << result->sizeTotal() << " cells." << endl;
			throw NumException("Transfer to device failed (compareMatrices != 0)");
		}
	}

	*/

#endif
	return result;
}

void BiCGStabSolver::applyBoundary(CLMatrix3d* matrix) {
	size_t size[3] = {matrix->mx(0),matrix->mx(1), matrix->mx(2)};
	const size_t rim = matrix->rim();

	this->_clKernelBoundary->setArgument(0, matrix->clMem());
	this->_clKernelBoundary->setArgument(1, size[0]+rim);
	this->_clKernelBoundary->setArgument(2, size[0]+rim);
	this->_clKernelBoundary->setArgument(3, size[0]+rim);
	cerr << "applyBoundary ... "; cerr.flush();
	this->_clKernelBoundary->enqueueNDRange(size[0]+2*rim, size[1]+2*rim, size[2]+2*rim);
	this->_context->join();
	cerr << "done" << endl; cerr.flush();

#if BICGSTAB_SOLVER_ADDITIONAL_CHECKS == 1
	this->_context->join();
	if(!this->checkMatrix(matrix)) throw "applyBoundary produces illegal values";
#endif
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
#if BICGSTAB_SOLVER_ADDITIONAL_CHECKS == 1
	this->_context->join();
	if(!this->checkMatrix(dst)) throw "generateAx_Full produces illegal values in dst";
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

	// When enqueueing use only mx without ghost cells!
	this->_clKernelGenerateAx_NoSpatial->enqueueNDRange(this->mx[0], this->mx[1], this->mx[2]);

#if PROFILING == 1
	this->_context->join();
	const unsigned long runtime_ms = this->_clKernelGenerateAx_NoSpatial->runtime() * 1e-6;
	cerr << "PROFILING: generateAx -- " << runtime_ms << " ms" << endl;
#endif
#if BICGSTAB_SOLVER_ADDITIONAL_CHECKS == 1
	this->_context->join();
	if(!this->checkMatrix(dst)) throw "generateAx_NoSpatial produces illegal values in dst";
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
#if PROFILING == 1
	unsigned long runtime = 0L;
#endif

	this->generateAx(phi, residual, lambda);
#if PROFILING == 1
	runtime += _clKernelGenerateAx_NoSpatial->runtime();
#endif
#if BICGSTAB_SOLVER_ADDITIONAL_CHECKS == 1
	if(!checkMatrix(phi))       throw "generateAx(generateAx) - checkMatrix(phi) failed";
	if(!checkMatrix(residual))  throw "generateAx(generateAx) - checkMatrix(residual) failed";
	if(!checkMatrix(lambda))    throw "generateAx(generateAx) - checkMatrix(lambda) failed";
#endif

	residual->add(rhs);
#if PROFILING == 1
	runtime += residual->lastKernelRuntime();
#endif
#if BICGSTAB_SOLVER_ADDITIONAL_CHECKS == 1
	if(!checkMatrix(residual))  throw "generateAx(residual->add) - checkMatrix(residual) failed";
#endif
	applyBoundary(residual);
#if PROFILING == 1
	runtime += this->_clKernelBoundary->runtime();
#endif
#if BICGSTAB_SOLVER_ADDITIONAL_CHECKS == 1
	if(!checkMatrix(residual))  throw "generateAx(applyBoundary) - checkMatrix(residual) failed";
#endif

#if PROFILING == 1
	this->_context->join();
	cerr << "PROFILING: calculated Residual (" << (runtime*1e-6) << " ms)" << endl;
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

#if TESTING == 1


	// Check matrix sizes
	if(!compareMatrixSize(phi, phi)) throw "Matrix size mismatch: phi, phi";
	if(!compareMatrixSize(phi, rhs)) throw "Matrix size mismatch: phi, rhs";
	if(!compareMatrixSize(phi, lambda)) throw "Matrix size mismatch: phi, lambda";
	if(!compareMatrixSize(phi, Dxx)) throw "Matrix size mismatch: phi, Dxx";
	if(!compareMatrixSize(phi, Dyy)) throw "Matrix size mismatch: phi, Dyy";
	if(!compareMatrixSize(phi, Dzz)) throw "Matrix size mismatch: phi, Dzz";
	if(!compareMatrixSize(phi, Dxy)) throw "Matrix size mismatch: phi, Dxy";



	const double test_scalar_phi = dot_product(phi);
	const double test_scalar_rhs = dot_product(rhs);
	const double test_scalar_lambda = dot_product(lambda);

#endif

	// Transfer matrices to OpenCL context
	CLMatrix3d *cl_phi = transferMatrix(this->_context, phi, this->_matrix_rhs, this->mx);
	CLMatrix3d *cl_rhs = transferMatrix(this->_context, rhs, this->_matrix_rhs, this->mx);
	CLMatrix3d *cl_lambda = transferMatrix(this->_context, lambda, this->_matrix_rhs, this->mx);

	cout << "Lambda:" << endl;
	print(lambda);
	cout << endl << "CL_Lambda:" << endl;
	print(cl_lambda);
	cout << endl;

#if TESTING == 1

	double test_scalar_cl_phi = cl_phi->dotProduct();
	double test_scalar_cl_rhs = cl_rhs->dotProduct();
	double test_scalar_cl_lambda = cl_lambda->dotProduct();

	double test_delta = fabs(test_scalar_phi - test_scalar_cl_phi) +
			fabs(test_scalar_rhs - test_scalar_cl_rhs) + fabs(test_scalar_lambda - test_scalar_cl_lambda);

	if(test_delta > epsilon) {

		cout << "<phi,phi>       = " << test_scalar_phi << "\t" << test_scalar_cl_phi << "\t DELTA = " << fabs(test_scalar_phi - test_scalar_cl_phi) << endl;
		cout << "<rhs,rhs>       = " << test_scalar_rhs << "\t" << test_scalar_cl_rhs << "\t DELTA = " << fabs(test_scalar_rhs - test_scalar_cl_rhs) << endl;
		cout << "<lambda,lambda> = " << test_scalar_lambda << "\t" << test_scalar_cl_lambda << "\t DELTA = " << fabs(test_scalar_lambda - test_scalar_cl_lambda) << endl;
		cout.flush();

		//print(lambda);
		//print(cl_lambda);

		cerr << "((!)) TEST DELTA (" << test_delta << ") OF SCALARS > Epsilon (" << epsilon << ")" << endl;
		cerr.flush();
		throw NumException("Scalar comparison failed");
	}


#endif

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
	CLMatrix3d *cl_resTilde = new CLMatrix3d(this->_context, cl_phi->mx(0), cl_phi->mx(1), cl_phi->mx(2), NULL, RIM);
	cl_resTilde->initializeContext();
	cl_resTilde->clear();

	this->_context->join();
	cout << "Matrices transferred to host device " << endl;

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
		double rho1 = 1.0;
		double alpha = 0.0;
		double omega = 1.0;
		double norm = 0.0;

		double tau[lValue+1][lValue+1];
		double sigma[lValue+1], gammap[lValue+1], gammapp[lValue+1];
		double _gamma[lValue+1];

		// Calculate r_0

		if(use_spatialDiffusion) {
			this->calculateResidual(this->_matrix_residuals[0], cl_phi, cl_rhs, cl_lambda, cl_Dxx, cl_Dyy, cl_Dzz, cl_Dxy);
		} else {
			this->calculateResidual(this->_matrix_residuals[0], cl_phi, cl_rhs, cl_lambda);
		}
#if BICGSTAB_SOLVER_ADDITIONAL_CHECKS == 1
		if(!this->checkMatrix(this->_matrix_residuals[0])) throw "matrix_residual check 0 failed";
#endif

		cl_resTilde->copyFrom(this->_matrix_residuals[0]);

		cout << "<resTilde,resTilde> = " << cl_resTilde->dotProduct() << endl;
		exit(EXIT_FAILURE);

		do {
			iterations++;
			cout << "Starting iteration " << iterations << " ... " << endl;

			rho0 *= -omega;

			// ==== BI-CG PART ============================================= //
			// cout << "BI-CG part of the solver ... " << endl;

			for(int jj=0; jj<lValue; ++jj) {

#if BICGSTAB_SOLVER_ADDITIONAL_CHECKS == 1
		if(!this->checkMatrix(this->_matrix_residuals[jj])) throw "matrix_residual check 1 failed";
#endif
				rho1 = this->_matrix_residuals[jj]->dotProduct(cl_resTilde);
				const double beta = alpha*rho1/rho0;
				rho0 = rho1;

				for(int ii=0; ii<=jj; ++ii) {
					// XXX: Performance: Replace with add mul
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
				if(!this->checkMatrix(cl_resTilde)) throw "cl_resTilde check 3.1 failed";
				double dotProduct = _uMat[jj+1]->dotProduct(cl_resTilde);
				if(::isnan(dotProduct) || ::isinf(dotProduct)) throw "dotProduct check 3.1 failed";
#endif

				alpha = rho0/ (_uMat[jj+1]->dotProduct(cl_resTilde));

				for(int ii=0; ii<=jj; ii++) {
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
			// cout << "MR part of the solver ... " << endl;
			for(int jj=1; jj<=lValue; ++jj) {
				for(int ii=1; ii<jj; ++ii) {
					tau[ii][jj] = _matrix_residuals[jj]->dotProduct(_matrix_residuals[ii]) / sigma[ii];
					_matrix_residuals[jj]->subMultiplied(_matrix_residuals[ii], tau[ii][jj]);
				}


				sigma[jj] = _matrix_residuals[jj]->dotProduct();
				gammap[jj] = _matrix_residuals[0]->dotProduct(_matrix_residuals[jj])/sigma[jj];
			}

			_gamma[lValue] = gammap[lValue];
			omega = _gamma[lValue];
			cout << omega << endl;

			for(int jj=lValue-1; jj>=1; --jj) {
				_gamma[jj] = gammap[jj];
				for(int ii=jj+1; ii<=lValue; ++ii) {
					_gamma[jj] -= tau[jj][ii]*_gamma[ii];
				}
			}

			for(int jj=1; jj<lValue; ++jj) {
				gammapp[jj] = _gamma[jj+1];
				for(int ii=jj+1; ii<lValue; ++ii) {
					gammapp[jj] += tau[jj][ii]*_gamma[ii+1];
				}
			}


			// Check break conditions:
			cl_phi->addMultiplied(_matrix_residuals[0], _gamma[1]);
			_matrix_residuals[0]->subMultiplied(_matrix_residuals[lValue], gammap[lValue]);
			_uMat[0]->subMultiplied(_uMat[lValue], _gamma[lValue]);

			for(int jj=1; jj<lValue; ++jj) {
				_uMat[0]->subMultiplied(_uMat[jj],_gamma[jj]);
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
