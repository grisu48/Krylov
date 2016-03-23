/*
 * BiCGStabSolver.cpp
 *
 *  Created on: Mar 23, 2015
 *      Author: phoenix
 */

#include "BiCGStabCL.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include "time_ms.h"


using namespace std;
using namespace flexCL;


/* Status codes */
#define _STATUS_UNINITIALIZED 0
#define _STATUS_READY 128

/* Kernel filename */
#define KERNEL_FILENAME "bicgstab_kernel.cl"



/* ==== Switches =============================================================================== */

// Additional checks in the BiCGStab solver used for debugging
#ifndef BICGSTAB_SOLVER_ADDITIONAL_CHECKS
#define BICGSTAB_SOLVER_ADDITIONAL_CHECKS 0
#endif

// Verbose output
#ifndef VERBOSE
#define VERBOSE 0
#endif
// Profiling mode on or off
#ifndef PROFILING
#define PROFILING 0
#endif

#ifndef TESTING
#define TESTING 0
#endif

// Tolerance for numerical operations
#ifndef epsilon
#define epsilon 1e-3
#endif
// Number of ghost cells
#define RIM 1


using namespace std;
using namespace flexCL;

// Maximum number of iterations before the solver quits or negative number, if this check should be disabled
#ifndef MAX_ITERATIONS
#define MAX_ITERATIONS 1000L
#endif

// Equal comparison for two floating point values
#define REAL_EQUAL(x,y) { fabs(x-y)<epsilon*(y/x) }
// Delete routine including a NULL check and assignment
#define DELETE(x) { if(x!=NULL) delete x; x = NULL; }

#define MARK_USED(x) (void)(x);

/* ==== DEBUGGING ACTIONS ====================================================================== */


// Some random hash function for the matrix
#if VERBOSE == 1 || TESTING == 1
double hash(NumMatrix<double, 3> &matrix) {
	//const long brick = 107534845447;		// Random prime number (HUGE)

	ssize_t _low[3];
	ssize_t _high[3];
	for(int i=0;i<3;i++) {
		_low[i]   = matrix.getLow(i);
		_high[i]  = matrix.getHigh(i);
	}

	double value = 1.0;

	for(ssize_t x = _low[0]; x < _high[0]; x++) {
		for(ssize_t y = _low[1]; y < _high[1]; y++) {
			for(ssize_t z = _low[2]; z < _high[2]; z++) {
				value += (x*y*z) * matrix(x,y,z);
			}
		}
	}

	return value;
}

double hash_cl(flexCL::CLMatrix3d *mat) {
	Matrix3d *matrix = mat->transferToHost();

	//const long brick = 107534845447;		// Random prime number (HUGE)

	ssize_t rim = matrix->rim();
	ssize_t mx[3] = { (ssize_t)matrix->mx(0), (ssize_t)matrix->mx(1), (ssize_t)matrix->mx(2) };


	double value = 1.0;

	for(ssize_t x = -rim; x < mx[0]+rim; x++) {
		for(ssize_t y = -rim; y < mx[1]+rim; y++) {
			for(ssize_t z = -rim; z < mx[2]+rim; z++) {
				value += (x*y*z) * matrix->get(x,y,z);
			}
		}
	}
	delete matrix;
	return value;
}

double dot_product(NumMatrix<double,3> &vecA, NumMatrix<double,3> &vecB) {
	// Comput scalar product of matrix without ghost cells
	const size_t rim = RIM;
	size_t size[3];
	for(int i=0;i<3;i++) {
		if (vecA.getLow(i) != -rim) throw "dotProduct - RIM field mismatch of NumMatrix A";
		if (vecB.getLow(i) != -rim) throw "dotProduct - RIM field mismatch of NumMatrix B";

		size[i] = vecA.getHigh(i) - rim;
		if(vecB.getHigh(i)-rim != size[i]) throw "dotProduct - vecB.size mismatch with vecA";
	}

	double result = 0.0;

	for(size_t x = 0; x < size[0]; x++) {
		for(size_t y = 0; y < size[1]; y++) {
			for(size_t z = 0; z < size[2]; z++) {
				result += vecA(x,y,z) * vecB(x,y,z);
			}
		}
	}
	return result;
}

inline double dot_product(NumMatrix<double,3> &vec) {
	return dot_product(vec,vec);
}

double get_l2Norm(NumMatrix<double,3> &vec) {
	const double dotProd = dot_product(vec);
	return sqrt(dotProd);
}

#endif

void print(NumMatrix<double, 3> &matrix) {
	// Print now output matrix at middle position
	size_t z = matrix.getHigh(2)/2;
	size_t y = matrix.getHigh(1)/2;
	size_t mx = matrix.getHigh(0);

	for(size_t x = 0; x < mx; x++) {
		cout << matrix(x,y,z) << '\t';
	}
	cout << endl;
}

void print(flexCL::CLMatrix3d *matrix) {
	// Print now output matrix at middle position
	size_t z = matrix->mx(2)/2-1;
	size_t y = matrix->mx(1)/2-1;
	size_t mx = matrix->mx(0)-1;

	Matrix3d *matLocal = matrix->transferToHost();
	for(size_t x = 0; x < mx; x++) {
	//	for(size_t y = 0; y < this->mx[1]; y++)
			cout << matLocal->get(x,y,z) << '\t';
	}
	cout << endl;
	delete matLocal;
}

void printFull(flexCL::Matrix3d *matrix, ostream &out = cout) {
	ssize_t mz = matrix->mx(2);
	ssize_t my = matrix->mx(1);
	ssize_t mx = matrix->mx(0);
	ssize_t rim = matrix->rim();

	for(ssize_t x = -rim; x < mx+rim; x++) {
		int row = -rim;
		for(ssize_t y = -rim; y < my+rim; y++) {
			out << ++row << "\t|";
			for(ssize_t z = -rim; z < mz+rim; z++) {
				out << '\t' << matrix->get(x,y,z);
			}
			out << endl;
		}
		out << endl;
	}
}

void printFull(flexCL::CLMatrix3d *matrix, ostream &out = cout) {
	// Print now output matrix at middle position

	Matrix3d *matLocal = matrix->transferToHost();
	printFull(matLocal, out);
	delete matLocal;
}

void printFull(NumMatrix<double, 3> &mat, ostream &out = cout) {
	// Print now output matrix at middle position
	ssize_t _low[3];
	ssize_t _high[3];
	for(int i=0;i<3;i++) {
		_low[i] = mat.getLow(i);
		_high[i] = mat.getHigh(i);
	}

	for(ssize_t x = _low[0]; x < _high[0]; x++) {
		int row = _low[0];
		for(ssize_t y = _low[1]; y < _high[1]; y++) {
			out << ++row << "\t|";
			for(ssize_t z = _low[2]; z < _high[2]; z++) {
				out << '\t' << mat(x,y,z);
			}
			out << endl;
		}
		out << endl;
	}
}

inline std::string f_filename(const char* filename) {
	std::string result = "/home/phoenix/temp/" + std::string(filename);
	return result;
}

void printFull(NumMatrix<double, 3> &matrix, const char* filename, bool relativePath = true) {
	ofstream out;
	string fname;
	if (relativePath) {
		fname = f_filename(filename);

	} else
		fname = string(filename);
	out.open(fname.c_str());
#if TESTING == 1
	cout << "\tWriting NumMatrix to " << fname << " ... "; cout.flush();
#endif
	printFull(matrix, out);
	cout << "done" << endl;
	out.close();
}

void printFull(flexCL::CLMatrix3d *matrix, const char* filename, bool relativePath = true) {
	ofstream out;
	string fname;
	if (relativePath) {
		fname = f_filename(filename);

	} else
		fname = string(filename);
	out.open(fname.c_str());
#if TESTING == 1
	cout << "\tWriting NumMatrix to " << fname << " ... "; cout.flush();
#endif
	printFull(matrix, out);
	cout << "done" << endl;
	out.close();
}
void printFull(flexCL::CLMatrix3d *matrix, string filename) {
	printFull(matrix, filename.c_str());
}

void printFull(flexCL::Matrix3d *matrix, const char* filename) {
	ofstream out;
	std::string fname = f_filename(filename);
	out.open(fname.c_str());
	printFull(matrix, out);
	out.close();
}

void printFull(flexCL::CLMatrix3d &matrix, const char* filename) {
	printFull(&matrix, filename);
}

void printFull(flexCL::Matrix3d &matrix, const char* filename) {
	printFull(&matrix, filename);
}


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



/* ==== INTERNAL DEFINES ======================================================================= */

// This is not intended to be edited by the user!
#if VERBOSE == 1 && TESTING == 1
#define COUT cout
#else
// Should normally never trigger
#define COUT if(false) cout
#endif


/* ==== ACTUAL BICGSTAB SOLVER SECTION ========================================================= */


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
	this->_steptime_min = 0;
	this->_steptime_max = 0;
	this->_iterations = 0;
	this->verbose = false;

}

BiCGStabSolver::~BiCGStabSolver() {
	this->cleanupContext();
}
bool BiCGStabSolver::isInitialized(void) { return this->status == _STATUS_READY; }

void BiCGStabSolver::setVerbose(bool verbose) { this->verbose = verbose; }

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

	this->_matrix_rhs = new CLMatrix3d(this->_context, mx,my,mz, NULL, RIM);
	this->_matrix_rhs->initializeContext();
	this->_matrix_rhs->clear();
	this->_matrix_lambda = new CLMatrix3d(this->_context, mx,my,mz, NULL, RIM);
	this->_matrix_lambda->initializeContext();
	this->_matrix_lambda->clear();

	this->_matrix_residuals = new CLMatrix3d*[this->lValue+1];
	this->_uMat = new CLMatrix3d*[this->lValue+1];
	for(int i=0;i<this->lValue+1;i++) {
		this->_matrix_residuals[i] = new CLMatrix3d(this->_context, mx,my,mz, NULL, RIM);
		this->_uMat[i] = new CLMatrix3d(this->_context, mx,my,mz, NULL, RIM);

		this->_matrix_residuals[i]->initializeContext();
		this->_uMat[i]->initializeContext();

		this->_matrix_residuals[i]->clear();
		this->_uMat[i]->clear();

#if PROFILING == 1
		this->_matrix_residuals[i]->setProfiling(true);
		this->_uMat[i]->setProfiling(true);
#endif
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
                     double D_xx, double D_yy, double D_zz, int debug,
                     double delt, bool evolve_time) {
	//! Main solver routine
	/*! Solves equations of the form:
	  \nabla\cdot\left(D\nabla \phi\right) - \lambda \phi = rhs
	 */
	this->debug = debug;
	this->use_spatialDiffusion = false;
	this->diffDiag[0] = D_xx;
	this->diffDiag[1] = D_yy;
	this->diffDiag[2] = D_zz;

	// Run solver routine
	solve_int(bounds, phi, rhs, lambda, rhs, rhs, rhs, rhs, debug);
}


void BiCGStabSolver::solve(BoundaryHandler3D &bounds, NumMatrix<double,3> &phi,
                     NumMatrix<double,3> &rhs, NumMatrix<double,3> &lambda,
                     NumMatrix<double,3> &Dxx, NumMatrix<double,3> &Dyy,
                     NumMatrix<double,3> &Dzz, NumMatrix<double,3> &Dxy,
                     int debug, bool use_offDiagDiffusion,
                     double delt, bool evolve_time) {
	//! Main solver routine
	/*! Solves equations of the form:
	  \nabla\cdot\left(D\nabla \phi\right) - \lambda \phi = rhs
	 */
	this->use_spatialDiffusion = true;
	this->debug = debug;
	this->use_offDiagDiffusion = use_offDiagDiffusion;
	if(!use_offDiagDiffusion)
		Dxy.clear();


	// Run solver routine
	solve_int(bounds, phi, rhs, lambda, Dxx, Dyy, Dzz, Dxy, debug);
}

void BiCGStabSolver::solve(BoundaryHandler3D &bounds, NumMatrix<double,3> &phi,
                     NumMatrix<double,3> &rhs, NumMatrix<double,3> &lambda,
                     NumMatrix<double,3> &Dxx, NumMatrix<double,3> &Dyy,
                     NumMatrix<double,3> &Dzz,
                     int debug, double delt, bool evolve_time) {
	//! Main solver routine
	/*! Solves equations of the form:
	  \nabla\cdot\left(D\nabla \phi\right) - \lambda \phi = rhs
	 */
	solve(bounds, phi, rhs, lambda, Dxx, Dyy, Dzz, Dzz, debug);
}

#if 0
void BiCGStabSolver::solve(BoundaryHandler3D &bounds, NumMatrix<double,3> &phi,
		NumMatrix<double,3> &rhs, NumMatrix<double,3> &lambda,
		double D_xx, double D_yy, double D_zz, int debug, double delta, bool evolve_time) {

	MARK_USED(evolve_time);
	MARK_USED(delta);

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
		NumMatrix<double,3> &Dzz,
		int debug, double delta, bool evolve_time) {

	MARK_USED(evolve_time);
	MARK_USED(delta);
	this->debug = debug;
	//! Main solver routine
	/*! Solves equations of the form:
		  \nabla\cdot\left(D\nabla \phi\right) - \lambda \phi = rhs
	 */
	solve(bounds, phi, rhs, lambda, Dxx, Dyy, Dzz, Dzz, debug);

}

void BiCGStabSolver::solve(BoundaryHandler3D &bounds, NumMatrix<double,3> &phi,
		NumMatrix<double,3> &rhs, NumMatrix<double,3> &lambda,
		NumMatrix<double,3> &Dxx, NumMatrix<double,3> &Dyy,
		NumMatrix<double,3> &Dzz, NumMatrix<double,3> &Dxy,
		int debug, bool use_offDiagDiffusion,
		double delta, bool evolve_time) {

	MARK_USED(evolve_time);
	MARK_USED(delta);
	this->debug = debug;

	this->use_spatialDiffusion = true;
	this->use_offDiagDiffusion = use_offDiagDiffusion;
	this->debug = debug;
	if(!use_offDiagDiffusion) {
		Dxy.clear();
	}

	solve_int(bounds, phi, rhs, lambda, Dxx, Dyy, Dzz, Dxy);

}
#endif


/** Transfers the given NumMatrix to the given OpenCL context */

// XXX: Remove parameter size
static CLMatrix3d* transferMatrix(Context *context, NumMatrix<double,3> &matrix, CLMatrix3d *copyContextFrom, size_t* size) {
	// =================================================
	// * Transfer tested. 2015-06-19
	// Considered: Done
	// =================================================


	const ssize_t rim = RIM;		// Number of ghost cells
	ssize_t _size[3];
	ssize_t _low[3];
	ssize_t _high[3];

	for(int i=0;i<3;i++) {
		_low[i] = matrix.getLow(i);
		_high[i] = matrix.getHigh(i);
		_size[i] = _high[i] - _low[i]-2*rim;

#if BICGSTAB_SOLVER_ADDITIONAL_CHECKS == 1
		if( -_low[i] != rim) throw OpenCLException("transferMatrix - NumMatrix RIM field mismatch");
#endif
	}

	// +2 because we want at least 1 ghost cell in each dimension
	//Matrix3d temp(_size[0]+2*rim, _size[1]+2*rim, _size[2]+2*rim);
	Matrix3d temp(_size[0], _size[1], _size[2], rim);
	temp.clear();

	// Copy the whole matrix
	for(ssize_t ix=_low[0]; ix<_high[0]; ix++) {
		for(ssize_t iy=_low[1]; iy<_high[1]; iy++) {
			for(ssize_t iz=_low[2]; iz<_high[2]; iz++) {
				const double value = matrix(ix,iy,iz);
#if BICGSTAB_SOLVER_ADDITIONAL_CHECKS == 1
				if(::isnan(value) || ::isinf(value))
					cerr << "transferMatrix - NAN or INF value at " << ix << "," << iy << "," << iz << endl;
#endif
				temp.set(ix,iy,iz, value);
			}
		}
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

/**
 * Transfer destination matrix to NumMatrix instance
 */
static void transferMatrix(flexCL::CLMatrix3d *matrix, NumMatrix<double, 3> &dst) {
	Matrix3d *temp = matrix->transferToHost();

	const ssize_t rim = RIM;		// Number of ghost cells
	ssize_t _size[3];
	ssize_t _low[3];
	ssize_t _high[3];

	for(int i=0;i<3;i++) {
		_low[i] = dst.getLow(i);
		_high[i] = dst.getHigh(i);
		_size[i] = _high[i] - _low[i]-2*rim;

#if BICGSTAB_SOLVER_ADDITIONAL_CHECKS == 1
		if( -_low[i] != rim) throw OpenCLException("transferMatrix - NumMatrix RIM field mismatch");
#endif
	}

	// Copy the whole matrix
	for(ssize_t ix=_low[0]; ix<_high[0]; ix++) {
		for(ssize_t iy=_low[1]; iy<_high[1]; iy++) {
			for(ssize_t iz=_low[2]; iz<_high[2]; iz++) {
				const double value = temp->get(ix,iy,iz);
#if BICGSTAB_SOLVER_ADDITIONAL_CHECKS == 1
				if(::isnan(value) || ::isinf(value))
					cerr << "transferMatrix - NAN or INF value at " << ix << "," << iy << "," << iz << endl;
#endif
				dst(ix,iy,iz) =  value;
			}
		}
	}

	delete temp;
}

void BiCGStabSolver::applyBoundary(CLMatrix3d* matrix) {
	//matrix->clearRim();
	//return;
	size_t size[3] = {matrix->mx(0), matrix->mx(1), matrix->mx(2)};
	const size_t rim = matrix->rim();

	this->_clKernelBoundary->setArgument(0, matrix->clMem());
	this->_clKernelBoundary->setArgument(1, size[0]);
	this->_clKernelBoundary->setArgument(2, size[1]);
	this->_clKernelBoundary->setArgument(3, size[2]);
	this->_clKernelBoundary->setArgument(4, rim);
#if PROFILING == 1
	cerr << "PROFILING: applyBoundary ... "; cerr.flush();
#endif

	size_t tot_size[3] = {size[0]+2*rim, size[1]+2*rim, size[2]+2*rim };

	this->_clKernelBoundary->enqueueNDRange(tot_size[0], tot_size[1], tot_size[2]);
#if PROFILING == 1
	this->_context->join();
	const unsigned long runtime_ms = this->_clKernelBoundary->runtime() * 1e-6;
	cerr << "done (" << runtime_ms << " ms)" << endl; cerr.flush();
#endif

#if BICGSTAB_SOLVER_ADDITIONAL_CHECKS == 1
	this->_context->join();
	if(!this->checkMatrix(matrix)) throw "applyBoundary produces illegal values";
#endif
}

void BiCGStabSolver::generateAx(flexCL::CLMatrix3d* phi, flexCL::CLMatrix3d* dst, flexCL::CLMatrix3d* lambda, flexCL::CLMatrix3d* Dxx, flexCL::CLMatrix3d* Dyy, flexCL::CLMatrix3d* Dzz, flexCL::CLMatrix3d* Dxy) {
	applyBoundary(phi);		// XXX: Could we switch that off?

#if BICGSTAB_SOLVER_ADDITIONAL_CHECKS == 1
	if(!this->checkMatrix(phi)) throw "generateAx_Full Preliminary check failed for phi";
	if(!this->checkMatrix(lambda)) throw "generateAx_Full Preliminary check failed for lambda";
	if(!this->checkMatrix(Dxx)) throw "generateAx_Full Preliminary check failed for Dxx";
	if(!this->checkMatrix(Dyy)) throw "generateAx_Full Preliminary check failed for Dyy";
	if(!this->checkMatrix(Dzz)) throw "generateAx_Full Preliminary check failed for Dzz";
	if(!this->checkMatrix(Dxy)) throw "generateAx_Full Preliminary check failed for Dxy";
#endif

	this->_clKernelGenerateAx_Full->setArgument(0, phi->clMem());
	this->_clKernelGenerateAx_Full->setArgument(1, lambda->clMem());
	this->_clKernelGenerateAx_Full->setArgument(2, Dxx->clMem());
	this->_clKernelGenerateAx_Full->setArgument(3, Dyy->clMem());
	this->_clKernelGenerateAx_Full->setArgument(4, Dzz->clMem());
	this->_clKernelGenerateAx_Full->setArgument(5, Dxy->clMem());
	this->_clKernelGenerateAx_Full->setArgument(6, dst->clMem());
	// bicstab_kernel calculates with full matrix size
	this->_clKernelGenerateAx_Full->setArgument(7, this->mx[0]+2*RIM);
	this->_clKernelGenerateAx_Full->setArgument(8, this->mx[1]+2*RIM);
	this->_clKernelGenerateAx_Full->setArgument(9, this->mx[2]+2*RIM);
	this->_clKernelGenerateAx_Full->setArgument(10, (size_t)RIM);
	this->_clKernelGenerateAx_Full->setArgument(11, this->deltaX[0]);
	this->_clKernelGenerateAx_Full->setArgument(12, this->deltaX[1]);
	this->_clKernelGenerateAx_Full->setArgument(13, this->deltaX[2]);


	// (!!) When enqueueing use only mx without ghost cells!
	this->_clKernelGenerateAx_Full->enqueueNDRange(this->mx[0], this->mx[1], this->mx[2]);

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
	applyBoundary(phi);

	// Initialize kernel with kernel arguments
	this->_clKernelGenerateAx_NoSpatial->setArgument(0, phi->clMem());
	this->_clKernelGenerateAx_NoSpatial->setArgument(1, lambda->clMem());
	this->_clKernelGenerateAx_NoSpatial->setArgument(2, dst->clMem());
	// bicstab_kernel calculates with full matrix size
	this->_clKernelGenerateAx_NoSpatial->setArgument(3, this->mx[0]+2*RIM);
	this->_clKernelGenerateAx_NoSpatial->setArgument(4, this->mx[1]+2*RIM);
	this->_clKernelGenerateAx_NoSpatial->setArgument(5, this->mx[2]+2*RIM);
	this->_clKernelGenerateAx_NoSpatial->setArgument(6, (size_t)RIM);
	this->_clKernelGenerateAx_NoSpatial->setArgument(7, this->deltaX[0]);
	this->_clKernelGenerateAx_NoSpatial->setArgument(8, this->deltaX[1]);
	this->_clKernelGenerateAx_NoSpatial->setArgument(9, this->deltaX[2]);
	this->_clKernelGenerateAx_NoSpatial->setArgument(10, this->diffDiag[0]);
	this->_clKernelGenerateAx_NoSpatial->setArgument(11, this->diffDiag[1]);
	this->_clKernelGenerateAx_NoSpatial->setArgument(12, this->diffDiag[2]);

	// (!!) When enqueueing use only mx without ghost cells!
	this->_clKernelGenerateAx_NoSpatial->enqueueNDRange(this->mx[0], this->mx[1], this->mx[2]);
	// WRONG: this->_clKernelGenerateAx_NoSpatial->enqueueNDRange(this->mx[0]+2*RIM, this->mx[1]+2*RIM, this->mx[2]+2*RIM);

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

long BiCGStabSolver::iterations(void) { return this->_iterations; }
long BiCGStabSolver::steptimeMin(void) { return this->_steptime_min; }
long BiCGStabSolver::steptimeMax(void) { return this->_steptime_max; }

std::vector<long> BiCGStabSolver::stepRuntimes(void) {
	const int size = this->stepTimes.size();
	std::vector<long> vector;
	for(int i=0;i<size;i++) {
		const long value = this->stepTimes[i];
		vector.push_back(value);
	}
	return vector;
}

void BiCGStabSolver::calculateResidual(flexCL::CLMatrix3d* residual, flexCL::CLMatrix3d* phi, flexCL::CLMatrix3d* rhs, flexCL::CLMatrix3d* lambda, flexCL::CLMatrix3d* Dxx, flexCL::CLMatrix3d* Dyy, flexCL::CLMatrix3d* Dzz, flexCL::CLMatrix3d* Dxy) {
	this->generateAx(phi, residual, lambda, Dxx, Dyy, Dzz, Dxy);
	residual->add(rhs);

#if PROFILING == 1
	this->_context->join();
	cerr << "PROFILING: calculated Residual" << endl;
#endif

	applyBoundary(residual);
}

void BiCGStabSolver::calculateResidual(flexCL::CLMatrix3d* residual, flexCL::CLMatrix3d* phi, flexCL::CLMatrix3d* rhs, flexCL::CLMatrix3d* lambda) {
#if PROFILING == 1
	unsigned long runtime = 0L;
#endif

	// Use residual as intermediate buffer
	this->generateAx(phi, residual, lambda);
#if PROFILING == 1
	this->_context->join();
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
#if PROFILING == 1
	this->_context->join();
	runtime += this->_clKernelBoundary->runtime();
#endif


#if PROFILING == 1
	this->_context->join();
	cerr << "PROFILING: calculated Residual (" << (runtime*1e-6) << " ms)" << endl;
#endif

	applyBoundary(residual);
#if BICGSTAB_SOLVER_ADDITIONAL_CHECKS == 1
	if(!checkMatrix(residual))  throw "generateAx(applyBoundary) - checkMatrix(residual) failed";
#endif
}

// Solver internal
void BiCGStabSolver::solve_int(BoundaryHandler3D &bounds,
		NumMatrix<double,3> &phi,
		NumMatrix<double,3> &rhs,
		NumMatrix<double,3> &lambda,
		NumMatrix<double,3> &Dxx,
		NumMatrix<double,3> &Dyy,
		NumMatrix<double,3> &Dzz,
		NumMatrix<double,3> &Dxy,
		int debug) {
	MARK_USED(bounds);

#if VERBOSE == 1
	cout << "BiCGStabSolver::solve_int(...)" << endl;
#endif
	if(!isInitialized()) this->setupContext();
	this->_steptime_min = 0L;
	this->_steptime_max = 0L;
	this->debug = debug;

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
#if PROFILING == 1
		cl_Dxx->setProfiling(true);
		cl_Dyy->setProfiling(true);
		cl_Dzz->setProfiling(true);
		cl_Dxy->setProfiling(true);
#endif
	}
	CLMatrix3d *cl_resTilde = new CLMatrix3d(this->_context, cl_phi->mx(0), cl_phi->mx(1), cl_phi->mx(2), NULL, RIM);
	cl_resTilde->initializeContext();
	cl_resTilde->clear();
#if PROFILING == 1
	cl_resTilde->setProfiling(true);
#endif

	this->_context->join();
//	cout << "Matrices transferred to host device " << endl;

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


#if VERBOSE == 1 || TESTING == 1
	cout << "Input variables:" << endl;
	cout << "\thash(phi)     = " << hash_cl(cl_phi) << endl;
	cout << "\thash(rhs)     = " << hash_cl(cl_rhs) << endl;
	cout << "\thash(lambda)  = " << hash_cl(cl_lambda) << endl;
#endif

#if 0
	// Write initial matrices to file
	cout << "Write inital matrices to file ... " << endl;
	printFull(cl_phi, "CL_Initial_phi");
	printFull(cl_rhs, "CL_Initial_rhs");
	printFull(cl_lambda, "CL_Initial_lambda");
	printFull(cl_Dxx, "CL_Initial_Dxx");
	printFull(cl_Dyy, "CL_Initial_Dyy");
	printFull(cl_Dzz, "CL_Initial_Dzz");
	printFull(cl_Dxy, "CL_Initial_Dxy");
#endif

	try {
		unsigned long iterations = 0;
		double normRhs = cl_rhs->l2Norm();
		if(normRhs < 1e-9 || ::isnan(normRhs) || ::isinf(normRhs)) normRhs = 1.0;

#if VERBOSE == 1
		cout << "  normRHS  = " << normRhs << endl;
		cout << "  Expected = " << get_l2Norm(rhs) << endl;
#endif

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
		// NOTE: Tested: Residual matrices are the same.
		//printFull(_matrix_residuals[0], "CL_PRIM_RESIDUAL");
		cl_resTilde->copyFrom(this->_matrix_residuals[0]);

		COUT << "<resTilde,resTilde> = " << cl_resTilde->dotProduct() << endl;
		this->stepTimes.clear();

		long runtime = -time_ms();
		bool initialStep = true;
		do {
			// Check for maximum iterations reached
			if(this->_maxIterations > 0 && (long)iterations > this->_maxIterations)
				throw NumException("Maximum iterations reached");

			iterations++;
			if(iterations > 1) {
				runtime += time_ms();
				this->stepTimes.push_back(runtime);
				if(initialStep) {
					initialStep = false;
					this->_steptime_min = runtime;
					this->_steptime_max = runtime;
				} else {
					if(runtime > this->_steptime_max) this->_steptime_max = runtime;
					if(runtime < this->_steptime_min) this->_steptime_min = runtime;
				}
			}
			if(this->verbose) {
				if(iterations > 1) {
					cout << "Starting iteration " << iterations << " ... (" << runtime << " ms)" << endl;
				} else
					cout << "Starting iteration " << iterations << " ... " << endl;
			}
#if TESTING == 1
			cout << "hash(phi) = " << hash_cl(cl_phi) << endl;
#endif
			runtime = -time_ms();
			COUT << "omega = " << omega << endl;

			rho0 *= -omega;
			COUT << "rho0 = " << rho0 << endl;

			// ==== BI-CG PART ============================================= //
			COUT << "BI-CG part of the solver ... " << endl;


			for(int jj=0; jj<lValue; ++jj) {
				COUT << "jj iteration " << jj << endl;

#if BICGSTAB_SOLVER_ADDITIONAL_CHECKS == 1
				if(!this->checkMatrix(this->_matrix_residuals[jj])) throw "matrix_residual check 1 failed";
#endif

				//cout << "hash(residual[" << jj << "]) = " << hash_cl(this->_matrix_residuals[jj]) << endl;


				rho1 = this->_matrix_residuals[jj]->dotProduct(cl_resTilde);
#if PROFILING == 1
				cerr << "PROFILING: dotProduct -- " << this->_matrix_residuals[jj]->lastKernelRuntime() << " ns" << endl;
#endif
				COUT << "\trho1 = " << rho1 << endl;
				const double beta = alpha*rho1/rho0;
				rho0 = rho1;
				COUT << "\tbeta = " << beta << endl;

				for(int ii=0; ii<=jj; ++ii) {
					//cout << "\thash(uMat[" << ii << "]) = " << hash_cl(_uMat[ii]) << endl;
					_uMat[ii]->mul(-beta);
#if PROFILING == 1
					cerr << "PROFILING: Mul: " << _uMat[ii]->lastKernelRuntime() << endl;
#endif
					//cout << "\thash(uMat[" << ii << "]) = " << hash_cl(_uMat[ii]) << endl;
					_uMat[ii]->add(_matrix_residuals[ii]);
#if PROFILING == 1
					cerr << "PROFILING: Add: " << _uMat[ii]->lastKernelRuntime() << endl;
#endif
					//cout << "\thash(uMat[" << ii << "]) = " << hash_cl(_uMat[ii]) << endl;


#if BICGSTAB_SOLVER_ADDITIONAL_CHECKS == 1
		if(!this->checkMatrix(this->_uMat[ii])) throw "_uMat check 2 failed";
#endif
				}

				if(use_spatialDiffusion)
					generateAx(_uMat[jj], _uMat[jj+1], cl_lambda, cl_Dxx, cl_Dyy, cl_Dzz, cl_Dxy);
				else
					generateAx(_uMat[jj], _uMat[jj+1], cl_lambda);


				//printFull(_uMat[jj], "CL_uMat");
				//printFull(cl_lambda, "CL_Lambda");
				//printFull(_uMat[jj+1], "CL_Ax");


				//cout << "\thash(uMat[jj]) = " << hash_cl(_uMat[jj]) << endl;
				//cout << "\thash(uMat[jj+1]) = " << hash_cl(_uMat[jj+1]) << endl;
				//cout << "\thash(lambda) = " << hash_cl(cl_lambda) << endl;


				//cout << "\t<uMat[" << jj+1 << "],uMat[" << jj+1 << "]> = " << _uMat[jj+1]->dotProduct() << endl;

#if BICGSTAB_SOLVER_ADDITIONAL_CHECKS == 1
				if(!this->checkMatrix(this->_uMat[jj+1])) throw "_uMat check 3 failed";
				if(!this->checkMatrix(cl_resTilde)) throw "cl_resTilde check 3.1 failed";
				double dotProduct = _uMat[jj+1]->dotProduct(cl_resTilde);
				if(::isnan(dotProduct) || ::isinf(dotProduct)) throw "dotProduct check 3.1 failed";
#endif

				alpha = rho0/ (_uMat[jj+1]->dotProduct(cl_resTilde));
#if PROFILING == 1
				cerr << "PROFILING: dotProduct -- " << _uMat[jj+1]->lastKernelRuntime() << " ns" << endl;
#endif
				COUT << "\talpha = " << alpha << endl;

				for(int ii=0; ii<=jj; ii++) {
					_matrix_residuals[ii]->subMultiplied(_uMat[ii+1], alpha);
#if BICGSTAB_SOLVER_ADDITIONAL_CHECKS == 1
					if(!this->checkMatrix(this->_matrix_residuals[ii])) throw "matrix_residual check 4 failed";
#endif
#if PROFILING == 1
				cerr << "PROFILING: subMultiplied -- " << this->_matrix_residuals[ii]->lastKernelRuntime() << " ns" << endl;
#endif
				}


				if(use_spatialDiffusion) {
					generateAx(_matrix_residuals[jj], _matrix_residuals[jj+1], cl_lambda, cl_Dxx, cl_Dyy, cl_Dzz, cl_Dxy);
				} else {
					generateAx(_matrix_residuals[jj], _matrix_residuals[jj+1], cl_lambda);
				}
				//cout << "hash(residuals[" << jj+1 << "]) = " << hash_cl(_matrix_residuals[jj+1]) << endl;


#if BICGSTAB_SOLVER_ADDITIONAL_CHECKS == 1
		if(!this->checkMatrix(this->_matrix_residuals[jj])) throw "matrix_residual check 5 failed";
#endif

				cl_phi->addMultiplied(_uMat[0], alpha);
#if PROFILING == 1
				cerr << "PROFILING: addMultiplied -- " << cl_phi->lastKernelRuntime() << " ns" << endl;
#endif

#if BICGSTAB_SOLVER_ADDITIONAL_CHECKS == 1
				if(!this->checkMatrix(this->_uMat[0])) throw "_uMat check 6 failed";
#endif

				//cout << "End jj iteration " << jj << " - hash(cl_phi) = " << hash_cl(cl_phi) << endl;
			}			// END BiCG PART

			//cout << "END BiCG Part. <cl_phi,cl_phi> = " << cl_phi->dotProduct() << endl;



			// ================================================================================= //
			// 2015-06-19
			// BiCG PART CONSISTENT WITH REFERENCE IMPLEMENTATION
			//
			// ================================================================================= //

			// ==== MR PART ================================================ //
			// cout << "MR part of the solver ... " << endl;
			for(int jj=1; jj<=lValue; ++jj) {
				for(int ii=1; ii<jj; ++ii) {
					tau[ii][jj] = _matrix_residuals[jj]->dotProduct(_matrix_residuals[ii]) / sigma[ii];
					_matrix_residuals[jj]->subMultiplied(_matrix_residuals[ii], tau[ii][jj]);
#if PROFILING == 1
				cerr << "PROFILING: subMultiplied -- " << _matrix_residuals[jj]->lastKernelRuntime() << " ns" << endl;
#endif
				}


				sigma[jj] = _matrix_residuals[jj]->dotProduct();
				gammap[jj] = _matrix_residuals[0]->dotProduct(_matrix_residuals[jj])/sigma[jj];
#if PROFILING == 1
				cerr << "PROFILING: dotProduct -- " << _matrix_residuals[jj]->lastKernelRuntime() << " ns" << endl;
				cerr << "PROFILING: dotProduct -- " << _matrix_residuals[0]->lastKernelRuntime() << " ns" << endl;
#endif
			}

			_gamma[lValue] = gammap[lValue];
			omega = _gamma[lValue];
			//cout << "\tOmega = " << omega << endl;
			// CHECKED UNTIL HERE.


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
#if PROFILING == 1
			cerr << "PROFILING: addMultiplied: " << cl_phi->lastKernelRuntime() << endl;
#endif
			_matrix_residuals[0]->subMultiplied(_matrix_residuals[lValue], gammap[lValue]);
#if PROFILING == 1
			cerr << "PROFILING: subMultiplied: " << _matrix_residuals[0]->lastKernelRuntime() << endl;
#endif
			_uMat[0]->subMultiplied(_uMat[lValue], _gamma[lValue]);
#if PROFILING == 1
			cerr << "PROFILING: addMultiplied: " << _uMat[0]->lastKernelRuntime() << endl;
#endif

			//cout << "uMat[0].hash = " << hash_cl(_uMat[0]) << endl;
			//cout << "res[0].hash = " << hash_cl(_matrix_residuals[0]) << endl;

			for(int jj=1; jj<lValue; ++jj) {
				_uMat[0]->subMultiplied(_uMat[jj],_gamma[jj]);
#if PROFILING == 1
			cerr << "PROFILING: subMultiplied: " << _uMat[0]->lastKernelRuntime() << endl;
#endif
				cl_phi->addMultiplied(_matrix_residuals[jj],gammapp[jj]);
#if PROFILING == 1
			cerr << "PROFILING: addMultiplied: " << cl_phi->lastKernelRuntime() << endl;
#endif
				_matrix_residuals[0]->subMultiplied(_matrix_residuals[jj],gammap[jj]);
#if PROFILING == 1
			cerr << "PROFILING: subMultiplied: " << _matrix_residuals[0]->lastKernelRuntime() << endl;
#endif
			}

			norm = _matrix_residuals[0]->l2Norm();
#if PROFILING == 1
			cerr << "PROFILING: l2Norm: " << _matrix_residuals[0]->lastKernelRuntime() << endl;
#endif

			if(this->verbose) {
				cout << "Iteration " << iterations << ": NORM = " << norm;

#if TESTING == 1 && VERBOSE == 1
				const double phi_hash = hash_cl(cl_phi);
				cout << " hash(PHI) = " << phi_hash;
				const double residual_hash = hash_cl(_matrix_residuals[0]);
				cout << ",  hash(RESIDUAL) = " << residual_hash;

				//printFull(cl_phi, "CL_Phi_" + ::to_string(iterations));
				//printFull(_matrix_residuals[0], "CL_Residual_" + ::to_string(iterations));
#endif
				cout << endl;
			}

#if MAX_ITERATIONS > 0L
			if(iterations > MAX_ITERATIONS) {
				cerr << "BiCGStab_CL_Solver: EMERGENCY BREAK (no_iteration = " << iterations << ")" << endl;

				// Transfer results
				transferMatrix(cl_phi, phi);
				// Exit
				throw "Maximum number of iterations reached";
				break;
			}
#endif
		} while(norm > tolerance*normRhs);
		this->_iterations = iterations;

		if(verbose) cout << "Completed after " << iterations << " iterations" << endl;


		// Transfer results
		transferMatrix(cl_phi, phi);

		// Write result to file
		//printFull(phi, "Result_Phi_CL", false);

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

	// Normal cleanup
	DELETE(cl_phi);
	DELETE(cl_rhs);
	DELETE(cl_lambda);
	DELETE(cl_Dxx);
	DELETE(cl_Dyy);
	DELETE(cl_Dzz);
	DELETE(cl_Dxy);
}

void BiCGStabSolver::set_Grid(grid_1D &xGrid, grid_1D &yGrid, grid_1D &zGrid) {
	// Mark unused variables used
	MARK_USED(xGrid);
	MARK_USED(yGrid);
	MARK_USED(zGrid);
}

void BiCGStabSolver::set_Advection(NumMatrix<double,3> &ux_fine, BoundaryHandler3D &bounds, int dir) {
	MARK_USED(ux_fine);
	MARK_USED(bounds);
	MARK_USED(dir);

	throw "Advection not yet implemented for BiCGStab solver";
}

void BiCGStabSolver::setup(grid_1D &xGrid, grid_1D &yGrid, grid_1D &zGrid, double epsilon_,
		NumArray<int> &solverPars,
		bool spatial_diffusion, bool allow_offDiagDiffusion,
#ifdef parallel
		 mpi_manager_3D &MyMPI,
#endif
		int maxIter) {

	// Mark unused variables used
	MARK_USED(xGrid);
	MARK_USED(yGrid);
	MARK_USED(zGrid);
	MARK_USED(solverPars);


#ifdef parallel
	this->comm3d = MyMPI.comm3d;
	this->rank   = MyMPI.get_rank();
#else
	this->rank = 0;
#endif


#ifdef parallel
	MARK_USED(MyMPI);
#endif

	this->tolerance = epsilon_;
	this->_maxIterations = maxIter;
	this->lValue = solverPars(0);
	this->dim = 3;
	this->use_spatialDiffusion = spatial_diffusion;
	this->use_offDiagDiffusion = allow_offDiagDiffusion;

	this->set_Grid(xGrid, yGrid, zGrid);
}
