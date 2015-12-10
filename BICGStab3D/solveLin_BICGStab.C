#include <iostream>
#include <fstream>
#include <stdlib.h>     /* exit, EXIT_FAILURE */
#include <iomanip>
#include <string>
#include <sstream>
#include <string.h>
#include <stdint.h>

#include "solveLin_BICGStab.H"
#include "time_ms.h"

using namespace std;




static inline uint64_t doubleToRawBits(double x) {
    uint64_t bits;
    memcpy(&bits, &x, sizeof bits);
    return bits;
}


uint64_t hash(NumMatrix<double, 3> &matrix) {
	//const long brick = 107534845447;		// Random prime number (HUGE)

	ssize_t _low[3];
	ssize_t _high[3];
	for(int i=0;i<3;i++) {
		_low[i]   = matrix.getLow(i);
		_high[i]  = matrix.getHigh(i);
	}

	uint64_t hashValue = 797003437L;			// Big prime number as starting point

	for(ssize_t x = _low[0]; x <= _high[0]; x++) {
		for(ssize_t y = _low[1]; y <= _high[1]; y++) {
			for(ssize_t z = _low[2]; z <= _high[2]; z++) {
				//value += (x*y*z) * matrix(x,y,z);
				hashValue = hashValue ^ doubleToRawBits(matrix(x,y,z));
			}
		}
	}

	return hashValue;
}

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
	if(relativePath) {
		string fname = f_filename(filename);
		out.open(fname.c_str());
	} else
		out.open(filename);
	printFull(matrix, out);
	out.close();
}




BICGStab::BICGStab(grid_manager &TheGrid, double tol, int LValue,
#ifdef parallel
                   mpi_manager_3D &MyMPI,
#endif
                   bool spatial_diffusion, bool allow_offDiagDiffusion) {

#ifdef parallel
	this->comm3d = MyMPI.comm3d;
	this->rank   = MyMPI.get_rank();
#else
	this->rank = 0;
#endif

	my_type = 20;

	this->LValue = LValue;
	dim = 3;

	int mx[3];
	for(int dir=0; dir<3; ++dir) {
		this->delx[dir] = TheGrid.get_delx(dir);
		mx[dir] = TheGrid.get_mx(dir);
	}
	// cout << " del: " << delx << " " << dely << " " << delz << endl;
	// exit(3);
	this->eps = tol;
	make_Arrays(mx[0], mx[1], mx[2]);
}


void BICGStab::setup(grid_1D &xGrid, grid_1D &yGrid, grid_1D &zGrid, double epsilon_,
		NumArray<int> &solverPars,
		bool spatial_diffusion, bool allow_offDiagDiffusion,
#ifdef parallel
		mpi_manager_3D &MyMPI,
#endif
		int maxIter) {

#ifdef parallel
	this->comm3d = MyMPI.comm3d;
	this->rank   = MyMPI.get_rank();
#else
	this->rank = 0;
#endif

	this->eps = epsilon_;
	this->LValue = solverPars(0);
	this->dim = 3;

	set_Grid(xGrid, yGrid, zGrid);

	make_Arrays(mx[0], mx[1], mx[2]);

}

void BICGStab::set_Grid(grid_1D &xGrid, grid_1D &yGrid, grid_1D &zGrid) {
	//! Store all relevant data of grid
	delx[0] = xGrid.get_del();
	delx[1] = yGrid.get_del();
	delx[2] = zGrid.get_del();

	mx[0] = xGrid.get_mx();
	mx[1] = yGrid.get_mx();
	mx[2] = zGrid.get_mx();

	//cout << " Grid " << mx[0] << " " << delx[0] << endl;

	return;
}


void BICGStab::make_Arrays(int mx, int my, int mz) {
	residuals = new NumMatrix<double,3> [LValue+1];
	uMat = new NumMatrix<double,3> [LValue+1];

	for(int level=0; level<=LValue; ++level) {
		residuals[level].resize(Index::set(-1,-1,-1),
		                        Index::set(mx+1, my+1, mz+1));
		residuals[level].clear();
		uMat[level].resize(Index::set(-1,-1,-1),
		                   Index::set(mx+1, my+1, mz+1));
		uMat[level].clear();
	}
	resTilde.resize(Index::set(-1,-1,-1),
	                Index::set(mx+1, my+1, mz+1));
	resTilde.clear();

}



void BICGStab::add_MatTimesVec(NumMatrix<double,3> &result,
                               NumMatrix<double,3> &vec) {
}


void BICGStab::solve(BoundaryHandler3D &bounds, NumMatrix<double,3> &phi,
                     NumMatrix<double,3> &rhs, NumMatrix<double,3> &lambda,
                     double D_xx, double D_yy, double D_zz, int debug,
                     double delt, bool evolve_time) {
	//! Main solver routine
	/*! Solves equations of the form:
	  \nabla\cdot\left(D\nabla \phi\right) - \lambda \phi = rhs
	 */
	this->debug = debug;
	use_spatialDiffusion = false;
	DiffDiag[0] = D_xx;
	DiffDiag[1] = D_yy;
	DiffDiag[2] = D_zz;

	solve_int(bounds, phi, rhs, lambda, rhs, rhs, rhs, rhs, debug);
}


void BICGStab::solve(BoundaryHandler3D &bounds, NumMatrix<double,3> &phi,
                     NumMatrix<double,3> &rhs, NumMatrix<double,3> &lambda,
                     NumMatrix<double,3> &Dxx, NumMatrix<double,3> &Dyy,
                     NumMatrix<double,3> &Dzz, NumMatrix<double,3> &Dxy,
                     int debug, bool use_offDiagDiffusion,
                     double delt, bool evolve_time) {
	//! Main solver routine
	/*! Solves equations of the form:
	  \nabla\cdot\left(D\nabla \phi\right) - \lambda \phi = rhs
	 */
	use_spatialDiffusion = true;
	this->debug = debug;
	this->use_offDiagDiffusion = use_offDiagDiffusion;
	if(!use_offDiagDiffusion) {
		Dxy.clear();
	}

	solve_int(bounds, phi, rhs, lambda, Dxx, Dyy, Dzz, Dxy, debug);
}

void BICGStab::solve(BoundaryHandler3D &bounds, NumMatrix<double,3> &phi,
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

void BICGStab::solve_int(BoundaryHandler3D &bounds,
		NumMatrix<double,3> &phi, NumMatrix<double,3> &rhs,
		NumMatrix<double,3> &lambda,
		NumMatrix<double,3> &Dxx, NumMatrix<double,3> &Dyy,
		NumMatrix<double,3> &Dzz, NumMatrix<double,3> &Dxy, int debug) {

//	rhs *= -1.;

	// Do boundaries if necessary for all variables:
#ifdef parallel
	bounds.do_BCs(Dxx, 1,-1,true);
	bounds.do_BCs(Dyy, 1,-1,true);
	bounds.do_BCs(Dzz, 1,-1,true);
	bounds.do_BCs(Dxy, 1,-1,true);
	// bounds.do_BCs(phi, 1);
	// bounds.do_BCs(lambda, 1);
	bounds.do_BCs(rhs, 1);
#endif
	double normRHS = get_l2Norm(rhs);
	if(normRHS == 0.) normRHS = 1.;


#if 0
	// Write initial matrices
	cout << "Writing initial matrices to file ... " << endl;
	printFull(phi, "Initial_phi");
	printFull(rhs, "Initial_rhs");
	printFull(lambda, "Initial_lambda");
	printFull(Dxx, "Initial_Dxx");
	printFull(Dyy, "Initial_Dyy");
	printFull(Dzz, "Initial_Dzz");
	printFull(Dxy, "Initial_Dxy");
#endif

	// cout << "Input variables:" << endl;
	// cout << "\thash(phi)     = " << hash(phi) << endl;
	// cout << "\thash(rhs)     = " << hash(rhs) << endl;
	// cout << "\thash(lambda)  = " << hash(lambda) << endl;



	int iter_steps=0;

	// compute r_0
	if(use_spatialDiffusion) {
		get_Residual(bounds, phi, rhs, lambda, residuals[0],
		             Dxx, Dyy, Dzz, Dxy);
	} else {
		get_Residual(bounds, phi, rhs, lambda, residuals[0]);
	}

	double norm_init = get_l2Norm(residuals[0]);

	if(norm_init < eps*normRHS) {
		if(debug>2 && true) {
			cout << " Initial Norm: " << norm_init << " " << norm_init/normRHS << " ";
			cout << normRHS << " " << eps*normRHS << " "  << iter_steps << endl;
			cout << " Doing no iterations " << endl;
		}
		return;
		//
	}


	resTilde = residuals[0];
	// cout << "<resTilde,resTilde> = " << dot_product(resTilde, resTilde) << endl;

	double norm(1.e99);
	double rho0(1.), rho1;
	double alpha(0.);
	double omega(1.);

	double tau[LValue+1][LValue+1];
	double sigma[LValue+1], gammap[LValue+1], gammapp[LValue+1];
	double gamma[LValue+1];

	long long iteration_runtime = -time_ms();
	do {
		iter_steps++;
		//		if(iter_steps==15) exit(3);
		if(rank==0 && debug>2 || false) {
			cout << " Iteration: " << iter_steps;
			if(iter_steps > 1) {
				iteration_runtime += time_ms();
				cout << " (" << iteration_runtime << " ms)";
			}
			cout << endl;
			iteration_runtime = -time_ms();
			//cout << "    hash(phi) = " << hash(phi) << endl;
		}

		rho0 *= -omega;


		// BI-CG part:
		for(int jj=0; jj<LValue; ++jj) {

			// cout << "residual[" << jj << "]) = " << hash(residuals[jj]) << endl;

			rho1 = dot_product(residuals[jj], resTilde);

			double beta = alpha*rho1/rho0;
			rho0 = rho1;


			// \hat u_i = \hat r_i - \beta \hat u_i
			for(int ii=0; ii<=jj; ++ii) {
				//cout << "\thash(uMat[" << ii << "]) = " << hash(uMat[ii]) << endl;
				uMat[ii] *= -beta;
				//cout << "\thash(uMat[" << ii << "]) = " << hash(uMat[ii]) << endl;
				uMat[ii] += residuals[ii];
				//cout << "\thash(uMat[" << ii << "]) = " << hash(uMat[ii]) << endl;
			}

#if 0
			cout << "Before multiply_withMat" << endl;
			cout << "\tuMat[jj] = " << hash(uMat[jj]) << endl;
			cout << "\tuMat[jj+1] = " << hash(uMat[jj+1]) << endl;
			cout << "\tlambda = " << hash(lambda) << endl;
			cout << "\tDxx = " << hash(Dxx) << endl;
			cout << "\tDyy = " << hash(Dyy) << endl;
			cout << "\tDzz = " << hash(Dzz) << endl;
			cout << "\tDxy = " << hash(Dxy) << endl;
#endif
			if(use_spatialDiffusion) {
				multiply_withMat(bounds, uMat[jj], lambda, Dxx, Dyy, Dzz, Dxy, uMat[jj+1]);
			} else {
				multiply_withMat(bounds, uMat[jj], lambda, uMat[jj+1]);
			}
#if 0
			cout << "generateAx completed for uMat[jj+1]" << endl;
			cout << "hash(uMat[jj = " << jj << "]) = " << hash(uMat[jj]) << endl;
			cout << "hash(uMat[jj+1 = " << jj+1 << "]) = " << hash(uMat[jj+1]) << endl;
			printFull(uMat[jj+1], "Ax");
			exit(8);
#endif
			//cout << "hash(lambda) = " << hash(lambda) << endl;

			//cout << "<uMat[" << jj+1 << "],uMat[" << jj+1 << "]> = " << dot_product(uMat[jj+1], uMat[jj+1]) << endl;

			alpha = rho0/dot_product(uMat[jj+1], resTilde);

			for(int ii=0; ii<=jj; ++ii) {
				residuals[ii] -= uMat[ii+1]*alpha;
			}

			// \hat r_{j+1} = A \hat r_j
			if(use_spatialDiffusion) {
				multiply_withMat(bounds, residuals[jj], lambda, Dxx, Dyy, Dzz, Dxy, residuals[jj+1]);
			} else {
				multiply_withMat(bounds, residuals[jj], lambda, residuals[jj+1]);
			}

			phi += uMat[0]*alpha;

			//cout << "End jj iteration " << jj << " - hash(phi) = " << hash(phi) << endl;
			if(reallyIsNan(alpha)) {
				cout << " alpha is nan " << jj << " " << rho0 << " " << rho1 << " " << beta << endl;
				exit(1);
			}
		}


		//cout << "alpha = " << alpha << endl;
		//cout << "<phi, phi> = " << dot_product(phi, phi) << endl;

		// if(iter_steps==1) exit(3);

		// MR part
		for(int jj=1; jj<=LValue; ++jj) {
			for(int ii=1; ii<jj; ++ii) {
				//				int ij = jj*(LValue+1) + ii;
				tau[ii][jj] = dot_product(residuals[jj], residuals[ii])/sigma[ii];

				residuals[jj] -= residuals[ii]*tau[ii][jj];

			}
			sigma[jj] = dot_product(residuals[jj],residuals[jj]);
			gammap[jj] = dot_product(residuals[0],residuals[jj])/sigma[jj];

		}
		omega = gamma[LValue] = gammap[LValue];
		//cout << " omega " << omega << endl;

		for(int jj=LValue-1; jj>=1; --jj) {
			gamma[jj] = gammap[jj];
			for(int ii=jj+1; ii<=LValue; ++ii) {
				gamma[jj] -= tau[jj][ii]*gamma[ii];
			}
		}

		for(int jj=1; jj<LValue; ++jj) {
			gammapp[jj] = gamma[jj+1];
			for(int ii=jj+1; ii<LValue; ++ii) {
				gammapp[jj] += tau[jj][ii]*gamma[ii+1];
			}
		}
//		cout << " gamma " << gamma[1] << " " << gamma[2] << " " << LValue << endl;

		// Set variables:
		phi += residuals[0]*gamma[1];
		residuals[0] -= residuals[LValue]*gammap[LValue];
		uMat[0] -= uMat[LValue]*gamma[LValue];

		//cout << "uMat[0].hash = " << hash(uMat[0]) << endl;
		//cout << "res[0].hash = " << hash(residuals[0]) << endl;


		for(int jj=1; jj<LValue; ++jj) {
			uMat[0] -= uMat[jj]*gamma[jj];
			phi     += residuals[jj]*gammapp[jj];
			residuals[0] -= residuals[jj]*gammap[jj];
		}

		norm = get_l2Norm(residuals[0]);
		if(rank==0 && debug>2 || false) {
			std::cout << " My error norm: " << norm << " ";
			std::cout << sqrt(norm) << " " << eps*normRHS << " ";
			std::cout << eps << " " << normRHS << " ";
			std::cout << endl;


#if 0
			const double hash_phi = hash(phi);
			const double hash_residual = hash(residuals[0]);
			std::cout << " hash(phi) = " << hash_phi;
			std::cout << ", hash(residual) " << hash_residual;
			std::cout << std::endl;

			stringstream ss;
			ss << "Phi" << iter_steps;
			string phi_filename = ss.str();
			ss.str("");
			ss << "Residual" << iter_steps;
			string res_filename = ss.str();
			printFull(phi, phi_filename.c_str());
			printFull(residuals[0], res_filename.c_str());
#endif
		}

//		if(iter_steps>1) exit(3);
		if(reallyIsNan(norm)) {
			cout << " Norm is nan at iteration " << iter_steps << endl;
			exit(3);
		}

	} while (norm > eps*normRHS);

	if(rank==0 && debug>2) {
//	if(rank==0 && debug>0) {
		std::cout << " Final error: " << norm << " after " << iter_steps;
		std::cout << " iterations " << endl;
	}

	if(rank==0 && debug>2) {
		cout << " Norm: " << norm << " " << norm/normRHS << " " << normRHS << " ";
		cout << eps*normRHS << " "  << iter_steps << endl;
	}
//	cout << residuals[0](3,11,8) << endl;

	// Check residual again:
	// compute r_0
	if(use_spatialDiffusion) {
		get_Residual(bounds, phi, rhs, lambda, residuals[0],
		             Dxx, Dyy, Dzz, Dxy);
	} else {
		get_Residual(bounds, phi, rhs, lambda, residuals[0], true);
	}
	norm = get_l2Norm(residuals[0]);

	if(rank==0 && false) {
		cout << " The norm: " << norm << endl;
	}

	// printFull(phi, "Result_Phi", false);

	// Solution is returned as stored in phi
}


