#include <iostream>
#include <fstream>
#include <stdlib.h>     /* exit, EXIT_FAILURE */
#include <iomanip>
#include <string>
#include <sstream>

#include "solveLin_BICGStab.H"
#include "time_ms.h"

using namespace std;




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

	this->LValue = LValue;
	this->dim = 3;

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


double BICGStab::get_l2Norm(NumMatrix<double,3> &vec) {
	//! Compute l2 norm of field
	/*! Note: here we are prepared to use the parallel version, where shared
	  gridpoints are weighted accordingly
	 */

	int mx[dim];
	for(int dir=0; dir<dim; ++dir) {
		mx[dir] = vec.getHigh(dir)-1;
	}

	double valMax = vec.max_norm();

	double mymax = -10;
	for(int iz = 0; iz <= mx[2]; iz++) {
		for(int iy = 0; iy <= mx[1]; iy++) {
			for(int ix = 0; ix <= mx[0]; ix++) {
			  mymax = std::max(mymax, std::abs(vec(ix,iy,iz)));
			}
		}
	}
	valMax = mymax;

#ifdef parallel
	double global_max;
	MPI_Allreduce(&valMax, &global_max, 1, MPI_DOUBLE, MPI_MAX,
	              comm3d);
	valMax = global_max;
#endif
	if(valMax == 0.) return 0;
	double scaleFactor = 1./valMax;

	double sum(0.);
	if(dim==3) {
		// for(int iz = 0; iz <= mx[2]; iz++) {
		// 	for(int iy = 0; iy <= mx[1]; iy++) {
		// 		for(int ix = 0; ix <= mx[0]; ix++) {
		// 			double valScale = scaleFactor*vec(ix,iy,iz);
		// 			sum += sqr(valScale);
		// 		}
		// 	}

//		cout << " Scale: " << valMax << " " << vec(16,16,16) << " " << vec(0,3,3) << " " << mymax << endl;;
		// }
		for(int iz = 1; iz < mx[2]; iz++) {
			for(int iy = 1; iy < mx[1]; iy++) {
				for(int ix = 1; ix < mx[0]; ix++) {
					double valScale = scaleFactor*vec(ix,iy,iz);
					sum += sqr(valScale);
				}
			}
		}

		// Now add boundary effects that may be shared by several
		// processes
		for(int iz = 1; iz < mx[2]; iz += 1) {
			for(int iy = 1; iy < mx[1]; iy += 1) {
				sum += 0.5*sqr(scaleFactor*vec(   0 ,iy,iz));
				sum += 0.5*sqr(scaleFactor*vec(mx[0],iy,iz));
			}
		}
		// at y-boundaries
		for(int iz = 1; iz < mx[2]; iz += 1) {
			for(int ix = 1; ix < mx[0]; ix += 1) {
				sum += 0.5*sqr(scaleFactor*vec(ix,   0 ,iz));
				sum += 0.5*sqr(scaleFactor*vec(ix,mx[1],iz));
			}
		}
		// at z-boundaries:
		for(int iy = 1; iy < mx[1]; iy += 1) {
			for(int ix = 1; ix < mx[0]; ix += 1) {
				sum += 0.5*sqr(scaleFactor*vec(ix,iy,   0 ));
				sum += 0.5*sqr(scaleFactor*vec(ix,iy,mx[2]));
			}
		}

		// Now add boundary values with weight 1/4 (possibly shared by
		// 4 processes)
		for(int iz = 1; iz < mx[2]; ++iz) {
			sum += 0.25*sqr(scaleFactor*vec(   0 ,   0 ,iz));
			sum += 0.25*sqr(scaleFactor*vec(   0 ,mx[1],iz));
			sum += 0.25*sqr(scaleFactor*vec(mx[0],   0 ,iz));
			sum += 0.25*sqr(scaleFactor*vec(mx[0],mx[1],iz));
		}
		for(int iy = 1; iy < mx[1]; ++iy) {
			sum += 0.25*sqr(scaleFactor*vec(   0 ,iy,   0 ));
			sum += 0.25*sqr(scaleFactor*vec(   0 ,iy,mx[2]));
			sum += 0.25*sqr(scaleFactor*vec(mx[0],iy,   0 ));
			sum += 0.25*sqr(scaleFactor*vec(mx[0],iy,mx[2]));
		}

		for(int ix = 1; ix < mx[0]; ++ix) {
			sum += 0.25*sqr(scaleFactor*vec(ix,   0 ,   0 ));
			sum += 0.25*sqr(scaleFactor*vec(ix,   0 ,mx[2]));
			sum += 0.25*sqr(scaleFactor*vec(ix,mx[1],   0 ));
			sum += 0.25*sqr(scaleFactor*vec(ix,mx[1],mx[2]));
		}

		// Finally add corner values with weight 1/8
		sum += 0.25*sqr(scaleFactor*vec(    0,    0,    0));
		sum += 0.25*sqr(scaleFactor*vec(    0,mx[1],    0));
		sum += 0.25*sqr(scaleFactor*vec(    0,    0,mx[2]));
		sum += 0.25*sqr(scaleFactor*vec(    0,mx[1],mx[2]));
		sum += 0.25*sqr(scaleFactor*vec(mx[0],    0,    0));
		sum += 0.25*sqr(scaleFactor*vec(mx[0],mx[1],    0));
		sum += 0.25*sqr(scaleFactor*vec(mx[0],    0,mx[2]));
		sum += 0.25*sqr(scaleFactor*vec(mx[0],mx[1],mx[2]));

	}

#ifdef parallel
	double global_sum;
	MPI_Allreduce(&sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM,
	              comm3d);
	sum = global_sum;
#endif
	// return l2 norm
	return valMax*sqrt(sum);
}



double BICGStab::dot_product(NumMatrix<double,3> &vecA,
                             NumMatrix<double,3> &vecB) {
	//! Compute the scalar product of two quantities
	int mx[dim];
	for(int dir=0; dir<dim; ++dir) {
		mx[dir] = vecA.getHigh(dir)-1;
	}

	double result=0.;
	if(dim==3) {
		for(int iz = 1; iz < mx[2]; iz += 1) {
			for(int iy = 1; iy < mx[1]; iy += 1) {
				for(int ix = 1; ix < mx[0]; ix += 1) {
					result += vecA(ix,iy,iz)*vecB(ix,iy,iz);
				}
			}
		}

// #ifdef parallel
		// Now add boundary values with weight 1/2:
		// at x-boundaries
		for(int iz = 1; iz < mx[2]; iz += 1) {
			for(int iy = 1; iy < mx[1]; iy += 1) {
				result +=  0.5*vecA(0,iy,iz)*vecB(0,iy,iz);
				result +=  0.5*vecA(mx[0],iy,iz)*vecB(mx[0],iy,iz);
			}
		}
		// at y-boundaries
		for(int iz = 1; iz < mx[2]; iz += 1) {
			for(int ix = 1; ix < mx[0]; ix += 1) {
				result += 0.5*vecA(ix,0,iz)*vecB(ix,0,iz);
				result += 0.5*vecA(ix,mx[1],iz)*vecB(ix,mx[1],iz);
			}
		}
		// at z-boundaries:
		for(int iy = 1; iy < mx[1]; iy += 1) {
			for(int ix = 1; ix < mx[0]; ix += 1) {
				result += 0.5*vecA(ix,iy,0)*vecB(ix,iy,0);
				result += 0.5*vecA(ix,iy,mx[2])*vecB(ix,iy,mx[2]);
			}
		}

		// Now add boundary values with weight 1/4 (possibly shared by
		// 4 processes)
		for(int iz = 1; iz < mx[2]; iz += 1) {
			result += 0.25*vecA(0,0,iz)*vecB(0,0,iz);
			result += 0.25*vecA(mx[0],0,iz)*vecB(mx[0],0,iz);
			result += 0.25*vecA(0,mx[1],iz)*vecB(0,mx[1],iz);
			result += 0.25*vecA(mx[0],mx[1],iz)*vecB(mx[0],mx[1],iz);
		}

		for(int iy = 1; iy < mx[1]; iy += 1) {
			result += 0.25*vecA(0,iy,0)*vecB(0,iy,0);
			result += 0.25*vecA(mx[0],iy,0)*vecB(mx[0],iy,0);
			result += 0.25*vecA(0,iy,mx[2])*vecB(0,iy,mx[2]);
			result += 0.25*vecA(mx[0],iy,mx[2])*vecB(mx[0],iy,mx[2]);
		}

		for(int ix = 1; ix < mx[0]; ix += 1) {
			result += 0.25*vecA(ix,0,0)*vecB(ix,0,0);
			result += 0.25*vecA(ix,mx[1],0)*vecB(ix,mx[1],0);
			result += 0.25*vecA(ix,0,mx[2])*vecB(ix,0,mx[2]);
			result += 0.25*vecA(ix,mx[1],mx[2])*vecB(ix,mx[1],mx[2]);
		}
		// Finally add boundary values with weight 1/8
		result += 0.125*vecA(0,0,0)*vecB(0,0,0);
		result += 0.125*vecA(mx[0],0,0)*vecB(mx[0],0,0);
		result += 0.125*vecA(0,mx[1],0)*vecB(0,mx[1],0);
		result += 0.125*vecA(mx[0],mx[1],0)*vecB(mx[0],mx[1],0);
		result += 0.125*vecA(0,0,mx[2])*vecB(0,0,mx[2]);
		result += 0.125*vecA(mx[0],0,mx[2])*vecB(mx[0],0,mx[2]);
		result += 0.125*vecA(0,mx[1],mx[2])*vecB(0,mx[1],mx[2]);
		result += 0.125*vecA(mx[0],mx[1],mx[2])*vecB(mx[0],mx[1],mx[2]);
// #endif
	}

#ifdef parallel
	double global_result;
	MPI_Allreduce(&result, &global_result, 1, MPI_DOUBLE, MPI_SUM,
	              comm3d);
	result = global_result;
#endif
	return result;
}


void BICGStab::get_Residual(BoundaryHandler3D &bounds,
                            NumMatrix<double,3> &psi, NumMatrix<double,3> &rhs,
                            NumMatrix<double,3> &lambda,
                            NumMatrix<double,3> &residual){

	//! Compute residual of matrix equation
	/*! Version with spatially constant diffusion*/

	multiply_withMat(bounds, psi, lambda, residual, false);
	residual += rhs;

	/*
	ofstream fout;
	fout.open("/home/phoenix/temp/Ax");
	printFull(psi, fout);
	fout.close();

	fout.open("/home/phoenix/temp/Residual");
	printFull(residual, fout);
	fout.close();
	cout << "Written" << endl;
	exit(8);
	*/

//	int mx[dim];
//	for(int dir=0; dir<dim; ++dir) {
//		mx[dir] = psi.getHigh(dir)-1;
//	}
//
//	double coeff[dim];
//	for(int dir=0; dir<dim; ++dir) {
//		coeff[dir] = DiffDiag[dir]/sqr(delx[dir]);
//	}
//
//
//	if(dim==3) {
//		for(int iz = 0; iz <= mx[2]; iz++) {
//			for(int iy = 0; iy <= mx[1]; iy++) {
//				for(int ix = 0; ix <= mx[0]; ix++) {
//					residual(ix,iy,iz) = (rhs(ix,iy,iz) +
//					                      (coeff[0]*(psi(ix+1,iy,iz) +
//					                                 psi(ix-1,iy,iz)) +
//					                       coeff[1]*(psi(ix,iy+1,iz) +
//					                                 psi(ix,iy-1,iz)) +
//					                       coeff[2]*(psi(ix,iy,iz+1) +
//					                                 psi(ix,iy,iz-1)) -
//					                       (2.*(coeff[0] + coeff[1] +
//					                            coeff[2]) + lambda(ix,iy,iz))*psi(ix,iy,iz)));
//
//				}
//			}
//		}
//	}

	bounds.do_BCs(residual, 1);

}




void BICGStab::get_Residual(BoundaryHandler3D &bounds,
                            NumMatrix<double,3> &psi, NumMatrix<double,3> &rhs,
                            NumMatrix<double,3> &lambda,
                            NumMatrix<double,3> &residual,
                            NumMatrix<double,3> &Dxx,
                            NumMatrix<double,3> &Dyy,
                            NumMatrix<double,3> &Dzz,
                            NumMatrix<double,3> &Dxy){

	//! Compute residual of matrix equation

	multiply_withMat(bounds, psi, lambda, Dxx, Dyy, Dzz, Dxy, residual, false);

	residual += rhs;
//	int mx[dim];
//	for(int dir=0; dir<dim; ++dir) {
//		mx[dir] = psi.getHigh(dir)-1;
//	}
//
//	double coeff[dim];
//	for(int dir=0; dir<dim; ++dir) {
//		coeff[dir] = 1./sqr(delx[dir]);
//	}
//
//	double coeff_xy = 1./(2.*delx[0]*delx[1]);
//
//
//	if(dim==3) {
//		for(int iz = 0; iz <= mx[2]; iz++) {
//			for(int iy = 0; iy <= mx[1]; iy++) {
//				for(int ix = 0; ix <= mx[0]; ix++) {
//					residual(ix,iy,iz) = rhs(ix,iy,iz) +
//						(coeff[0]*Dxx(ix,iy,iz)*(psi(ix+1,iy,iz) +
//						                         psi(ix-1,iy,iz)) +
//						 coeff[1]*Dyy(ix,iy,iz)*(psi(ix,iy+1,iz) +
//						                         psi(ix,iy-1,iz)) +
//						 coeff[2]*Dzz(ix,iy,iz)*(psi(ix,iy,iz+1) +
//						                         psi(ix,iy,iz-1)) -
//						 (2.*(coeff[0]*Dxx(ix,iy,iz) + coeff[1]*Dyy(ix,iy,iz) +
//						      coeff[2]*Dzz(ix,iy,iz)) +
//						  lambda(ix,iy,iz))*psi(ix,iy,iz) +
//						 coeff_xy*Dxy(ix,iy,iz)*(psi(ix+1,iy+1,iz) -
//						                         psi(ix+1,iy-1,iz) -
//						                         psi(ix-1,iy+1,iz) +
//						                         psi(ix-1,iy-1,iz)) +
//						 ((Dxx(ix+1,iy,iz) - Dxx(ix-1,iy,iz))/(2.*delx[0]) +
//						  (Dxy(ix,iy+1,iz) - Dxy(ix,iy-1,iz))/(2.*delx[1]))*
//						 (psi(ix+1,iy,iz) - psi(ix-1,iy,iz))/(2.*delx[0]) +
//						 ((Dxy(ix+1,iy,iz) - Dxy(ix-1,iy,iz))/(2.*delx[0]) +
//						  (Dyy(ix,iy+1,iz) - Dyy(ix,iy-1,iz))/(2.*delx[1]))*
//						 (psi(ix,iy+1,iz) - psi(ix,iy-1,iz))/(2.*delx[1]) +
//						 ((Dzz(ix,iy,iz+1) - Dzz(ix,iy,iz-1))/(2.*delx[2]))*
//						 (psi(ix,iy,iz+1) - psi(ix,iy,iz-1))/(2.*delx[2]));
//
//
//					// residual(ix,iy,iz) = rhs(ix,iy,iz) +
//					// 	(coeff[0]*Dxx(ix,iy,iz)*(psi(ix+1,iy,iz) +
//					// 	                         psi(ix-1,iy,iz)) +
//					// 	 coeff[1]*Dyy(ix,iy,iz)*(psi(ix,iy+1,iz) +
//					// 	                         psi(ix,iy-1,iz)) +
//					// 	 coeff[2]*Dzz(ix,iy,iz)*(psi(ix,iy,iz+1) +
//					// 	                         psi(ix,iy,iz-1)) -
//					// 	 (2.*(coeff[0]*Dxx(ix,iy,iz) + coeff[1]*Dyy(ix,iy,iz) +
//					// 	      coeff[2]*Dzz(ix,iy,iz)) +
//					// 	  lambda(ix,iy,iz))*psi(ix,iy,iz) +
//					// 	 ((Dxx(ix+1,iy,iz) - Dxx(ix-1,iy,iz))/(2.*delx[0]))*
//					// 	 (psi(ix+1,iy,iz) - psi(ix-1,iy,iz))/(2.*delx[0]) +
//					// 	 ((Dyy(ix,iy+1,iz) - Dyy(ix,iy-1,iz))/(2.*delx[1]))*
//					// 	 (psi(ix,iy+1,iz) - psi(ix,iy-1,iz))/(2.*delx[1]) +
//					// 	 ((Dzz(ix,iy,iz+1) - Dzz(ix,iy,iz-1))/(2.*delx[2]))*
//					// 	 (psi(ix,iy,iz+1) - psi(ix,iy,iz-1))/(2.*delx[2]));
//
//					// if(use_offDiagDiffusion) {
//					// 	residual(ix,iy,iz) +=
//					// 		coeff_xy*Dxy(ix,iy,iz)*(psi(ix+1,iy+1,iz) -
//					// 		                        psi(ix+1,iy-1,iz) -
//					// 		                        psi(ix-1,iy+1,iz) +
//					// 		                        psi(ix-1,iy-1,iz)) +
//					// 		((Dxy(ix,iy+1,iz) - Dxy(ix,iy-1,iz))/(2.*delx[1]))*
//					// 		(psi(ix+1,iy,iz) - psi(ix-1,iy,iz))/(2.*delx[0]) +
//					// 		((Dxy(ix+1,iy,iz) - Dxy(ix-1,iy,iz))/(2.*delx[0]))*
//					// 		(psi(ix,iy+1,iz) - psi(ix,iy-1,iz))/(2.*delx[1]);
//					// }
//
//				}
//			}
//		}
//	}

	bounds.do_BCs(residual, 1);
}

/**
 * Wendet (lambda, Dxx, Dyy, Dzz, Dxy) auf psi an und schreibt das Ergebnis auf vecOut
 */
void BICGStab::multiply_withMat(BoundaryHandler3D &bounds,
                                NumMatrix<double,3> &psi,
                                NumMatrix<double,3> &lambda,
                                NumMatrix<double,3> &Dxx,
                                NumMatrix<double,3> &Dyy,
                                NumMatrix<double,3> &Dzz,
                                NumMatrix<double,3> &Dxy,
                                NumMatrix<double,3> &vecOut,
                                bool apply_bcs) {
	//! Compute residual of matrix equation
	int mx[dim];
	for(int dir=0; dir<dim; ++dir) {
		mx[dir] = psi.getHigh(dir)-1;
	}

	double coeff[dim];
	for(int dir=0; dir<dim; ++dir) {
		coeff[dir] = 1./sqr(delx[dir]);
	}

	double coeff_xy = 1./(2.*delx[0]*delx[1]);

	if(dim==3) {
		for(int iz = 0; iz <= mx[2]; iz++) {
			for(int iy = 0; iy <= mx[1]; iy++) {
				for(int ix = 0; ix <= mx[0]; ix++) {

					vecOut(ix,iy,iz) =
						(coeff[0]*Dxx(ix,iy,iz)*(psi(ix+1,iy,iz) +
						                         psi(ix-1,iy,iz)) +
						 coeff[1]*Dyy(ix,iy,iz)*(psi(ix,iy+1,iz) +
						                         psi(ix,iy-1,iz)) +
						 coeff[2]*Dzz(ix,iy,iz)*(psi(ix,iy,iz+1) +
						                         psi(ix,iy,iz-1)) -
						 (2.*(coeff[0]*Dxx(ix,iy,iz) + coeff[1]*Dyy(ix,iy,iz) +
						      coeff[2]*Dzz(ix,iy,iz)) +
						  lambda(ix,iy,iz))*psi(ix,iy,iz) +

						  coeff_xy*Dxy(ix,iy,iz)*(psi(ix+1,iy+1,iz) -
						                         psi(ix+1,iy-1,iz) -
						                         psi(ix-1,iy+1,iz) +
						                         psi(ix-1,iy-1,iz)) +
						 ((Dxx(ix+1,iy,iz) - Dxx(ix-1,iy,iz))/(2.*delx[0]) +
						  (Dxy(ix,iy+1,iz) - Dxy(ix,iy-1,iz))/(2.*delx[1]))*
						 (psi(ix+1,iy,iz) - psi(ix-1,iy,iz))/(2.*delx[0]) +
						 ((Dxy(ix+1,iy,iz) - Dxy(ix-1,iy,iz))/(2.*delx[0]) +
						  (Dyy(ix,iy+1,iz) - Dyy(ix,iy-1,iz))/(2.*delx[1]))*
						 (psi(ix,iy+1,iz) - psi(ix,iy-1,iz))/(2.*delx[1]) +
						 ((Dzz(ix,iy,iz+1) - Dzz(ix,iy,iz-1))/(2.*delx[2]))*
						 (psi(ix,iy,iz+1) - psi(ix,iy,iz-1))/(2.*delx[2]));

					// vecOut(ix,iy,iz) =
					// 	(coeff[0]*Dxx(ix,iy,iz)*(psi(ix+1,iy,iz) +
					// 	                         psi(ix-1,iy,iz)) +
					// 	 coeff[1]*Dyy(ix,iy,iz)*(psi(ix,iy+1,iz) +
					// 	                         psi(ix,iy-1,iz)) +
					// 	 coeff[2]*Dzz(ix,iy,iz)*(psi(ix,iy,iz+1) +
					// 	                         psi(ix,iy,iz-1)) -
					// 	 (2.*(coeff[0]*Dxx(ix,iy,iz) + coeff[1]*Dyy(ix,iy,iz) +
					// 	      coeff[2]*Dzz(ix,iy,iz)) +
					// 	  lambda(ix,iy,iz))*psi(ix,iy,iz) +
					// 	 ((Dxx(ix+1,iy,iz) - Dxx(ix-1,iy,iz))/(2.*delx[0]))*
					// 	 (psi(ix+1,iy,iz) - psi(ix-1,iy,iz))/(2.*delx[0]) +
					// 	 ((Dyy(ix,iy+1,iz) - Dyy(ix,iy-1,iz))/(2.*delx[1]))*
					// 	 (psi(ix,iy+1,iz) - psi(ix,iy-1,iz))/(2.*delx[1]) +
					// 	 ((Dzz(ix,iy,iz+1) - Dzz(ix,iy,iz-1))/(2.*delx[2]))*
					// 	 (psi(ix,iy,iz+1) - psi(ix,iy,iz-1))/(2.*delx[2]));

					// if(use_offDiagDiffusion) {
					// 	vecOut(ix,iy,iz) +=
					// 		coeff_xy*Dxy(ix,iy,iz)*(psi(ix+1,iy+1,iz) -
					// 		                        psi(ix+1,iy-1,iz) -
					// 		                        psi(ix-1,iy+1,iz) +
					// 		                        psi(ix-1,iy-1,iz)) +
					// 		((Dxy(ix,iy+1,iz) - Dxy(ix,iy-1,iz))/(2.*delx[1]))*
					// 		(psi(ix+1,iy,iz) - psi(ix-1,iy,iz))/(2.*delx[0]) +
					// 		((Dxy(ix+1,iy,iz) - Dxy(ix-1,iy,iz))/(2.*delx[0]))*
					// 		(psi(ix,iy+1,iz) - psi(ix,iy-1,iz))/(2.*delx[1]);
					// }

				}
			}
		}
	}

	if(apply_bcs) {
		bounds.do_BCs(vecOut, 1);
	}

}



void BICGStab::multiply_withMat(BoundaryHandler3D &bounds,
                                NumMatrix<double,3> &psi,
                                NumMatrix<double,3> &lambda,
                                NumMatrix<double,3> &vecOut,
                                bool apply_bcs) {
	//! Compute residual of matrix equation (constant diffusion)
	int mx[dim];
	for(int dir=0; dir<dim; ++dir) {
		mx[dir] = psi.getHigh(dir)-1;
	}

	double coeff[dim];
	// cout << " Coeffs: ";
	for(int dir=0; dir<dim; ++dir) {
		coeff[dir] = DiffDiag[dir]/sqr(delx[dir]);
		// cout << coeff[dir] << " " << sqr(delx[dir]) << " ";
	}
	// cout << endl;
	// cout << " mx " << mx[1] << endl;

	bounds.do_BCs(psi, 1);

	if(dim==3) {


		for(int iz = 0; iz <= mx[2]; iz++) {
			for(int iy = 0; iy <= mx[1]; iy++) {
				for(int ix = 0; ix <= mx[0]; ix++) {
					const double value = (coeff[0]*(psi(ix+1,iy,iz) +
					                              psi(ix-1,iy,iz)) +
					                    coeff[1]*(psi(ix,iy+1,iz) +
					                              psi(ix,iy-1,iz)) +
					                    coeff[2]*(psi(ix,iy,iz+1) +
					                              psi(ix,iy,iz-1)) -
					                    (2.*(coeff[0] + coeff[1] + coeff[2]) + lambda(ix,iy,iz)) * psi(ix,iy,iz));
					vecOut(ix,iy,iz) = value;
				}
			}
		}
	}

	if(apply_bcs) {
		bounds.do_BCs(vecOut, 1);
	}

}

void BICGStab::add_MatTimesVec(NumMatrix<double,3> &result,
                               NumMatrix<double,3> &vec) {
}


void BICGStab::solve(BoundaryHandler3D &bounds, NumMatrix<double,3> &phi,
                     NumMatrix<double,3> &rhs, NumMatrix<double,3> &lambda,
                     double D_xx, double D_yy, double D_zz, int debug) {
	//! Main solver routine
	/*! Solves equations of the form:
	  \nabla\cdot\left(D\nabla \phi\right) - \lambda \phi = rhs
	 */
	this->debug = debug;
	use_spatialDiffusion = false;
	DiffDiag[0] = D_xx;
	DiffDiag[1] = D_yy;
	DiffDiag[2] = D_zz;

	solve_int(bounds, phi, rhs, lambda, rhs, rhs, rhs, rhs);
}


void BICGStab::solve(BoundaryHandler3D &bounds, NumMatrix<double,3> &phi,
                     NumMatrix<double,3> &rhs, NumMatrix<double,3> &lambda,
                     NumMatrix<double,3> &Dxx, NumMatrix<double,3> &Dyy,
                     NumMatrix<double,3> &Dzz, NumMatrix<double,3> &Dxy,
                     int debug, bool use_offDiagDiffusion) {
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

	solve_int(bounds, phi, rhs, lambda, Dxx, Dyy, Dzz, Dxy);
}

void BICGStab::solve_int(BoundaryHandler3D &bounds,
                         NumMatrix<double,3> &phi, NumMatrix<double,3> &rhs,
                         NumMatrix<double,3> &lambda,
                         NumMatrix<double,3> &Dxx, NumMatrix<double,3> &Dyy,
                         NumMatrix<double,3> &Dzz, NumMatrix<double,3> &Dxy) {

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

	cout << "Input variables:" << endl;
	cout << "\thash(phi)     = " << hash(phi) << endl;
	cout << "\thash(rhs)     = " << hash(rhs) << endl;
	cout << "\thash(lambda)  = " << hash(lambda) << endl;


	cout << "  normRHS = " << normRHS << endl;


	int iter_steps=0;

	// compute r_0
	if(use_spatialDiffusion) {
		get_Residual(bounds, phi, rhs, lambda, residuals[0],
		             Dxx, Dyy, Dzz, Dxy);
	} else {
		get_Residual(bounds, phi, rhs, lambda, residuals[0]);
	}

	resTilde = residuals[0];
	cout << "<resTilde,resTilde> = " << dot_product(resTilde, resTilde) << endl;

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
		if(rank==0 && debug>2) {
			cout << " Iteration: " << iter_steps;
			if(iter_steps > 1) {
				iteration_runtime += time_ms();
				cout << " (" << iteration_runtime << " ms)";
			}
			cout << endl;
			iteration_runtime = -time_ms();
			//cout << "    hash(phi) = " << hash(phi) << endl;
		}

		cout << "omega = " << omega << endl;
		rho0 *= -omega;
		cout << "rho0 = " << rho0 << endl;




		// BI-CG part:
		for(int jj=0; jj<LValue; ++jj) {
			cout << "jj iteration " << jj << endl;

			cout << "residual[" << jj << "]) = " << hash(residuals[jj]) << endl;

			rho1 = dot_product(residuals[jj], resTilde);
			cout << "rho1 = " << rho1 << endl;

			double beta = alpha*rho1/rho0;
			cout << "beta = " << beta << endl;
			//			cout << " Anf: " << beta << " " << rho0 << " " << rho1 << " " << alpha << endl;
			//			cout << " Some vals " << residuals[jj](3,5,9) << " " << resTilde(3,5,9) << endl;
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
			cout << "alpha = " << alpha << endl;
			// cout << " rho " << alpha << " " << rho0 << " " << dot_product(uMat[jj+1], resTilde) << endl;
			// cout << " Beta " << beta << " " << rho1 << endl;

			// cout << " resjj f " << residuals[jj](3,5,9) << " " << uMat[jj+1](3,5,9) << " " << alpha << endl;
			for(int ii=0; ii<=jj; ++ii) {
				residuals[ii] -= uMat[ii+1]*alpha;
			}
			// cout << " resjj " << residuals[jj](3,5,9) << " " << uMat[jj+1](3,5,9) <<  endl;
			// exit(3);

			// \hat r_{j+1} = A \hat r_j
			if(use_spatialDiffusion) {
				multiply_withMat(bounds, residuals[jj], lambda, Dxx, Dyy, Dzz, Dxy, residuals[jj+1]);
			} else {
				multiply_withMat(bounds, residuals[jj], lambda, residuals[jj+1]);
			}

			phi += uMat[0]*alpha;

			//cout << "End jj iteration " << jj << " - hash(phi) = " << hash(phi) << endl;
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
		if(rank==0 && debug>2) {
			std::cout << " My error norm: " << norm << " ";
			std::cout << sqrt(norm) << " " << eps*normRHS;

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



		if(iter_steps > 10) {
			cerr << "STOP iteration " << iter_steps << endl;
			exit(9);
		}


	} while (norm > eps*normRHS);

	if(rank==0) {
//	if(rank==0 && debug>0) {
		std::cout << " Final error: " << norm << " after " << iter_steps;
		std::cout << " iterations " << endl;
	}
	cout << residuals[0](3,11,8) << endl;

	// Check residual again:
	// compute r_0
	if(use_spatialDiffusion) {
		get_Residual(bounds, phi, rhs, lambda, residuals[0],
		             Dxx, Dyy, Dzz, Dxy);
	} else {
		get_Residual(bounds, phi, rhs, lambda, residuals[0]);
	}
	norm = get_l2Norm(residuals[0]);
	cout << " Other? norm " << norm << endl;
	cout << residuals[0](3,11,8) << endl;

	printFull(phi, "Result_Phi", false);

	// Solution is returned as stored in phi
}
