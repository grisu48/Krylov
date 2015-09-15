/*
 * Krylov.cpp
 *
 *  Created on: Sep 10, 2015
 *      Author: phoenix
 */

#include "Krylov.H"
#include "math.h"
#include <stdlib.h>     /* exit, EXIT_FAILURE */

using namespace std;


static const int DIM = 3;


Krylov::Krylov() {
	// Set default values. These values are expected to be then set by the underlying subclasses
	dim = 3;

	for(int i=0;i<3;i++) {
		delx[3] = 1.0;
		DiffDiag[3] = 1.0;
	}
	breaktol = 1e-3;
	eps = 1e-3;
}

Krylov::~Krylov() {
}




double Krylov::get_l2Norm(NumMatrix<double,3> &vec) {
	//! Compute l2 norm of field
	/*! Note: here we are prepared to use the parallel version, where shared
	  gridpoints are weighted accordingly
	 */

	int mx[DIM];
	for(int dir=0; dir<DIM; ++dir) {
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
//	valMax = 1.;

	if(valMax == 0.) return 0;
	double scaleFactor = 1./valMax;

	double sum(0.);
	if(DIM==3) {
		// for(int iz = 0; iz <= mx[2]; iz++) {
		// 	for(int iy = 0; iy <= mx[1]; iy++) {
		// 		for(int ix = 0; ix <= mx[0]; ix++) {
		// 			double valScale = scaleFactor*vec(ix,iy,iz);
		// 			sum += sqr(valScale);
		// 		}
		// 	}

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



double Krylov::dot_product(NumMatrix<double,3> &vecA,
                             NumMatrix<double,3> &vecB) {
	//! Compute the scalar product of two quantities
	int mx[DIM];
	for(int dir=0; dir<DIM; ++dir) {
		mx[dir] = vecA.getHigh(dir)-1;
	}

	double result=0.;
	if(DIM==3) {
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



void Krylov::get_Residual(BoundaryHandler3D &bounds,
                            NumMatrix<double,3> &psi, NumMatrix<double,3> &rhs,
                            NumMatrix<double,3> &lambda,
                            NumMatrix<double,3> &residual, bool hint){

	//! Compute residual of matrix equation
	/*! Version with spatially constant diffusion*/

	multiply_withMat(bounds, psi, lambda, residual, false, hint);

	//residual += rhs;
	residual = rhs - residual;

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




void Krylov::get_Residual(BoundaryHandler3D &bounds,
                            NumMatrix<double,3> &psi, NumMatrix<double,3> &rhs,
                            NumMatrix<double,3> &lambda,
                            NumMatrix<double,3> &residual,
                            NumMatrix<double,3> &Dxx,
                            NumMatrix<double,3> &Dyy,
                            NumMatrix<double,3> &Dzz,
                            NumMatrix<double,3> &Dxy){

	//! Compute residual of matrix equation

	multiply_withMat(bounds, psi, lambda, Dxx, Dyy, Dzz, Dxy, residual, false);

	//residual += rhs;
	residual = rhs - residual;
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
void Krylov::multiply_withMat(BoundaryHandler3D &bounds,
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



void Krylov::multiply_withMat(BoundaryHandler3D &bounds,
                                NumMatrix<double,3> &psi,
                                NumMatrix<double,3> &lambda,
                                NumMatrix<double,3> &vecOut,
                                bool apply_bcs, bool hint) {
	//! Compute residual of matrix equation (constant diffusion)
	int mx[dim];
	for(int dir=0; dir<dim; ++dir) {
		mx[dir] = psi.getHigh(dir)-1;
	}

	double coeff[dim];
	for(int dir=0; dir<dim; ++dir) {
		coeff[dir] = DiffDiag[dir]/sqr(delx[dir]);
	}

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


void Krylov::set_Advection(NumMatrix<double,3> &uext,
			BoundaryHandler3D &bounds, int dir) {
	//! Assign vector field component of external velocity field
	cerr << " Advection not implemented yet for Krylov solvers -- Exiting " << endl;
	exit(3);
}



bool Krylov::reallyIsNan(float x)
{
    //Assumes sizeof(float) == sizeof(int)
    int intIzedX = *(reinterpret_cast<int *>(&x));
    int clearAllNonNanBits = intIzedX & 0x7F800000;
    return clearAllNonNanBits == 0x7F800000;
}

