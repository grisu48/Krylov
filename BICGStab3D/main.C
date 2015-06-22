#include <iostream>
#include <fstream>
#include <stdio.h>
#include "grid_manager.H"
#include "BoundaryHandler.H"
#include "mpi_manager.H"
#include "LinSolver3D.hpp"
#include "solveLin_BICGStab.H"
// #include "SteadyStateMultigrid3DDiff.H"

using namespace std;


const double tolerance = 1e-6;



static void printMatXY(NumMatrix<double, 3> &mat, int z, ostream &out = cout) {
	ssize_t low[3];
	ssize_t high[3];
	for(int i=0;i<3;i++) {
		low[i] = mat.getLow(i);
		high[i] = mat.getHigh(i);
	}

	int rows = 0;
	for(ssize_t x=low[0]; x<high[0]; x++) {
		out << ++rows << "\t|";
		for(ssize_t y=low[1]; y<high[1]; y++) {
			out << '\t' << mat(x,y,z);
		}
		out << '\n';
	}
	out.flush();
}
static void printMat(NumMatrix<double, 3> &mat, ostream &out = cout) {
	const ssize_t low = mat.getLow(2);
	const ssize_t high = mat.getHigh(2);

	for(ssize_t z = low; z < high; z++) {
		printMatXY(mat, z, out);
	}
}



int main(int argc, char *argv[]) {
#ifdef parallel
	int ntasks;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
#endif

	NumArray<int> mx_global(3);
	mx_global[0] = 32;
	mx_global[1] = 32;
	mx_global[2] = 32;

	int mx1D;
	if(argc>1) {
		sscanf(argv[1],"%d",&mx1D);
		mx_global[0] = mx1D;
		mx_global[1] = mx1D;
		// mx[2] = mx1D/2;
		mx_global[2] = mx1D;
	}

	//int mx[3] = {64,64,64};


	int Nx_global[3] = {mx_global[0]+1, mx_global[1]+1,
	                    mx_global[2]+1};

	grid_manager MyGrid(0., 0., 0., 1., 1., 1., Nx_global[0],
	                    Nx_global[1], Nx_global[2], 1);

#ifdef parallel
	NumArray<int> nproc(3);
	nproc[0] = 2;
	nproc[1] = 2;
	nproc[2] = 1;
	mpi_manager_3D MyMPI(nproc, mx_global);

	// Grid for single mpi process
	grid_manager LocalGrid = MyMPI.make_LocalGrid(MyGrid);
#endif

	for(int bound=0; bound<6; ++bound) {
		MyGrid.set_bcType(bound, 0); // Set all boundaries to dirichlet
#ifdef parallel
		LocalGrid.set_bcType(bound, 0);
#endif
	}

	NumArray<int> mx(3);
#ifdef parallel
	mx[0] = LocalGrid.get_mx(0);
	mx[1] = LocalGrid.get_mx(1);
	mx[2] = LocalGrid.get_mx(2);
#else
	mx[0] = mx_global[0];
	mx[1] = mx_global[1];
	mx[2] = mx_global[2];
#endif

	//	std::cout << " using mx " << mx[0] << " " << mx[1] << " " << mx[2] << std::endl;

	// Boundary handler:
#ifdef parallel
	BoundaryHandler3D MyBounds(MyMPI);
#else
	BoundaryHandler3D MyBounds;
#endif
	MyBounds.set_bcType(0);

	// Prepare the problem:
	NumMatrix<double,3> phi_ana, phi;
	NumMatrix<double,3> rhs;
	NumMatrix<double,3> lambda;
	NumMatrix<double,3> DiffTens[4];

	phi.resize(Index::set(-1,-1,-1),
	           Index::set(mx[0]+1, mx[1]+1, mx[2]+1));
	phi_ana.resize(Index::set(-1,-1,-1),
	           Index::set(mx[0]+1, mx[1]+1, mx[2]+1));
	rhs.resize(Index::set(-1,-1,-1),
	           Index::set(mx[0]+1, mx[1]+1, mx[2]+1));
	lambda.resize(Index::set(-1,-1,-1),
	              Index::set(mx[0]+1, mx[1]+1, mx[2]+1));
	DiffTens[0].resize(Index::set(-1,-1,-1),
	                   Index::set(mx[0]+1, mx[1]+1, mx[2]+1));
	DiffTens[1].resize(Index::set(-1,-1,-1),
	                   Index::set(mx[0]+1, mx[1]+1, mx[2]+1));
	DiffTens[2].resize(Index::set(-1,-1,-1),
	                   Index::set(mx[0]+1, mx[1]+1, mx[2]+1));
	DiffTens[3].resize(Index::set(-1,-1,-1),
	                   Index::set(mx[0]+1, mx[1]+1, mx[2]+1));

	double dx = MyGrid.get_delx(0);
	double dy = MyGrid.get_delx(1);
	double dz = MyGrid.get_delx(2);

	double pi = M_PI;

	phi.clear();
	rhs.clear();

	NumArray<double> Diff(3);
	Diff(0) = 1.;
	Diff(1) = 1.;
	// Diff(2) = 100.;
	// Diff(2) = 100.;
	Diff(2) = 100.;
	Diff(2) = 10.;

	DiffTens[0].set_constVal(1.);
	DiffTens[1].set_constVal(1.);
	DiffTens[2].set_constVal(1.);
	DiffTens[3].clear();

	// choice of test to be run
	int switch_test(1);

	double DPar = 1.;
	double DPerp = 0.1;

	for(int iz = 0; iz <= mx[2]; ++iz) {
		for(int iy = 0; iy <= mx[1]; ++iy) {
			for(int ix = 0; ix <= mx[0]; ++ix) {
	// for(int iz = -1; iz <= mx[2]+1; ++iz) {
	// 	for(int iy = -1; iy <= mx[1]+1; ++iy) {
	// 		for(int ix = -1; ix <= mx[0]+1; ++ix) {

				// double xVal = ix*dx;
				// double yVal = iy*dy;
				// double zVal = iz*dz;

#ifdef parallel
				double xVal = LocalGrid.get_Pos(0,ix);
				double yVal = LocalGrid.get_Pos(1,iy);
				double zVal = LocalGrid.get_Pos(2,iz);
#else
				double xVal = MyGrid.get_Pos(0,ix);
				double yVal = MyGrid.get_Pos(1,iy);
				double zVal = MyGrid.get_Pos(2,iz);
#endif

				//cout << "[" << ix << "," << iy << "," << iz << "] = (" << xVal << "," << yVal << "," << zVal << ")" << endl;

				// phi_ana(ix,iy,iz) = sin(2.*pi*xVal)*sin(2.*pi*yVal)*
				// 	sin(2.*pi*zVal);
				phi_ana(ix,iy,iz) = sin(pi*xVal)*sin(pi*yVal)*sin(pi*zVal);
				lambda(ix,iy,iz) = 0.2*xVal*sqr(yVal)*zVal;
//				lambda(ix,iy,iz) = xVal;

				// rhs(ix,iy,iz) = -4*sqr(pi)*(Diff(0) + Diff(1) + Diff(2))*
				// 	phi_ana(ix,iy,iz);
				if(switch_test==1) {
					rhs(ix,iy,iz) = -(sqr(pi)*(Diff(0) + Diff(1) + Diff(2)) +
					                  lambda(ix,iy,iz))*phi_ana(ix,iy,iz);

					// DiffTens[0](ix,iy,iz) = 1.;
					// DiffTens[1](ix,iy,iz) = 1.;
					// DiffTens[2](ix,iy,iz) = zVal;

					// rhs(ix,iy,iz) = -sqr(pi)*(DiffTens[0](ix,iy,iz) +
					//                           DiffTens[1](ix,iy,iz) +
					//                           DiffTens[2](ix,iy,iz))*phi_ana(ix,iy,iz) +
					// 	pi*sin(pi*xVal)*sin(pi*yVal)*cos(pi*zVal);

				// if(ix==mx[0]/2 && iy==mx[1]/2 && iz==mx[2]/2) {
				// 	std::cout << " Example: " << rhs(ix,iy,iz) << std::endl;
				// }

				} else if (switch_test==2) {
				// if(ix==mx[0]/2 && iy==mx[1]/2 && iz==mx[2]/2) {
				// 	std::cout << " Example: " << rhs(ix,iy,iz) << std::endl;
				// }


				// Test 2 (räumliche Diffusion)
					phi_ana(ix,iy,iz) = sin(pi*xVal)*sin(pi*yVal)*sin(pi*zVal);

					DiffTens[0](ix,iy,iz) = yVal;
					DiffTens[1](ix,iy,iz) = xVal;
					DiffTens[2](ix,iy,iz) = zVal;

					rhs(ix,iy,iz) = -(sqr(pi)*(xVal + yVal + zVal) +
					                  lambda(ix,iy,iz))*phi_ana(ix,iy,iz) +
						pi*sin(pi*xVal)*sin(pi*yVal)*cos(pi*zVal);

				} else if (switch_test==-1) {
				// if(ix==mx[0]/2 && iy==mx[1]/2 && iz==mx[2]/2) {
				// 	std::cout << " Example: " << rhs(ix,iy,iz) << std::endl;
				// }


				// Test 2 (räumliche Diffusion)
					phi_ana(ix,iy,iz) = sin(pi*xVal)*sin(pi*yVal)*sin(pi*zVal);
					if(ix==mx[0] || iy==mx[1] || iz==mx[2]) {
						phi_ana(ix,iy,iz) = 0.;
					}

					DiffTens[0](ix,iy,iz) = 1.;
					DiffTens[1](ix,iy,iz) = 1.;
					DiffTens[2](ix,iy,iz) = 1. + 0.00000001*xVal;
					DiffTens[2](ix,iy,iz) = 1.;
					DiffTens[3](ix,iy,iz) = 0.;

					rhs(ix,iy,iz) = -(sqr(pi)*(1. + 0.00000001*xVal + 1. + 1.) +
					                  lambda(ix,iy,iz))*phi_ana(ix,iy,iz);
					rhs(ix,iy,iz) = -(sqr(pi)*(1. + 1. + 1.) +
					                  lambda(ix,iy,iz))*phi_ana(ix,iy,iz);
					if(iy==5 && iz==9) {
//						cout << ix << " " << rhs(ix,iy,iz) << " " << 1. + 0.00001*xVal + 1. + 1. << endl;
						cout << ix << " " << phi_ana(ix,iy,iz) << " ";
						cout << endl;
					}


				} else if (switch_test==3) {
					// Test 3 (räumliche Diffusion mit D_xy)

					double AVal = 0.1;//1.8;

					DiffTens[0](ix,iy,iz) = yVal;
					DiffTens[1](ix,iy,iz) = xVal;
					DiffTens[2](ix,iy,iz) = zVal;
					// DiffTens[3](ix,iy,iz) = AVal*sqr(xVal)*yVal*zVal;
					DiffTens[3](ix,iy,iz) = AVal*sqr(xVal)*yVal*zVal;
					double D_xy = DiffTens[3](ix,iy,iz);

					rhs(ix,iy,iz) = -(sqr(pi)*(xVal + yVal + zVal) +
					                  lambda(ix,iy,iz))*phi_ana(ix,iy,iz) +
						pi*sin(pi*xVal)*sin(pi*yVal)*cos(pi*zVal) +
						2.*D_xy*sqr(pi)*cos(pi*xVal)*cos(pi*yVal)*sin(pi*zVal)+
						2.*AVal*xVal*yVal*zVal*pi*sin(pi*xVal)*cos(pi*yVal)*sin(pi*zVal) +
						AVal*sqr(xVal)*zVal*pi*cos(pi*xVal)*sin(pi*yVal)*sin(pi*zVal);


				} else if (switch_test==4) {
					// cout << " muii" << endl;
					// Test 4 (räumliche Diffusion - Zylinderrichtung)
					phi_ana(ix,iy,iz) = sin(pi*xVal)*sin(pi*yVal)*sin(pi*zVal);

					double angle = atan2(yVal, xVal);

					DiffTens[0](ix,iy,iz) = (DPar*sqr(sin(angle)) +
					                         DPerp*sqr(cos(angle)));
					DiffTens[1](ix,iy,iz) = (DPar*sqr(cos(angle)) +
					                         DPerp*sqr(sin(angle)));
					DiffTens[2](ix,iy,iz) = DPerp;
					DiffTens[3](ix,iy,iz) = (DPerp - DPar)*sin(angle)*cos(angle);

					double Dxx = DiffTens[0](ix,iy,iz);
					double Dyy = DiffTens[1](ix,iy,iz);
					double Dzz = DiffTens[2](ix,iy,iz);
					double Dxy = DiffTens[3](ix,iy,iz);

					double sqrRad = sqr(xVal) + sqr(yVal);

					double dphidx = -yVal/sqrRad;
					double dphidy = xVal/sqrRad;

					double dDxxDx = 2.*(DPar - DPerp)*sin(angle)*cos(angle)*dphidx;
					double dDyyDy = 2.*(DPerp - DPar)*sin(angle)*cos(angle)*dphidy;

					double dDxyDx = (DPerp - DPar)*(sqr(cos(angle)) -
					                                sqr(sin(angle)))*dphidx;
					double dDxyDy = (DPerp - DPar)*(sqr(cos(angle)) -
					                                sqr(sin(angle)))*dphidy;

					rhs(ix,iy,iz) =
						(dDyyDy + dDxyDx)*pi*sin(pi*xVal)*cos(pi*yVal)*sin(pi*zVal) +
						(dDxxDx + dDxyDy)*pi*cos(pi*xVal)*sin(pi*yVal)*sin(pi*zVal) +
						2.*Dxy*sqr(pi)*cos(pi*xVal)*cos(pi*yVal)*sin(pi*zVal) -
						((Dxx + Dyy + Dzz)*sqr(pi) + lambda(ix,iy,iz))*phi_ana(ix,iy,iz);

				}

				// // Test 2 (räumliche Diffusion)
				// phi_ana(ix,iy,iz) = sin(pi*xVal)*sin(pi*yVal)*sin(pi*zVal);

				// DiffTens[0](ix,iy,iz) = yVal;
				// DiffTens[1](ix,iy,iz) = xVal;
				// DiffTens[2](ix,iy,iz) = zVal;

				// rhs(ix,iy,iz) = (-sqr(pi)*(xVal + yVal + zVal)*phi_ana(ix,iy,iz) +
				//                   pi*sin(pi*xVal)*sin(pi*yVal)*cos(pi*zVal));

				// // Test 3 (räumliche Diffusion in eine Richtung)
				// phi_ana(ix,iy,iz) = sin(pi*xVal)*sin(pi*yVal)*sin(pi*zVal);

				// DiffTens[0](ix,iy,iz) = 1.;
				// DiffTens[1](ix,iy,iz) = yVal;
				// DiffTens[2](ix,iy,iz) = 1.;

				// rhs(ix,iy,iz) = -sqr(pi)*(DiffTens[0](ix,iy,iz) +
				//                           DiffTens[1](ix,iy,iz) +
				//                           DiffTens[2](ix,iy,iz))*phi_ana(ix,iy,iz) +
				// 	pi*sin(pi*xVal)*cos(pi*yVal)*sin(pi*zVal);
				// DiffTens[0](ix,iy,iz) = xVal;
				// DiffTens[1](ix,iy,iz) = 1.;
				// DiffTens[2](ix,iy,iz) = 1.;

				// rhs(ix,iy,iz) = -sqr(pi)*(DiffTens[0](ix,iy,iz) +
				//                           DiffTens[1](ix,iy,iz) +
				//                           DiffTens[2](ix,iy,iz))*phi_ana(ix,iy,iz) +
				// 	pi*cos(pi*xVal)*sin(pi*yVal)*sin(pi*zVal);

			}
		}
	}

	// exit(3);

	//	lambda.set_constVal(0.);
	double diffErr = 1.e-8;

	// // // constructor without spatial diffusion
	// Steady_MGDiff SteadyDiffSolver(diffErr,1,1,1, MyGrid, false);

	// SteadyDiffSolver.solve(MyBounds, phi, rhs, lambda,
	//                        Diff(0), Diff(1), Diff(2), true);

	// exit(3);
	// constructor with spatial diffusion
	//	Steady_MGDiff SteadyDiffSolver(diffErr,1,1,1, MyGrid, true);

	cout << " bauen " << endl;
	Linsolver3D *lin_solver = NULL;

#ifdef parallel
	lin_solver = new BICGStab(LocalGrid, tolerance, 2, MyMPI);
#else
	lin_solver = new BICGStab(MyGrid, tolerance, 2);
#endif

	// SteadyDiffSolver.solve(MyBounds, phi, rhs, lambda,
	//                        DiffTens[0], DiffTens[1], DiffTens[2], true);

	cout << " lösen " << endl;

	if(switch_test==1) {
		lin_solver->solve(MyBounds, phi, rhs, lambda,
		                  Diff[0], Diff[1], Diff[2],8);
	} else if (switch_test==2 || switch_test==-1) {
		lin_solver->solve(MyBounds, phi, rhs, lambda,
		                  DiffTens[0], DiffTens[1], DiffTens[2], DiffTens[3],8);
	} else {
		lin_solver->solve(MyBounds, phi, rhs, lambda,
		                  DiffTens[0], DiffTens[1], DiffTens[2], DiffTens[3],8,true);
	}

	NumMatrix<double,3> & Dxx = DiffTens[0];
	NumMatrix<double,3> & Dyy = DiffTens[1];
	NumMatrix<double,3> & Dzz = DiffTens[2];
	NumMatrix<double,3> & Dxy = DiffTens[3];

	double coeff[3];
	coeff[0] = 1./sqr(dx);
	coeff[1] = 1./sqr(dy);
	coeff[2] = 1./sqr(dz);

	double coeff_xy = 1./(2.*dx*dy);


	// Now let's have a look:
	double error(0.), num(0.);
	for(int iz = 0; iz <= mx[2]; ++iz) {
		for(int iy = 0; iy <= mx[1]; ++iy) {
			for(int ix = 0; ix <= mx[0]; ++ix) {
				error += sqr((phi_ana(ix,iy,iz)) - phi(ix,iy,iz));
				num += 1.;
				// if(iy==mx[1]/2 && iz==mx[2]/2) {
				if(ix==mx[0]/2 && iz==mx[2]/2) {
					std::cout << iy << " " << phi_ana(ix,iy,iz) << " ";
					std::cout << phi(ix,iy,iz) << " ";
					std::cout << phi_ana(ix,iy,iz)-phi(ix,iy,iz) << " ";
					std::cout << std::endl;

					// double disc =
					// 	(coeff[0]*Dxx(ix,iy,iz)*(phi(ix+1,iy,iz) +
					// 	                         phi(ix-1,iy,iz)) +
					// 	 coeff[1]*Dyy(ix,iy,iz)*(phi(ix,iy+1,iz) +
					// 	                         phi(ix,iy-1,iz)) +
					// 	 coeff[2]*Dzz(ix,iy,iz)*(phi(ix,iy,iz+1) +
					// 	                         phi(ix,iy,iz-1)) -
					// 	 2.*(coeff[0]*Dxx(ix,iy,iz) + coeff[1]*Dyy(ix,iy,iz) +
					// 	     coeff[2]*Dzz(ix,iy,iz))*phi(ix,iy,iz) +
					// 	 coeff_xy*Dxy(ix,iy,iz)*(phi(ix+1,iy+1,iz) -
					// 	                         phi(ix+1,iy-1,iz) -
					// 	                         phi(ix-1,iy+1,iz) +
					// 	                         phi(ix-1,iy-1,iz)) +
					// 	 ((Dxx(ix+1,iy,iz) - Dxx(ix-1,iy,iz))/(2.*dx) +
					// 	  (Dxy(ix,iy+1,iz) - Dxy(ix,iy-1,iz))/(2.*dy))*
					// 	 (phi(ix+1,iy,iz) - phi(ix-1,iy,iz))/(2.*dx) +
					// 	 ((Dxy(ix+1,iy,iz) - Dxy(ix-1,iy,iz))/(2.*dx) +
					// 	  (Dyy(ix,iy+1,iz) - Dyy(ix,iy-1,iz))/(2.*dy))*
					// 	 (phi(ix,iy+1,iz) - phi(ix,iy-1,iz))/(2.*dy) +
					// 	 ((Dzz(ix,iy,iz+1) - Dzz(ix,iy,iz-1))/(2.*dz))*
					// 	 (phi(ix,iy,iz+1) - phi(ix,iy,iz-1))/(2.*dz));
					// std::cout << " res: " << disc << " ";
					// std::cout << rhs(ix,iy,iz) << " ";
					// std::cout << disc-rhs(ix,iy,iz) << " ";
					// std::cout << std::endl;

				}
			}
		}
	}
#ifdef parallel
	double error_global;
	MPI_Allreduce(&error, &error_global, 1, MPI_DOUBLE, MPI_SUM,
	              MyMPI.comm3d);

	error = error_global;
	double num_global;
	MPI_Allreduce(&num, &num_global, 1, MPI_DOUBLE, MPI_SUM,
	              MyMPI.comm3d);
	num = num_global;
#endif

	double l2err = sqrt(error/num);

#ifdef parallel
	if(MyMPI.get_rank()==0)
#endif
	std::cout << " l2 error for " << mx[0] << " is " << l2err << std::endl;

	delete lin_solver;

#ifdef parallel
	MPI_Finalize();
#endif

}
