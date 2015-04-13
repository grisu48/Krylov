 #include <iostream>
#include "BoundaryHandler.H"
#include <stdlib.h>

using namespace std;


#ifdef parallel
BoundaryHandler4D::BoundaryHandler4D(mpi_manager_3D YourMPI) : BoundaryHandler(4) {

	MyMPI = YourMPI;

}
#endif

BoundaryHandler4D::BoundaryHandler4D() : BoundaryHandler(4) {


}


#ifdef parallel
BoundaryHandler3D::BoundaryHandler3D(mpi_manager_3D YourMPI, int type) : BoundaryHandler(3, type) {

	MyMPI = YourMPI;

}
#endif

BoundaryHandler3D::BoundaryHandler3D(int type) : BoundaryHandler(3, type) {


}

#ifdef parallel
BoundaryHandler2D::BoundaryHandler2D(mpi_manager_2D YourMPI, int type) : BoundaryHandler(2, type) {

	MyMPI = YourMPI;

}
#endif

BoundaryHandler2D::BoundaryHandler2D(int type) : BoundaryHandler(2, type) {

	
}


BoundaryHandler::BoundaryHandler(int DIM, int bcType, int test_case) {
	// Types of boundary conditions
	// (0) -> Dirichlet
	// (1) -> von Neumann
	// (2) -> periodic
	// (3) -> MPI periodic

	this->DIM = DIM;

	if(test_case>0) {
		is_testRun = true;
		cout << " ALERT: This is a test-run " << endl;
	} else {
		is_testRun = false;
	}

	// So far just Dirichlet
	bc_Type.resize(2*DIM);
	for(int dir=0; dir<2*DIM; ++dir) {
		cout << " Push " << dir << " " << bcType << endl;
		// bc_Type.push_back(bcType);
		bc_Type(dir) = bcType;
	}

	// bc_Type(0) = 0;
	// bc_Type(1) = 0;
	// bc_Type(2) = 0;
	// bc_Type(3) = 0;
	// bc_Type(4) = 0;
	// bc_Type(5) = 0;
	// bc_Type(1) = 0;
	eps = 1.e-42;
}

void BoundaryHandler::set_bcType(int bcType) {
	// bc_Type.clear();
	for(int dir=0; dir<2*DIM; ++dir) {
		cout << " Re-Push " << dir << " " << bcType << endl;
		// bc_Type.push_back(bcType);
		bc_Type(dir) = bcType;
	}
	//	bc_Type(0) = 0;
}

int BoundaryHandler::get_bcType(int bound) {
	//! Get type of boundary condition in direction dir

	cout << " at bound " << bound << " " << bc_Type(bound) << endl;

	return 549; //bc_Type(bound);
}
// #endif




void BoundaryHandler::do_BCsCellAve(NumMatrix<double,1> &spec,
                                    const grid_1D &TheGrid, int rim,
                                    double &S_lower, double &S_upper,
                                    double delt, int type) const {
	//! Boundary conditions for momentum direction
	
	/*!  For the lower boundary in the momentum direction we assume a
	  constant power law of the form spec = A p^-s, where A and s are
	  computed from the two gridpoints next to the
	  boundary. Computational grid runs from 0 to mx. These are the
	  bcs only for the cell averages.
	  At the high energy end we assume spec=0 beyond the computational
	  range instead

	  \param spec The field for which we need the bcs
	  \param TheGrid Grid manager
	  \param rim Number of ghost cells
	  \param type [optional] Indicate special cases like bcs for dedicated
	  tests or bcs for derivatives
	  \param delt [optional] timestep sice
	 */


	// Extrapolation at lower boundary:
	for(int ipos=-1; ipos>=-rim; --ipos) {
		double fac = ((TheGrid.get_del(ipos) + TheGrid.get_del(ipos+1))/
		              (TheGrid.get_del(ipos+1) + TheGrid.get_del(ipos+2)));
		spec(ipos) = spec(ipos+1) + fac*(spec(ipos+1) - spec(ipos+2));
		// cout << " Ipos: " << ipos << " " << spec(ipos) << endl;
	}

	// at the upper boundary we use values as they are
	return;


	// Lower boundary
	// Get parameters for power law spec = A p^-s
	// double SVal = log(spec(0)/spec(1))/log(TheGrid.get_xCen(1)/
	//                                        TheGrid.get_xCen(0));

	if(spec(0) > 0. && spec(1) > 0.) {
		S_lower = log(spec(0)/(spec(1)))/log(TheGrid.get_xCen(1)/
		                                     TheGrid.get_xCen(0));
	} else {
		S_lower = 0.;
	}
	double AVal = spec(0)*pow(TheGrid.get_xCen(0), S_lower);

	for(int ipos=-rim; ipos<0; ++ipos) {
		spec(ipos) = AVal*pow(TheGrid.get_xCen(ipos), -S_lower);
	}

	// Same for upper boundary
	int mx = TheGrid.get_mx();
	int Nx = TheGrid.get_Nx();
	
	// cout << " En Iks " << mx << " " << Nx << endl;
	// exit(2);
	// SVal = log(spec(mx-1)/spec(mx))/log(TheGrid.get_xCen(mx)/
	//                                     TheGrid.get_xCen(mx-1));
	if(spec(mx-1) > 0. && spec(mx) > 0.) {
		S_upper = log(spec(mx-1)/spec(mx))/log(TheGrid.get_xCen(mx)/
		                                       TheGrid.get_xCen(mx-1));
	} else {
		S_upper = 0.;
	}
	AVal = spec(mx)*pow(TheGrid.get_xCen(mx), S_upper);


	for(int ipos=0; ipos<rim; ++ipos) {
		spec(Nx+ipos) = AVal*pow(TheGrid.get_xCen(Nx+ipos), -S_upper);
	}

	// For analytical test use specific values at upper boundary:

	// power indices:
	double n_dpdt = 1.5;
	double n_src = 2.2;

	// pre factors
	double fac_dpdt(1.e-16);
	double fac_src(1.e-10);

	if(type==0) { // need solution directly:
		// lower boundary
		for(int ipos=-rim; ipos<0; ++ipos) {
			double mom = TheGrid.get_xCen(ipos);
			//			spec(Nx) = (pow(mom, -1.5) + 5./6.*1.e6*pow(mom, -2.7));
			spec(ipos) = (pow(mom, -n_dpdt) +
			              5./6.*1.e6*pow(mom, 1.-n_dpdt-n_src));
		}
		// // -> linear extrapolation instead
		// for(int ipos=-1; ipos>=-rim; --ipos) {
		// 	double fac = ((TheGrid.get_del(ipos) + TheGrid.get_del(ipos+1))/
		// 	              (TheGrid.get_del(ipos+1) + TheGrid.get_del(ipos+2)));
		// 	spec(ipos) = spec(ipos+1) + fac*(spec(ipos+1) - spec(ipos+2));
		// }

		// upper boundary
		for(int ipos=0; ipos<rim; ++ipos) {
			double mom = TheGrid.get_xCen(ipos+Nx);
			spec(Nx+ipos) = (pow(mom, -n_dpdt) +
			                 5./6.*1.e6*pow(mom, 1.-n_dpdt-n_src));
		}
	} else if (type==1) { // solution plus part of sources
		// lower boundary
		for(int ipos=-rim; ipos<0; ++ipos) {
			double mom = TheGrid.get_xCen(ipos);
			spec(ipos) = (pow(mom, -n_dpdt) +
			              5./6.*1.e6*pow(mom, 1.-n_dpdt-n_src) +
			              delt*0.5*fac_src*pow(mom, -n_src));
		}
		// // -> linear extrapolation instead
		// for(int ipos=-1; ipos>=-rim; --ipos) {
		// 	double fac = ((TheGrid.get_del(ipos) + TheGrid.get_del(ipos+1))/
		// 	              (TheGrid.get_del(ipos+1) + TheGrid.get_del(ipos+2)));
		// 	spec(ipos) = spec(ipos+1) + fac*(spec(ipos+1) - spec(ipos+2));
		// }

		// upper boundary
		for(int ipos=0; ipos<rim; ++ipos) {
			double mom = TheGrid.get_xCen(ipos+Nx);
			spec(Nx+ipos) = (pow(mom, -n_dpdt) +
			                 5./6.*1.e6*pow(mom,  1.-n_dpdt-n_src) +
			                 delt*0.5*fac_src*pow(mom, -n_src));
		}
	} else if (type==2) { // derivative of solution plus part of sources
		
		// -> linear extrapolation at lower boundary
		for(int ipos=-1; ipos>=-rim; --ipos) {
			double fac = ((TheGrid.get_del(ipos) + TheGrid.get_del(ipos+1))/
			              (TheGrid.get_del(ipos+1) + TheGrid.get_del(ipos+2)));
			spec(ipos) = spec(ipos+1) + fac*(spec(ipos+1) - spec(ipos+2));
		}

		// upper boundary
		for(int ipos=0; ipos<rim; ++ipos) {
			double mom = TheGrid.get_xCen(ipos+Nx);
			spec(Nx+ipos) = (-n_dpdt*pow(mom, -n_dpdt-1) +
			                 (1.-n_dpdt-n_src)*5./6.*1.e6*pow(mom, n_dpdt-n_src)-
			                 n_src*delt*0.5*fac_src*pow(mom, -n_src-1));
		}
		
	} else { // derivative of solution

		// -> linear extrapolation instead
		for(int ipos=-1; ipos>=-rim; --ipos) {
			double fac = ((TheGrid.get_del(ipos) + TheGrid.get_del(ipos+1))/
			              (TheGrid.get_del(ipos+1) + TheGrid.get_del(ipos+2)));
			spec(ipos) = spec(ipos+1) + fac*(spec(ipos+1) - spec(ipos+2));
		}

		// upper boundary
		for(int ipos=0; ipos<rim; ++ipos) {
			double mom = TheGrid.get_xCen(ipos+Nx);
			spec(Nx+ipos) = (-n_dpdt*pow(mom, -n_dpdt-1) +
			                 (1.-n_dpdt-n_src)*5./6.*1.e6*pow(mom, n_dpdt-n_src));
		}

	}
	// // Now use zero at upper boundaries instead
	// for(int ipos=0; ipos<rim; ++ipos) {
	// 	spec(Nx+ipos) = 0.;
	// }


	// For spectrum test:
	double s0 = 1.e-10;
	double powS = 2.2;
	powS = 2.5;
	powS = 2.5;
	powS = 2.2;
	double dotp0 = 1.e-16;
	double powdotp = 1.5;
	powdotp = 2.;
	powdotp = 1.2;
	powdotp = 1.5;
	double psi1 = 5.e6/6.;
	psi1 = -s0/(dotp0*(1.-powS));
	double hom_par = 1.; // Indicate of homogeneous solution is suppressed

	if(type==0) {
		for(int ipos=-rim; ipos<0; ++ipos) {

			double xPos = TheGrid.get_xCen(ipos);
			spec(ipos) = (hom_par*pow(xPos, -powdotp) +
			              psi1*pow(xPos, 1.-powdotp-powS));

		}
		
		for(int ipos=0; ipos<rim; ++ipos) {

			double xPos = TheGrid.get_xCen(Nx+ipos);
			spec(Nx+ipos) = (hom_par*pow(xPos, -powdotp) +
			                 psi1*pow(xPos, 1.-powdotp-powS));

		}
	} else if(type==1) {
		for(int ipos=-rim; ipos<0; ++ipos) {

			double xPos = TheGrid.get_xCen(ipos);
			spec(ipos) = (hom_par*pow(xPos, -powdotp) +
			              psi1*pow(xPos, 1.-powdotp-powS) +
			              delt*0.5*s0*pow(xPos, -powS));
			// cout << " Im Rand " << ipos << " " << spec(ipos) << endl;
		}
		
		for(int ipos=0; ipos<rim; ++ipos) {

			double xPos = TheGrid.get_xCen(Nx+ipos);
			spec(Nx+ipos) = (hom_par*pow(xPos, -powdotp) +
			                 psi1*pow(xPos, 1.-powdotp-powS) +
			                 delt*0.5*s0*pow(xPos, -powS));

		}
	} else if(type==2) {
		for(int ipos=-rim; ipos<0; ++ipos) {
			
			double xPos = TheGrid.get_xCen(ipos);
			double derivSol = (-powdotp*hom_par*pow(xPos, -powdotp-1) +
			                   (1.-powdotp-powS)*psi1*pow(xPos, -powdotp-powS));
			double derivSrc = -powS*s0*pow(xPos, -powS-1);

			spec(ipos) = derivSol + delt*0.5*derivSrc;
		}
		
		// std::cout << " Type " << type << std::endl;
		for(int ipos=0; ipos<rim; ++ipos) {

			double xPos = TheGrid.get_xCen(Nx+ipos);
			double derivSol = (-powdotp*hom_par*pow(xPos, -powdotp-1) +
			                   (1.-powdotp-powS)*psi1*pow(xPos, -powdotp-powS));
			double derivSrc = -powS*s0*pow(xPos, -powS-1);

			spec(Nx+ipos) = derivSol + delt*0.5*derivSrc;
			// std::cout << " bound " << Nx+ipos << " " << ipos << std::endl;
		}
	}

	// std::cout << " What is going on? " << rim << std::endl;
	// exit(3);

	// extrapolate at lower boundary
	for(int ipos=-1; ipos>=-rim; --ipos) {
		double fac = ((TheGrid.get_del(ipos) + TheGrid.get_del(ipos+1))/
		              (TheGrid.get_del(ipos+1) + TheGrid.get_del(ipos+2)));
		spec(ipos) = spec(ipos+1) + fac*(spec(ipos+1) - spec(ipos+2));
		// cout << " Ipos: " << ipos << " " << spec(ipos) << endl;
	}

	// specific bcs at upper boundary:
	if(is_testRun) {
		if(test_case==2) {

		}
	} else {
		// for production run - set zero at upper boundary
		for(int ipos=0; ipos<rim; ++ipos) {
			spec(Nx+ipos) = 0;
		}
	}

}


void BoundaryHandler3D::do_BCs(NumMatrix<double,3> &dist, int rim, int dir,
                               bool keepBoundVals) {
	//! Boundary conditions for spatial stuff
	/*!
	  Doing all directions at once
	  Option keepBoundVals allows to leave all non-MPI boundaries
	  constant
	 */
	
	bool xL_isOuterBound(true);
	bool xR_isOuterBound(true);
	bool yL_isOuterBound(true);
	bool yR_isOuterBound(true);
	bool zL_isOuterBound(true);
	bool zR_isOuterBound(true);
#ifdef parallel
	// In the parallel case we only do the normal boundaries at the
	// real boundaries of the domain
	xL_isOuterBound = MyMPI.is_OuterBoundary(0);
	xR_isOuterBound = MyMPI.is_OuterBoundary(1);
	yL_isOuterBound = MyMPI.is_OuterBoundary(2);
	yR_isOuterBound = MyMPI.is_OuterBoundary(3);
	zL_isOuterBound = MyMPI.is_OuterBoundary(4);
	zR_isOuterBound = MyMPI.is_OuterBoundary(5);
#endif

	int rimVar = -dist.getLow(0);
	NumArray<int> mx(3);
	mx[0]=dist.getHigh(0)-rimVar;
	mx[1]=dist.getHigh(1)-rimVar;
	mx[2]=dist.getHigh(2)-rimVar;

	bool do_xBound = false;
	bool do_yBound = false;
	bool do_zBound = false;
	if(dir==-1 || dir==0) {
		do_xBound = true;
	}
	if(dir==-1 || dir==1) {
		do_yBound = true;
	}
	if(dir==-1 || dir==2) {
		do_zBound = true;
	}

	// lower x-bondary
	if(do_xBound) {
		if(xL_isOuterBound && !keepBoundVals) {
			if(bc_Type[0] == 0) { // Dirichlet-boundaries
				// cout << " Look I am working " << endl;
				for(int iy = -rim; iy <= mx[1]+rim; ++iy) {
					for(int iz = -rim; iz <= mx[2]+rim; ++iz) { 
						for(int ipos=-rim; ipos<=0; ++ipos) {
							dist(ipos,iy,iz) = 0.;
						}
					}
				}
			} else if(bc_Type[0] == 1) { // Neumann-boundaries
				// cout << " Look I am screwing up " << endl;
				for(int iy = -rim; iy <= mx[1]+rim; ++iy) {
					for(int iz = -rim; iz <= mx[2]+rim; ++iz) { 
						for(int ipos=-1; ipos>=-rim; --ipos) {
							dist(ipos,iy,iz) = 0.;
							dist(ipos,iy,iz) = 2.*dist(ipos+1,iy,iz) -
								dist(ipos+2,iy,iz);
							// dist(ipos,iy,iz) = dist(ipos+1,iy,iz);
						}
					}
				}
			}
		}

		// upper x-boundary
		if(xR_isOuterBound && !keepBoundVals) {
			if(bc_Type[1] == 0) { // Dirichlet-boundaries
				for(int iy = -rim; iy <= mx[1]+rim; ++iy) {
					for(int iz = -rim; iz <= mx[2]+rim; ++iz) { 
						for(int ipos=mx[0]; ipos<=mx[0]+rim; ++ipos) {
							dist(ipos,iy,iz) = 0.;
						}
					}
				}
			} else if(bc_Type[1] == 1) { // Dirichlet-boundaries
				for(int iy = -rim; iy <= mx[1]+rim; ++iy) {
					for(int iz = -rim; iz <= mx[2]+rim; ++iz) { 
						for(int ipos=mx[0]; ipos<=mx[0]+rim; ++ipos) {
							dist(ipos,iy,iz) = 2.*dist(ipos-1,iy,iz) -
								dist(ipos-2,iy,iz);
						}
					}
				}
			}
		}
#ifdef parallel
		do_bc_MPI(dist, mx, 0, rim);
#endif
	}

// #ifdef parallel
// 	MyMPI.Finalise();
// #endif

// 	exit(3);
	
	if(do_yBound) {
	  //	  cout << " Doing y " << keepBoundVals << " " << yL_isOuterBound << " " << bc_Type[2] << endl;
		// lower y-boundary
		if(yL_isOuterBound && !keepBoundVals) {
			if(bc_Type[2] == 0) { // Dirichlet bcs:
				for(int ix = 0; ix <= mx[0]; ix++) {
					for(int iz = -rim; iz <= mx[2]+rim; ++iz) { 
						for(int ipos=-rim; ipos<=0; ++ipos) {
							dist(ix,ipos,iz) = 0.;
						}
					}
				}
			} else if(bc_Type[2] == 1) { // Neumann bcs:
				for(int ix = 0; ix <= mx[0]; ix++) {
					for(int iz = -rim; iz <= mx[2]+rim; ++iz) { 
						for(int ipos=-1; ipos>=-rim; --ipos) {
							dist(ix,ipos,iz) = 2.*dist(ix,ipos+1,iz) -
								dist(ix,ipos+2,iz);
						}
					}
				}
			}
		}
		

		// upper y-boundary
		if(yR_isOuterBound && !keepBoundVals) {
			if(bc_Type[3] == 0) { // Dirichlet bcs:
				for(int ix = 0; ix <= mx[0]; ix++) {
					for(int iz = -rim; iz <= mx[2]+rim; ++iz) { 
						for(int ipos=mx[1]; ipos<=mx[1]+rim; ++ipos) {
							dist(ix,ipos,iz) = 0.;
						}
					}
				}
			} else if(bc_Type[3] == 1) { // Neumann bcs:
				for(int ix = 0; ix <= mx[0]; ix++) {
					for(int iz = -rim; iz <= mx[2]+rim; ++iz) { 
						for(int ipos=mx[1]; ipos<=mx[1]+rim; ++ipos) {
							dist(ix,ipos,iz) = 2.*dist(ix,ipos-1,iz) -
								dist(ix,ipos-2,iz);
						}
					}
				}
			}
		}
#ifdef parallel
		do_bc_MPI(dist, mx, 1, rim);
#endif
	}		

	if(do_zBound) {
		// lower z-boundary
		if(zL_isOuterBound && !keepBoundVals) {
			if(bc_Type[4] == 0) {// Dirichlet bcs:
				for(int iy = 0; iy <= mx[1]; ++iy) {
					for(int ix = 0; ix <= mx[0]; ix++) {
						for(int ipos=-rim; ipos<=0; ++ipos) {
							dist(ix,iy,ipos) = 0.;
						}
					}
				}
			} else if(bc_Type[4] == 1) {// Neumann bcs:
				for(int iy = 0; iy <= mx[1]; ++iy) {
					for(int ix = 0; ix <= mx[0]; ix++) {
						for(int ipos=-rim; ipos<=0; ++ipos) {
							dist(ix,iy,ipos) = 2.*dist(ix,iy,ipos+1) -
								dist(ix,iy,ipos+2);
						}
					}
				}
			}
		}

		// upper z-boundary
		if(zR_isOuterBound && !keepBoundVals) {
			if(bc_Type[5] == 0) {// Dirichlet bcs:
				for(int iy = 0; iy <= mx[1]; ++iy) {
					for(int ix = 0; ix <= mx[0]; ix++) {
						for(int ipos=mx[2]; ipos<=mx[2]+rim; ++ipos) {
							dist(ix,iy,ipos) = 0.;
						}
					}
				}
			} else if(bc_Type[5] == 1) {// Neumann bcs:
				for(int iy = 0; iy <= mx[1]; ++iy) {
					for(int ix = 0; ix <= mx[0]; ix++) {
						for(int ipos=mx[2]; ipos<=mx[2]+rim; ++ipos) {
							dist(ix,iy,ipos) = 2.*dist(ix,iy,ipos-1) -
								dist(ix,iy,ipos-2);
						}
					}
				}
			}
		}
#ifdef parallel
		do_bc_MPI(dist, mx, 2, rim);
#endif
	}


}


#ifdef parallel
void BoundaryHandler3D::do_bc_MPI(NumMatrix<double,3> &data, NumArray<int> &mx,
                                  int dir, int rim) {
	//! MPI boundaries for a specific direction for 3D data
	/*! Here MPI boundaries are done where necessary. This necessity is first
	  determined from the mpi-handler.
	 */

	// Determine necessity to send / receive data in the current
	// dimension
	bool SendLeft  = !MyMPI.is_OuterBoundary(2*dir);
	bool SendRight = !MyMPI.is_OuterBoundary(2*dir+1);
	bool RecvRight = !MyMPI.is_OuterBoundary(2*dir+1);
	bool RecvLeft  = !MyMPI.is_OuterBoundary(2*dir);
	
	bool extended = false;
	if(rim>1) extended = true;


	int ext_num(2*rim+1); // Number of additional cells 
	int sizex = (mx[1]+ext_num)*(mx[2]+ext_num)*rim;
	int sizey = (mx[0]+ext_num)*(mx[2]+ext_num)*rim;
	int sizez = (mx[0]+ext_num)*(mx[1]+ext_num)*rim;
	int from(0), into(0); // Send & receive directions
	int size(0);


	// Prepare sending of data (if necessary)
	if(SendLeft) {

		if(dir == 0) { // x-direction

			if(!extended) {

				SendBuff2D.resize(Index::set(     -rim,      -rim),
				                  Index::set(mx[1]+rim, mx[2]+rim));

				for(int iz = -1; iz <= mx[2]+1; iz++) {
					for(int iy = -1; iy <= mx[1]+1; iy++) {
						SendBuff2D(iy,iz) = data(1,iy,iz);
					}
				}

			} else {

				SendBuff3D.resize(Index::set(   1,      -rim,      -rim),
				                  Index::set( rim, mx[1]+rim, mx[2]+rim));
				
				for(int iz = -1; iz <= mx[2]+1; iz++) {
					for(int iy = -1; iy <= mx[1]+1; iy++) {
						for(int ix = 1; ix<=rim; ix++) {
							SendBuff3D(ix,iy,iz) = data(ix,iy,iz);
						}
					}
				}
			}
			into = MyMPI.get_left();

		} else if(dir == 1) { // y-direction

			if(!extended) {

				SendBuff2D.resize(Index::set(     -rim,      -rim),
				                  Index::set(mx[0]+rim, mx[2]+rim));
				
				for(int iz = -1; iz <= mx[2]+1; iz++) {
					for(int ix = -1; ix <= mx[0]+1; ix++) {
						SendBuff2D(ix,iz) = data(ix,1,iz);
					}
				}

			} else {
				
				SendBuff3D.resize(Index::set(     -rim,   1,      -rim),
				                  Index::set(mx[0]+rim, rim, mx[2]+rim));
				
				for(int iz = -1; iz <= mx[2]+1; iz++) {
					for(int iy = 1; iy<=rim; iy++) {
						for(int ix = -1; ix <= mx[0]+1; ix++) {
							SendBuff3D(ix,iy,iz) = data(ix,iy,iz);
						}
					}
				}
			}
			into = MyMPI.get_front();

		} else { // z-direction

			if(!extended) {
				
				SendBuff2D.resize(Index::set(     -rim,      -rim),
				                  Index::set(mx[0]+rim, mx[1]+rim));
				
				for(int iy = -1; iy <= mx[1]+1; iy++) {
					for(int ix = -1; ix <= mx[0]+1; ix++) {
						SendBuff2D(ix,iy) = data(ix,iy,1);
					}
				}

			} else {

				SendBuff3D.resize(Index::set(     -rim,      -rim,   1),
				                  Index::set(mx[0]+rim, mx[1]+rim, rim));
				
				for(int iz = 1; iz<=rim; iz++) {
					for(int iy = -1; iy <= mx[1]+1; iy++) {
						for(int ix = -1; ix <= mx[0]+1; ix++) {
							SendBuff3D(ix,iy,iz) = data(ix,iy,iz);
						}
					}
				}
			}
			into = MyMPI.get_bottom();

		}

	} 

	if(RecvRight) {
		if(dir==0) { // x-direction

			if(!extended) {
				RecvBuff2D.resize(Index::set(     -rim,      -rim),
				                  Index::set(mx[1]+rim, mx[2]+rim));
			} else {
				RecvBuff3D.resize(Index::set(  1,      -rim,      -rim),
				                  Index::set(rim, mx[1]+rim, mx[2]+rim));
			}
			from = MyMPI.get_right();

		} else if (dir==1) { // y-direction

			if(!extended) {
				RecvBuff2D.resize(Index::set(     -rim,      -rim),
				                  Index::set(mx[0]+rim, mx[2]+rim));
			} else {
				RecvBuff3D.resize(Index::set(     -rim,   1,      -rim),
				                  Index::set(mx[0]+rim, rim, mx[2]+rim));
			}
			from = MyMPI.get_back();

		} else { // z-direction

			if(!extended) {
				RecvBuff2D.resize(Index::set(     -rim,      -rim),
				                  Index::set(mx[0]+rim, mx[1]+rim));
			} else {
				RecvBuff3D.resize(Index::set(     -rim,      -rim,   1),
				                  Index::set(mx[0]+rim, mx[1]+rim, rim));
			}
			from = MyMPI.get_top();

		}

	}

	if(SendLeft || RecvRight) { // Periodic

		// Set correct size:
		if(dir==0) {
			size = sizex;
		} else if (dir==1) {
			size = sizey;
		} else {
			size = sizez;
		}

		if(!extended) {
			do_MpiSendRecv(SendBuff2D, RecvBuff2D, from, into, size,
			               SendLeft, RecvRight);
		} else {
			do_MpiSendRecv(SendBuff3D, RecvBuff3D, from, into, size,
			               SendLeft, RecvRight);
		}
	}
	

	// Now assign the data if necessary
	if(RecvRight) {

		if(dir == 0) { // x-direction

			if(!extended) {
				for(int iz = -1; iz <= mx[2]+1; iz++) {
					for(int iy = -1; iy <= mx[1]+1; iy++) {
						data(mx[0]+1,iy,iz) = RecvBuff2D(iy,iz);
					}
				}
			} else {
				for(int iz = -1; iz <= mx[2]+1; iz++) {
					for(int iy = -1; iy <= mx[1]+1; iy++) {
						for(int ix = 1; ix<=rim; ix++) {
							data(mx[0]+ix,iy,iz) = RecvBuff3D(ix,iy,iz);
						}
					}
				}
			}

		} else if (dir==1) { // y-direction

			if(!extended) {
				for(int iz = -1; iz <= mx[2]+1; iz++) {
					for(int ix = -1; ix <= mx[0]+1; ix++) {
						data(ix, mx[1]+1,iz) = RecvBuff2D(ix,iz);
					}
				}
			} else {
				for(int iz = -1; iz <= mx[2]+1; iz++) {
					for(int iy = 1; iy<=rim; iy++) {
						for(int ix = -1; ix <= mx[0]+1; ix++) {
							data(ix, mx[1]+iy,iz) = RecvBuff3D(ix,iy,iz);
						}
					}
				}
			}

		} else { // z-direction

			if(!extended) {
				for(int iy = -1; iy <= mx[1]+1; iy++) {
					for(int ix = -1; ix <= mx[0]+1; ix++) {
						data(ix, iy, mx[2]+1) = RecvBuff2D(ix,iy);
					}
				}
			} else {
				for(int iz = 1; iz<=rim; iz++) {
					for(int iy = -1; iy <= mx[1]+1; iy++) {
						for(int ix = -1; ix <= mx[0]+1; ix++) {
							data(ix, iy, mx[2]+iz) = RecvBuff3D(ix,iy,iz);
						}
					}
				}
			}

		}
				
	}




	// Second part: transfer from right to left
	if(SendRight) {

		if(dir == 0) { // x-direction

			if(!extended) {

				SendBuff2D.resize(Index::set(     -rim,      -rim),
				                  Index::set(mx[1]+rim, mx[2]+rim));

				for(int iz = -1; iz <= mx[2]+1; iz++) {
					for(int iy = -1; iy <= mx[1]+1; iy++) {
						SendBuff2D(iy,iz) = data(mx[0]-1,iy,iz);
					}
				}

			} else {

				SendBuff3D.resize(Index::set(  1,      -rim,      -rim),
				                  Index::set(rim, mx[1]+rim, mx[2]+rim));

				for(int iz = -1; iz <= mx[2]+1; iz++) {
					for(int iy = -1; iy <= mx[1]+1; iy++) {
						for(int ix = 1; ix<=rim; ix++) {
							SendBuff3D(ix,iy,iz) = data(mx[0]-ix,iy,iz);
						}
					}
				}
				
			}
			into = MyMPI.get_right();

		} else if (dir == 1) { // y-direction

			if(!extended) {
				
				SendBuff2D.resize(Index::set(     -rim,      -rim),
				                  Index::set(mx[0]+rim, mx[2]+rim));

				for(int iz = -1; iz <= mx[2]+1; iz++) {
					for(int ix = -1; ix <= mx[0]+1; ix++) {
						SendBuff2D(ix,iz) = data(ix,mx[1]-1,iz);
					}
				}

			} else {

				SendBuff3D.resize(Index::set(     -rim,   1,      -rim),
				                  Index::set(mx[0]+rim, rim, mx[2]+rim));

				for(int iz = -1; iz <= mx[2]+1; iz++) {
					for(int iy = 1; iy<=rim; iy++) {
						for(int ix = -1; ix <= mx[0]+1; ix++) {
							SendBuff3D(ix,iy,iz) = data(ix,mx[1]-iy,iz);
						}
					}
				}

			}
			into = MyMPI.get_back();

		} else { // z-direction

			if(!extended) {
				
				SendBuff2D.resize(Index::set(     -rim,      -rim),
				                  Index::set(mx[0]+rim, mx[1]+rim));

				for(int iy = -1; iy <= mx[1]+1; iy++) {
					for(int ix = -1; ix <= mx[0]+1; ix++) {
						SendBuff2D(ix,iy) = data(ix,iy,mx[2]-1);
					}
				}

			} else {

				SendBuff3D.resize(Index::set(     -rim,      -rim,   1),
				                  Index::set(mx[0]+rim, mx[1]+rim, rim));

				for(int iz = 1; iz<=rim; iz++) {
					for(int iy = -1; iy <= mx[1]+1; iy++) {
						for(int ix = -1; ix <= mx[0]+1; ix++) {
							SendBuff3D(ix,iy,iz) = data(ix,iy,mx[2]-iz);
						}
					}
				}
				
			}
			into = MyMPI.get_top();

		}

	}

	if(RecvLeft) {

		if(dir==0) { // x-direction

			if(!extended) {

				RecvBuff2D.resize(Index::set(     -rim,      -rim),
				                  Index::set(mx[1]+rim, mx[2]+rim));

			} else {

				RecvBuff3D.resize(Index::set(  1,      -rim,      -rim),
				                  Index::set(rim, mx[1]+rim, mx[2]+rim));

			}
			from = MyMPI.get_left();

		} else if (dir==1) { // y-direction

			if(!extended) {

				RecvBuff2D.resize(Index::set(     -rim,      -rim),
				                  Index::set(mx[0]+rim, mx[2]+rim));

			} else {

				RecvBuff3D.resize(Index::set(     -rim,   1,      -rim),
				                  Index::set(mx[0]+rim, rim, mx[2]+rim));

			}
			from = MyMPI.get_front();

		} else { // z-direction

			if(!extended) {

				RecvBuff2D.resize(Index::set(     -rim,      -rim),
				                  Index::set(mx[0]+rim, mx[1]+rim));

			} else {

				RecvBuff3D.resize(Index::set(     -rim,      -rim,   1),
				                  Index::set(mx[0]+rim, mx[1]+rim, rim));

			}
			from = MyMPI.get_bottom();

		}
		
	}

	// Do the actual communication
	if(SendRight || RecvLeft) {

		// Set correct size:
		if(dir==0) {
			size = sizex;
		} else if (dir==1) {
			size = sizey;
		} else {
			size = sizez;
		}

		if(!extended) {
			do_MpiSendRecv(SendBuff2D, RecvBuff2D, from, into, size,
			               SendRight, RecvLeft);
		} else {
			do_MpiSendRecv(SendBuff3D, RecvBuff3D, from, into, size,
			               SendRight, RecvLeft);
		}
	}

	// Now assign the data if necessary:
	if(RecvLeft) {

		if(dir==0) { // x-direction

			if(!extended) {
				for(int iz = -1; iz <= mx[2]+1; iz++) {
					for(int iy = -1; iy <= mx[1]+1; iy++) {
						data(-1,iy,iz) = RecvBuff2D(iy,iz);
					}
				}
			} else {
				for(int iz = -1; iz <= mx[2]+1; iz++) {
					for(int iy = -1; iy <= mx[1]+1; iy++) {
						for(int ix = 1; ix<=rim; ix++) {
							data(-ix,iy,iz) = RecvBuff3D(ix,iy,iz);
						}
					}
				}
			}

		} else if(dir==1) { // y-direction

			if(!extended) {
				for(int iz = -1; iz <= mx[2]+1; iz++) {
					for(int ix = -1; ix <= mx[0]+1; ix++) {
						data(ix, -1,iz) = RecvBuff2D(ix,iz);
					}
				}
			} else {
				for(int iz = -1; iz <= mx[2]+1; iz++) {
					for(int iy = 1; iy<=rim; iy++) {
						for(int ix = -1; ix <= mx[0]+1; ix++) {
							data(ix,-iy,iz) = RecvBuff3D(ix,iy,iz);
						}
					}
				}
			}

		} else { // z-direction

			if(!extended) {
				for(int iy = -1; iy <= mx[1]+1; iy++) {
					for(int ix = -1; ix <= mx[0]+1; ix++) {
						data(ix, iy, -1) = RecvBuff2D(ix,iy);
					}
				}
			} else {
				for(int iz = 1; iz<=rim; iz++) {
					for(int iy = -1; iy <= mx[1]+1; iy++) {
						for(int ix = -1; ix <= mx[0]+1; ix++) {
							data(ix, iy, -iz) = RecvBuff3D(ix,iy,iz);
						}
					}
				}
			}

		}
	}

}

#endif


void BoundaryHandler2D::do_BCs(NumMatrix<double,2> &dist, int rim, int dir,
                               bool keepBoundVals) {
	NumMatrix<double,1> dummy;
	do_BCs(dist, dummy, dummy, dummy, dummy, rim, dir, keepBoundVals, false);
}





void BoundaryHandler2D::do_BCs(NumMatrix<double,2> &dist,
                               NumMatrix<double,1> &bcVals_xb,
                               NumMatrix<double,1> &bcVals_xe,
                               NumMatrix<double,1> &bcVals_yb,
                               NumMatrix<double,1> &bcVals_ye,
                               int rim, int dir,
                               bool keepBoundVals, bool use_ExtBCVal) {
	//! Boundary conditions for spatial stuff
	/*!
	  Doing only bcs in a 2D plane

	  \param dir restrict bc to direction dir - if dir=-1 do all directions

	  \param plane_normal normal to the 2D plane in 3D space. Options
	  are 0: x-y plane, 1: x-z plane, 2: y-z plane

	 */

	bool xL_isOuterBound(true);
	bool xR_isOuterBound(true);
	bool yL_isOuterBound(true);
	bool yR_isOuterBound(true);
#ifdef parallel
	// int dir_x(0), dir_y(1);
	// switch(plane_normal) {
	// case 0: // x-y plane
	// 	dir_x = 0;
	// 	dir_y = 1;
	// 	break;
	// case 1: // x-z plane
	// 	dir_x = 0;
	// 	dir_y = 2;
	// 	break;
	// case 2: // y-z plane
	// 	dir_x = 1;
	// 	dir_y = 2;
	// 	break;
	// }


	// xL_isOuterBound = MyMPI.is_OuterBoundary(2*dir_x);
	// xR_isOuterBound = MyMPI.is_OuterBoundary(2*dir_x+1);
	// yL_isOuterBound = MyMPI.is_OuterBoundary(2*dir_y);
	// yR_isOuterBound = MyMPI.is_OuterBoundary(2*dir_y+1);
	xL_isOuterBound = MyMPI.is_OuterBoundary(0);
	xR_isOuterBound = MyMPI.is_OuterBoundary(1);
	yL_isOuterBound = MyMPI.is_OuterBoundary(2);
	yR_isOuterBound = MyMPI.is_OuterBoundary(3);

#endif


	int rimVar = -dist.getLow(0);
	NumArray<int> mx(2);
	mx[0]=dist.getHigh(0)-rimVar;
	mx[1]=dist.getHigh(1)-rimVar;

	bool do_xBound = false;
	bool do_yBound = false;

	if(dir==-1 || dir==0) {
		do_xBound = true;
	}
	if(dir==-1 || dir==1) {
		do_yBound = true;
	}
// #ifdef parallel
// 	if(keepBoundVals == true && MyMPI.get_rank()==1) {
// 		cout << endl << endl << " My rim: " << rim << " " << rimVar << " " << dist(0,0) << endl;
// 	}
// #endif

	// lower x-bondary
	if(do_xBound) {
		// if(bc_Type[0] == 1) { // Dirichlet-boundaries
		if(xL_isOuterBound && !keepBoundVals) {
			if(use_ExtBCVal) {
// #ifdef parallel
// 				if(MyMPI.get_rank()==1) {
// 					cout << endl << endl << " Really?? " << endl << endl;
// 				}
// #endif
				for(int iy = -rim; iy <= mx[1]+rim; ++iy) {
					for(int ipos=-rim; ipos<=0; ++ipos) {
						dist(ipos,iy) = bcVals_xb(iy);
					}
				}
			} else {
				for(int iy = -rim; iy <= mx[1]+rim; ++iy) {
					for(int ipos=-rim; ipos<=0; ++ipos) {
						dist(ipos,iy) = 0.;
					}
				}
			}
		}
	

		// upper x-boundary
		// if(bc_Type[1] == 1) { // Dirichlet-boundaries
		if(xR_isOuterBound && !keepBoundVals) {
			if(use_ExtBCVal) {
				for(int iy = -rim; iy <= mx[1]+rim; ++iy) {
					for(int ipos=mx[0]; ipos<=mx[0]+rim; ++ipos) {
						dist(ipos,iy) = bcVals_xe(iy);
					}
				}
			} else {
				for(int iy = -rim; iy <= mx[1]+rim; ++iy) {
					for(int ipos=mx[0]; ipos<=mx[0]+rim; ++ipos) {
						dist(ipos,iy) = 0.;
					}
				}
			}
		}

#ifdef parallel
		do_bc_MPI(dist, mx, 0, rim);
#endif

	}

	if(do_yBound) {
// #ifdef parallel
// 		if(keepBoundVals && MyMPI.get_rank()==1) {
// 			cout << " Dist (b0): " << dist(0,0) << " " << rim << endl;
// 		}
// #endif
		// lower y-boundary
		// if(bc_Type[2] == 1) { // Dirichlet bcs:
		if(yL_isOuterBound && !keepBoundVals) {
			if(use_ExtBCVal) {
				for(int ix = 0; ix <= mx[0]; ix++) {
					for(int ipos=-rim; ipos<=0; ++ipos) {
						dist(ix,ipos) = bcVals_yb(ix);
					}
				}
			} else {
				for(int ix = 0; ix <= mx[0]; ix++) {
					for(int ipos=-rim; ipos<=0; ++ipos) {
						dist(ix,ipos) = 0.;
					}
				}
			}
		}

// #ifdef parallel
// 		if(keepBoundVals && MyMPI.get_rank()==1) {
// 			cout << " Dist (b1): " << dist(0,0) << endl;
// 		}
// #endif

		// upper y-boundary
	// if(bc_Type[3] == 1) { // Dirichlet bcs:
		if(yR_isOuterBound && !keepBoundVals) {
			if(use_ExtBCVal) {
				for(int ix = 0; ix <= mx[0]; ix++) {
					for(int ipos=mx[1]; ipos<=mx[1]+rim; ++ipos) {
						dist(ix,ipos) = bcVals_ye(ix);
					}
				}
			} else {
				for(int ix = 0; ix <= mx[0]; ix++) {
					for(int ipos=mx[1]; ipos<=mx[1]+rim; ++ipos) {
						dist(ix,ipos) = 0.;
					}
				}
			}
		}

// #ifdef parallel
// 		if(keepBoundVals && MyMPI.get_rank()==1) {
// 			cout << " Dist (c): " << dist(0,0) << endl;
// 		}
// #endif
#ifdef parallel
		do_bc_MPI(dist, mx, 1, rim);
#endif
	}


}



#ifdef parallel
void BoundaryHandler2D::do_bc_MPI(NumMatrix<double,2> &data, NumArray<int> &mx,
                                  int dir2D, int rim) {
	//! MPI boundaries for a specific direction for 2D data
	/*! Here MPI boundaries are done where necessary. This necessity is first
	  determined from the mpi-handler.

	  \param plane_normal normal to the 2D plane in 3D space. Options
	  are 0: x-y plane, 1: x-z plane, 2: y-z plane
	 */

	// assert(plane_normal < 3);

	// translate 2D direction to 3D space
	// int dir3D(0);
	// switch(plane_normal) {
	// case 0: // x-y plane
	// 	dir3D = dir2D;
	// 	break;
	// case 1: // x-z plane
	// 	if(dir2D==0) {
	// 		dir3D = 0;
	// 	} else if (dir2D==1) {
	// 		dir3D = 1;
	// 	}
	// 	break;
	// case 2: // y-z plane
	// 	dir3D = dir2D+1;
	// 	break;
	// }
	// Determine necessity to send / receive data in the current
	// dimension
	bool SendLeft  = !MyMPI.is_OuterBoundary(2*dir2D);
	bool SendRight = !MyMPI.is_OuterBoundary(2*dir2D+1);
	bool RecvRight = !MyMPI.is_OuterBoundary(2*dir2D+1);
	bool RecvLeft  = !MyMPI.is_OuterBoundary(2*dir2D);
	
	bool extended = false;
	if(rim>1) extended = true;

	
	int ext_num(2*rim+1); // Number of additional cells 
	int sizex = (mx[1]+ext_num)*rim;
	int sizey = (mx[0]+ext_num)*rim;

	int from(0), into(0); // Send & receive directions
	int size(0);

	
	// Prepare sending of data (if necessary)
	if(SendLeft) {

		if(dir2D == 0) {
			into = MyMPI.get_left();
		} else {
			into = MyMPI.get_front();
		}

		if(dir2D == 0) { // 1-direction in 2D space

			if(!extended) {

				SendBuff1D.resize(Index::set(     -rim),
				                  Index::set(mx[1]+rim));

				for(int iy = -1; iy <= mx[1]+1; iy++) {
					SendBuff1D(iy) = data(1,iy);
				}

			} else {

				SendBuff2D.resize(Index::set(   1,      -rim),
				                  Index::set( rim, mx[1]+rim));
				
				for(int iy = -1; iy <= mx[1]+1; iy++) {
					for(int ix = 1; ix<=rim; ix++) {
						SendBuff2D(ix,iy) = data(ix,iy);
					}
				}
			}

		} else { // 2-direction in 2D space

			if(!extended) {

				SendBuff1D.resize(Index::set(     -rim),
				                  Index::set(mx[0]+rim));
				
				for(int ix = -1; ix <= mx[0]+1; ix++) {
					SendBuff1D(ix) = data(ix,1);
				}

			} else {
				
				SendBuff2D.resize(Index::set(     -rim,   1),
				                  Index::set(mx[0]+rim, rim));
				
				for(int iy = 1; iy<=rim; iy++) {
					for(int ix = -1; ix <= mx[0]+1; ix++) {
						SendBuff2D(ix,iy) = data(ix,iy);
					}
				}
			}

		}

	} 

	if(RecvRight) {
		
		if(dir2D == 0) {
			from = MyMPI.get_right();
		} else {
			from = MyMPI.get_back();
		}

		if(dir2D==0) { // 1-direction in 2D space

			if(!extended) {
				RecvBuff1D.resize(Index::set(     -rim),
				                  Index::set(mx[1]+rim));
			} else {
				RecvBuff2D.resize(Index::set(  1,      -rim),
				                  Index::set(rim, mx[1]+rim));
			}

		} else { // 2-direction in 2D space

			if(!extended) {
				RecvBuff1D.resize(Index::set(     -rim),
				                  Index::set(mx[0]+rim));
			} else {
				RecvBuff2D.resize(Index::set(     -rim,   1),
				                  Index::set(mx[0]+rim, rim));
			}

		}

	}

	if(SendLeft || RecvRight) { // Periodic

		// Set correct size:
		if(dir2D==0) {
			size = sizex;
		} else {
			size = sizey;
		}

		if(!extended) {
			do_MpiSendRecv(SendBuff1D, RecvBuff1D, from, into, size,
			               SendLeft, RecvRight);
		} else {
			do_MpiSendRecv(SendBuff2D, RecvBuff2D, from, into, size,
			               SendLeft, RecvRight);
		}
	}
	

	// Now assign the data if necessary
	if(RecvRight) {

		if(dir2D == 0) { // 1-direction in 2D space

			if(!extended) {
				for(int iy = -1; iy <= mx[1]+1; iy++) {
					data(mx[0]+1,iy) = RecvBuff1D(iy);
				}
			} else {
				for(int iy = -1; iy <= mx[1]+1; iy++) {
					for(int ix = 1; ix<=rim; ix++) {
						data(mx[0]+ix,iy) = RecvBuff2D(ix,iy);
					}
				}
			}

		} else { // 2-direction in 2D space

			if(!extended) {
				for(int ix = -1; ix <= mx[0]+1; ix++) {
					data(ix, mx[1]+1) = RecvBuff1D(ix);
				}
			} else {
				for(int iy = 1; iy<=rim; iy++) {
					for(int ix = -1; ix <= mx[0]+1; ix++) {
						data(ix, mx[1]+iy) = RecvBuff2D(ix,iy);
					}
				}
			}

		}
				
	}


	// Second part: transfer from right to left
	if(SendRight) {

		// Find neighbours for MPI communication
		if(dir2D == 0) {
			into = MyMPI.get_right();
		} else {
			into = MyMPI.get_back();
		}

		if(dir2D == 0) { // 1-direction in 2D space

			if(!extended) {

				SendBuff1D.resize(Index::set(     -rim),
				                  Index::set(mx[1]+rim));

				for(int iy = -1; iy <= mx[1]+1; iy++) {
					SendBuff1D(iy) = data(mx[0]-1,iy);
				}

			} else {

				SendBuff2D.resize(Index::set(  1,      -rim),
				                  Index::set(rim, mx[1]+rim));

				for(int iy = -1; iy <= mx[1]+1; iy++) {
					for(int ix = 1; ix<=rim; ix++) {
						SendBuff2D(ix,iy) = data(mx[0]-ix,iy);
					}
				}
				
			}
			size = sizex;

		} else { // 2-direction in 2D space

			if(!extended) {
				
				SendBuff1D.resize(Index::set(     -rim),
				                  Index::set(mx[0]+rim));

				for(int ix = -1; ix <= mx[0]+1; ix++) {
					SendBuff1D(ix) = data(ix,mx[1]-1);
				}

			} else {

				SendBuff2D.resize(Index::set(     -rim,   1),
				                  Index::set(mx[0]+rim, rim));
				
				for(int iy = 1; iy<=rim; iy++) {
					for(int ix = -1; ix <= mx[0]+1; ix++) {
						SendBuff2D(ix,iy) = data(ix,mx[1]-iy);
					}
				}

			}
			size = sizey;

		}

	}

	if(RecvLeft) {

		if(dir2D == 0) {
			from = MyMPI.get_left();
		} else {
			from = MyMPI.get_front();
		}

		if(dir2D==0) { // 1-direction in 2D space

			if(!extended) {
				RecvBuff1D.resize(Index::set(     -rim),
				                  Index::set(mx[1]+rim));
			} else {
				RecvBuff2D.resize(Index::set(  1,      -rim),
				                  Index::set(rim, mx[1]+rim));
			}

		} else { // 2-direction in 2D space

			if(!extended) {
				RecvBuff1D.resize(Index::set(     -rim),
				                  Index::set(mx[0]+rim));
			} else {
				RecvBuff2D.resize(Index::set(     -rim,   1),
				                  Index::set(mx[0]+rim, rim));
			}

		}
		
	}

	// Do the actual communication
	if(SendRight || RecvLeft) {
		if(!extended) {
			do_MpiSendRecv(SendBuff1D, RecvBuff1D, from, into, size,
			               SendRight, RecvLeft);
		} else {
			do_MpiSendRecv(SendBuff2D, RecvBuff2D, from, into, size,
			               SendRight, RecvLeft);
		}
	}

	// Now assign the data if necessary:
	if(RecvLeft) {

		if(dir2D==0) { // 1-direction in 2D space

			if(!extended) {
				for(int iy = -1; iy <= mx[1]+1; iy++) {
					data(-1,iy) = RecvBuff1D(iy);
				}
			} else {
				for(int iy = -1; iy <= mx[1]+1; iy++) {
					for(int ix = 1; ix<=rim; ix++) {
						data(-ix,iy) = RecvBuff2D(ix,iy);
					}
				}
			}

		} else { // 2-direction in 2D space

			if(!extended) {
				for(int ix = -1; ix <= mx[0]+1; ix++) {
					data(ix, -1) = RecvBuff1D(ix);
				}
			} else {
				for(int iy = 1; iy<=rim; iy++) {
					for(int ix = -1; ix <= mx[0]+1; ix++) {
						data(ix,-iy) = RecvBuff2D(ix,iy);
					}
				}
			}

		}
	}

}





void BoundaryHandler3D::do_MpiSendRecv(NumMatrix<double,3> &Send,
                                       NumMatrix<double,3> &Recv,
                                       int from, int into, int size,
                                       bool do_Send, bool do_Receive)
{

	int numRequests = 0;
	if(do_Send) numRequests++;
	if(do_Receive) numRequests++;


	// Transfer of data:
	// initialize
	MPI_Request requests[numRequests];
	MPI_Status statusrl[numRequests];
	for(int ireq=0; ireq<numRequests; ireq++) {
		requests[ireq] = MPI_REQUEST_NULL;
	}
    
	// receive data
	// message tag -- must not be less than 0!
	int ireq(0);
	if(do_Receive) {
		int tag = from; 

		MPI_Irecv((double *)Recv, size, MPI_DOUBLE, from , tag,
		          MyMPI.comm3d, &requests[ireq]);
		ireq++;
	}
	
	if(do_Send) {
		int tag = MyMPI.get_rank();
    
		MPI_Isend((double *)Send, size, MPI_DOUBLE, into, tag,
		          MyMPI.comm3d, &requests[ireq]);
	}


	/* wait for all communication to complete */
	
	if(numRequests > 0) {
		MPI_Waitall(numRequests, requests, statusrl);
	}
    
}


void BoundaryHandler3D::do_MpiSendRecv(NumMatrix<double,2> &Send,
                                       NumMatrix<double,2> &Recv,
                                       int from, int into, int size,
                                       bool do_Send, bool do_Receive)
{

	int numRequests = 0;
	if(do_Send) numRequests++;
	if(do_Receive) numRequests++;


	// Transfer of data:
	// initialize
	MPI_Request requests[numRequests];
	MPI_Status statusrl[numRequests];
	for(int ireq=0; ireq<numRequests; ireq++) {
		requests[ireq] = MPI_REQUEST_NULL;
	}
    
	// receive data
	// message tag -- must not be less than 0!
	int ireq(0);
	if(do_Receive) {
		int tag = from; 

		MPI_Irecv((double *)Recv, size, MPI_DOUBLE, from , tag,
		          MyMPI.comm3d, &requests[ireq]);
		ireq++;
	}
	
	if(do_Send) {
		int tag = MyMPI.get_rank();
    
		MPI_Isend((double *)Send, size, MPI_DOUBLE, into, tag,
		          MyMPI.comm3d, &requests[ireq]);
	}


	/* wait for all communication to complete */
	
	if(numRequests > 0) {
		MPI_Waitall(numRequests, requests, statusrl);
	}
    
}


void BoundaryHandler2D::do_MpiSendRecv(NumMatrix<double,2> &Send,
                                       NumMatrix<double,2> &Recv,
                                       int from, int into, int size,
                                       bool do_Send, bool do_Receive)
{

	int numRequests = 0;
	if(do_Send) numRequests++;
	if(do_Receive) numRequests++;


	// Transfer of data:
	// initialize
	MPI_Request requests[numRequests];
	MPI_Status statusrl[numRequests];
	for(int ireq=0; ireq<numRequests; ireq++) {
		requests[ireq] = MPI_REQUEST_NULL;
	}
    
	// receive data
	// message tag -- must not be less than 0!
	int ireq(0);
	if(do_Receive) {
		int tag = from; 

		MPI_Irecv((double *)Recv, size, MPI_DOUBLE, from , tag,
		          MyMPI.comm2d, &requests[ireq]);
		ireq++;
	}
	
	if(do_Send) {
		int tag = MyMPI.get_rank();
    
		MPI_Isend((double *)Send, size, MPI_DOUBLE, into, tag,
		          MyMPI.comm2d, &requests[ireq]);
	}


	/* wait for all communication to complete */
	
	if(numRequests > 0) {
		MPI_Waitall(numRequests, requests, statusrl);
	}
    
}


void BoundaryHandler2D::do_MpiSendRecv(NumMatrix<double,1> &Send,
                                       NumMatrix<double,1> &Recv,
                                       int from, int into, int size,
                                       bool do_Send, bool do_Receive)
{

	int numRequests = 0;
	if(do_Send) numRequests++;
	if(do_Receive) numRequests++;


	// Transfer of data:
	// initialize
	MPI_Request requests[numRequests];
	MPI_Status statusrl[numRequests];
	for(int ireq=0; ireq<numRequests; ireq++) {
		requests[ireq] = MPI_REQUEST_NULL;
	}
    
	// receive data
	// message tag -- must not be less than 0!
	int ireq(0);
	if(do_Receive) {
		int tag = from; 

		MPI_Irecv((double *)Recv, size, MPI_DOUBLE, from , tag,
		          MyMPI.comm2d, &requests[ireq]);
		ireq++;
	}
	
	if(do_Send) {
		int tag = MyMPI.get_rank();
    
		MPI_Isend((double *)Send, size, MPI_DOUBLE, into, tag,
		          MyMPI.comm2d, &requests[ireq]);
	}


	/* wait for all communication to complete */
	
	if(numRequests > 0) {
		MPI_Waitall(numRequests, requests, statusrl);
	}
    
}


#endif



void BoundaryHandler4D::do_BCs(NumMatrix<double,4> &dist, int rim, int dir,
                               bool keepBoundVals) {
	//! Boundary conditions for spatial stuff
	/*!
	  Doing all directions at once
	  Here momentum dimension is also taken into account - but this is
	  done passively (no boundaries are done in that dimension)
	 */
	
	bool xL_isOuterBound(true);
	bool xR_isOuterBound(true);
	bool yL_isOuterBound(true);
	bool yR_isOuterBound(true);
	bool zL_isOuterBound(true);
	bool zR_isOuterBound(true);
#ifdef parallel
	// In the parallel case we only do the normal boundaries at the
	// real boundaries of the domain
	xL_isOuterBound = MyMPI.is_OuterBoundary(0);
	xR_isOuterBound = MyMPI.is_OuterBoundary(1);
	yL_isOuterBound = MyMPI.is_OuterBoundary(2);
	yR_isOuterBound = MyMPI.is_OuterBoundary(3);
	zL_isOuterBound = MyMPI.is_OuterBoundary(4);
	zR_isOuterBound = MyMPI.is_OuterBoundary(5);
#endif

	int rimVar = -dist.getLow(0);
	NumArray<int> mx(4);
	mx[0]=dist.getHigh(0)-rimVar;
	mx[1]=dist.getHigh(1)-rimVar;
	mx[2]=dist.getHigh(2)-rimVar;
	int rimP = -dist.getLow(3);
	mx[3]=dist.getHigh(3)-rimP;

	bool do_xBound = false;
	bool do_yBound = false;
	bool do_zBound = false;
	if(dir==-1 || dir==0) {
		do_xBound = true;
	}
	if(dir==-1 || dir==1) {
		do_yBound = true;
	}
	if(dir==-1 || dir==2) {
		do_zBound = true;
	}

	// lower x-bondary
	if(do_xBound) {
		// if(bc_Type[0] == 1) { // Dirichlet-boundaries
		if(xL_isOuterBound && !keepBoundVals) {
			for(int ip = -rimP; ip <= mx[3]+rimP; ++ip) {
				for(int iy = -rim; iy <= mx[1]+rim; ++iy) {
					for(int iz = -rim; iz <= mx[2]+rim; ++iz) { 
						for(int ipos=-rim; ipos<=0; ++ipos) {
							dist(ipos,iy,iz,ip) = 0.;
						}
					}
				}
			}
		}

		// upper x-boundary
		// if(bc_Type[1] == 1) { // Dirichlet-boundaries
		if(xR_isOuterBound && !keepBoundVals) {
			for(int ip = -rimP; ip <= mx[3]+rimP; ++ip) {
				for(int iy = -rim; iy <= mx[1]+rim; ++iy) {
					for(int iz = -rim; iz <= mx[2]+rim; ++iz) { 
						for(int ipos=mx[0]; ipos<=mx[0]+rim; ++ipos) {
							dist(ipos,iy,iz,ip) = 0.;
						}
					}
				}
			}
		}
#ifdef parallel
		do_bc_MPI(dist, mx, 0, rim, rimP);
#endif
	}

// #ifdef parallel
// 	MyMPI.Finalise();
// #endif

// 	exit(3);
	
	if(do_yBound) {
		// lower y-boundary
		// if(bc_Type[2] == 1) { // Dirichlet bcs:
		if(yL_isOuterBound && !keepBoundVals) {
			for(int ip = -rimP; ip <= mx[3]+rimP; ++ip) {
				for(int ix = 0; ix <= mx[0]; ix++) {
					for(int iz = -rim; iz <= mx[2]+rim; ++iz) { 
						for(int ipos=-rim; ipos<=0; ++ipos) {
							dist(ix,ipos,iz,ip) = 0.;
						}
					}
				}
			}
		}
		

		// upper y-boundary
		// if(bc_Type[3] == 1) { // Dirichlet bcs:
		if(yR_isOuterBound && !keepBoundVals) {
			for(int ip = -rimP; ip <= mx[3]+rimP; ++ip) {
				for(int ix = 0; ix <= mx[0]; ix++) {
					for(int iz = -rim; iz <= mx[2]+rim; ++iz) { 
						for(int ipos=mx[1]; ipos<=mx[1]+rim; ++ipos) {
							dist(ix,ipos,iz,ip) = 0.;
						}
					}
				}
			}
		}
#ifdef parallel
		do_bc_MPI(dist, mx, 1, rim, rimP);
#endif
	}		

	if(do_zBound) {
		// lower z-boundary
		// if(bc_Type[4] == 1) {// Dirichlet bcs:
		if(zL_isOuterBound && !keepBoundVals) {
			for(int ip = -rimP; ip <= mx[3]+rimP; ++ip) {
				for(int iy = 0; iy <= mx[1]; ++iy) {
					for(int ix = 0; ix <= mx[0]; ix++) {
						for(int ipos=-rim; ipos<=0; ++ipos) {
							dist(ix,iy,ipos,ip) = 0.;
						}
					}
				}
			}
		}

		// upper z-boundary
		// if(bc_Type[5] == 1) {// Dirichlet bcs:
		if(zR_isOuterBound && !keepBoundVals) {
			for(int ip = -rimP; ip <= mx[3]+rimP; ++ip) {
				for(int iy = 0; iy <= mx[1]; ++iy) {
					for(int ix = 0; ix <= mx[0]; ix++) {
						for(int ipos=mx[2]; ipos<=mx[2]+rim; ++ipos) {
							dist(ix,iy,ipos,ip) = 0.;
						}
					}
				}
			}
		}
#ifdef parallel
		do_bc_MPI(dist, mx, 2, rim, rimP);
#endif
	}


}



#ifdef parallel
void BoundaryHandler4D::do_bc_MPI(NumMatrix<double,4> &data, NumArray<int> &mx,
                                  int dir, int rim, int rimP) {
	//! MPI boundaries for a specific direction for 4D data
	/*! Here MPI boundaries are done where necessary. This necessity is first
	  determined from the mpi-handler.
	  fourth direction may have a different number of boundary cells.
	 */

	// Determine necessity to send / receive data in the current
	// dimension
	bool SendLeft  = !MyMPI.is_OuterBoundary(2*dir);
	bool SendRight = !MyMPI.is_OuterBoundary(2*dir+1);
	bool RecvRight = !MyMPI.is_OuterBoundary(2*dir+1);
	bool RecvLeft  = !MyMPI.is_OuterBoundary(2*dir);
	
	bool extended = false;
	if(rim>1) extended = true;


	int ext_num(2*rim+1); // Number of additional cells 
	int ext_numP(2*rimP+1); // same for momentum dimension
	int sizex = (mx[3]+ext_numP)*(mx[1]+ext_num)*(mx[2]+ext_num)*rim;
	int sizey = (mx[3]+ext_numP)*(mx[0]+ext_num)*(mx[2]+ext_num)*rim;
	int sizez = (mx[3]+ext_numP)*(mx[0]+ext_num)*(mx[1]+ext_num)*rim;
	int from(0), into(0); // Send & receive directions
	int size(0);


	// Prepare sending of data (if necessary)
	if(SendLeft) {

		if(dir == 0) { // x-direction

			if(!extended) {

				SendBuff3D.resize(Index::set(     -rim,      -rim,      -rimP),
				                  Index::set(mx[1]+rim, mx[2]+rim, mx[3]+rimP));

				for(int ip = -rimP; ip <= mx[3]+rimP; ++ip) {
					for(int iz = -1; iz <= mx[2]+1; iz++) {
						for(int iy = -1; iy <= mx[1]+1; iy++) {
							SendBuff3D(iy,iz,ip) = data(1,iy,iz,ip);
						}
					}
				}

			} else {

				SendBuff4D.resize(Index::set(   1,      -rim,      -rim,
				                                -rimP),
				                  Index::set( rim, mx[1]+rim, mx[2]+rim,
				                              mx[3]+rimP));
				
				for(int ip = -rimP; ip <= mx[3]+rimP; ++ip) {
					for(int iz = -1; iz <= mx[2]+1; iz++) {
						for(int iy = -1; iy <= mx[1]+1; iy++) {
							for(int ix = 1; ix<=rim; ix++) {
								SendBuff4D(ix,iy,iz,ip) = data(ix,iy,iz,ip);
							}
						}
					}
				}
			}
			into = MyMPI.get_left();

		} else if(dir == 1) { // y-direction

			if(!extended) {

				SendBuff3D.resize(Index::set(     -rim,      -rim,      -rimP),
				                  Index::set(mx[0]+rim, mx[2]+rim, mx[3]+rimP));
				
				for(int ip = -rimP; ip <= mx[3]+rimP; ++ip) {
					for(int iz = -1; iz <= mx[2]+1; iz++) {
						for(int ix = -1; ix <= mx[0]+1; ix++) {
							SendBuff3D(ix,iz,ip) = data(ix,1,iz,ip);
						}
					}
				}

			} else {
				
				SendBuff4D.resize(Index::set(     -rim,   1,      -rim,
				                                -rimP),
				                  Index::set(mx[0]+rim, rim, mx[2]+rim,
				                             mx[3]+rimP));
				
				for(int ip = -rimP; ip <= mx[3]+rimP; ++ip) {
					for(int iz = -1; iz <= mx[2]+1; iz++) {
						for(int iy = 1; iy<=rim; iy++) {
							for(int ix = -1; ix <= mx[0]+1; ix++) {
								SendBuff4D(ix,iy,iz,ip) = data(ix,iy,iz,ip);
							}
						}
					}
				}
			}
			into = MyMPI.get_front();

		} else { // z-direction

			if(!extended) {
				
				SendBuff3D.resize(Index::set(     -rim,      -rim,      -rimP),
				                  Index::set(mx[0]+rim, mx[1]+rim, mx[3]+rimP));
				
				for(int ip = -rimP; ip <= mx[3]+rimP; ++ip) {
					for(int iy = -1; iy <= mx[1]+1; iy++) {
						for(int ix = -1; ix <= mx[0]+1; ix++) {
							SendBuff3D(ix,iy,ip) = data(ix,iy,1,ip);
						}
					}
				}

			} else {

				SendBuff4D.resize(Index::set(     -rim,      -rim,   1,
				                                  -rimP),
				                  Index::set(mx[0]+rim, mx[1]+rim, rim,
				                             mx[3]+rimP));
				
				for(int ip = -rimP; ip <= mx[3]+rimP; ++ip) {
					for(int iz = 1; iz<=rim; iz++) {
						for(int iy = -1; iy <= mx[1]+1; iy++) {
							for(int ix = -1; ix <= mx[0]+1; ix++) {
								SendBuff4D(ix,iy,iz,ip) = data(ix,iy,iz,ip);
							}
						}
					}
				}
			}
			into = MyMPI.get_bottom();

		}

	} 

	if(RecvRight) {
		if(dir==0) { // x-direction

			if(!extended) {
				RecvBuff3D.resize(Index::set(     -rim,      -rim,      -rimP),
				                  Index::set(mx[1]+rim, mx[2]+rim, mx[3]+rimP));
			} else {
				RecvBuff4D.resize(Index::set(  1,      -rim,      -rim,
				                               -rimP),
				                  Index::set(rim, mx[1]+rim, mx[2]+rim,
				                             mx[3]+rimP));
			}
			from = MyMPI.get_right();

		} else if (dir==1) { // y-direction

			if(!extended) {
				RecvBuff3D.resize(Index::set(     -rim,      -rim,      -rimP),
				                  Index::set(mx[0]+rim, mx[2]+rim, mx[3]+rimP));
			} else {
				RecvBuff4D.resize(Index::set(     -rim,   1,      -rim,
				                               -rimP),
				                  Index::set(mx[0]+rim, rim, mx[2]+rim,
				                             mx[3]+rimP));
			}
			from = MyMPI.get_back();

		} else { // z-direction

			if(!extended) {
				RecvBuff3D.resize(Index::set(     -rim,      -rim,      -rimP),
				                  Index::set(mx[0]+rim, mx[1]+rim, mx[3]+rimP));
			} else {
				RecvBuff4D.resize(Index::set(     -rim,      -rim,   1,
				                               -rimP),
				                  Index::set(mx[0]+rim, mx[1]+rim, rim,
				                             mx[3]+rimP));
			}
			from = MyMPI.get_top();

		}

	}

	if(SendLeft || RecvRight) { // Periodic

		// Set correct size:
		if(dir==0) {
			size = sizex;
		} else if (dir==1) {
			size = sizey;
		} else {
			size = sizez;
		}

		if(!extended) {
			do_MpiSendRecv(SendBuff3D, RecvBuff3D, from, into, size,
			               SendLeft, RecvRight);
		} else {
			do_MpiSendRecv(SendBuff4D, RecvBuff4D, from, into, size,
			               SendLeft, RecvRight);
		}
	}
	

	// Now assign the data if necessary
	if(RecvRight) {

		if(dir == 0) { // x-direction

			if(!extended) {
				for(int ip = -rimP; ip <= mx[3]+rimP; ++ip) {
					for(int iz = -1; iz <= mx[2]+1; iz++) {
						for(int iy = -1; iy <= mx[1]+1; iy++) {
							data(mx[0]+1,iy,iz,ip) = RecvBuff3D(iy,iz,ip);
						}
					}
				}
			} else {
				for(int ip = -rimP; ip <= mx[3]+rimP; ++ip) {
					for(int iz = -1; iz <= mx[2]+1; iz++) {
						for(int iy = -1; iy <= mx[1]+1; iy++) {
							for(int ix = 1; ix<=rim; ix++) {
								data(mx[0]+ix,iy,iz,ip) = RecvBuff4D(ix,iy,iz,ip);
							}
						}
					}
				}
			}

		} else if (dir==1) { // y-direction

			if(!extended) {
				for(int ip = -rimP; ip <= mx[3]+rimP; ++ip) {
					for(int iz = -1; iz <= mx[2]+1; iz++) {
						for(int ix = -1; ix <= mx[0]+1; ix++) {
							data(ix, mx[1]+1,iz,ip) = RecvBuff3D(ix,iz,ip);
						}
					}
				}
			} else {
				for(int ip = -rimP; ip <= mx[3]+rimP; ++ip) {
					for(int iz = -1; iz <= mx[2]+1; iz++) {
						for(int iy = 1; iy<=rim; iy++) {
							for(int ix = -1; ix <= mx[0]+1; ix++) {
								data(ix, mx[1]+iy,iz,ip) = RecvBuff4D(ix,iy,iz,ip);
							}
						}
					}
				}
			}

		} else { // z-direction

			if(!extended) {
				for(int ip = -rimP; ip <= mx[3]+rimP; ++ip) {
					for(int iy = -1; iy <= mx[1]+1; iy++) {
						for(int ix = -1; ix <= mx[0]+1; ix++) {
							data(ix, iy, mx[2]+1,ip) = RecvBuff3D(ix,iy,ip);
						}
					}
				}
			} else {
				for(int ip = -rimP; ip <= mx[3]+rimP; ++ip) {
					for(int iz = 1; iz<=rim; iz++) {
						for(int iy = -1; iy <= mx[1]+1; iy++) {
							for(int ix = -1; ix <= mx[0]+1; ix++) {
								data(ix, iy, mx[2]+iz,ip) = RecvBuff4D(ix,iy,iz,ip);
							}
						}
					}
				}
			}

		}
				
	}




	// Second part: transfer from right to left
	if(SendRight) {

		if(dir == 0) { // x-direction

			if(!extended) {

				SendBuff3D.resize(Index::set(     -rim,      -rim,      -rimP),
				                  Index::set(mx[1]+rim, mx[2]+rim, mx[3]+rimP));

				for(int ip = -rimP; ip <= mx[3]+rimP; ++ip) {
					for(int iz = -1; iz <= mx[2]+1; iz++) {
						for(int iy = -1; iy <= mx[1]+1; iy++) {
							SendBuff3D(iy,iz,ip) = data(mx[0]-1,iy,iz,ip);
						}
					}
				}

			} else {

				SendBuff4D.resize(Index::set(  1,      -rim,      -rim,
				                               -rimP),
				                  Index::set(rim, mx[1]+rim, mx[2]+rim,
				                             mx[3]+rimP));

				for(int ip = -rimP; ip <= mx[3]+rimP; ++ip) {
					for(int iz = -1; iz <= mx[2]+1; iz++) {
						for(int iy = -1; iy <= mx[1]+1; iy++) {
							for(int ix = 1; ix<=rim; ix++) {
								SendBuff4D(ix,iy,iz,ip) = data(mx[0]-ix,iy,iz,ip);
							}
						}
					}
				}
				
			}
			into = MyMPI.get_right();

		} else if (dir == 1) { // y-direction

			if(!extended) {
				
				SendBuff3D.resize(Index::set(     -rim,      -rim,      -rimP),
				                  Index::set(mx[0]+rim, mx[2]+rim, mx[3]+rimP));

				for(int ip = -rimP; ip <= mx[3]+rimP; ++ip) {
					for(int iz = -1; iz <= mx[2]+1; iz++) {
						for(int ix = -1; ix <= mx[0]+1; ix++) {
							SendBuff3D(ix,iz,ip) = data(ix,mx[1]-1,iz,ip);
						}
					}
				}

			} else {

				SendBuff4D.resize(Index::set(     -rim,   1,      -rim,
				                                  -rimP),
				                  Index::set(mx[0]+rim, rim, mx[2]+rim,
				                             mx[3]+rimP));

				for(int ip = -rimP; ip <= mx[3]+rimP; ++ip) {
					for(int iz = -1; iz <= mx[2]+1; iz++) {
						for(int iy = 1; iy<=rim; iy++) {
							for(int ix = -1; ix <= mx[0]+1; ix++) {
								SendBuff4D(ix,iy,iz,ip) = data(ix,mx[1]-iy,iz,ip);
							}
						}
					}
				}

			}
			into = MyMPI.get_back();

		} else { // z-direction

			if(!extended) {
				
				SendBuff3D.resize(Index::set(     -rim,      -rim,      -rimP),
				                  Index::set(mx[0]+rim, mx[1]+rim, mx[3]+rimP));

				for(int ip = -rimP; ip <= mx[3]+rimP; ++ip) {
					for(int iy = -1; iy <= mx[1]+1; iy++) {
						for(int ix = -1; ix <= mx[0]+1; ix++) {
							SendBuff3D(ix,iy,ip) = data(ix,iy,mx[2]-1,ip);
						}
					}
				}

			} else {

				SendBuff4D.resize(Index::set(     -rim,      -rim,   1,
				                                  -rimP),
				                  Index::set(mx[0]+rim, mx[1]+rim, rim,
				                             mx[3]+rimP));

				for(int ip = -rimP; ip <= mx[3]+rimP; ++ip) {
					for(int iz = 1; iz<=rim; iz++) {
						for(int iy = -1; iy <= mx[1]+1; iy++) {
							for(int ix = -1; ix <= mx[0]+1; ix++) {
								SendBuff4D(ix,iy,iz,ip) = data(ix,iy,mx[2]-iz,ip);
							}
						}
					}
				}
				
			}
			into = MyMPI.get_top();

		}

	}

	if(RecvLeft) {

		if(dir==0) { // x-direction

			if(!extended) {

				RecvBuff3D.resize(Index::set(     -rim,      -rim,      -rimP),
				                  Index::set(mx[1]+rim, mx[2]+rim, mx[3]+rimP));

			} else {

				RecvBuff4D.resize(Index::set(  1,      -rim,      -rim,
				                               -rimP),
				                  Index::set(rim, mx[1]+rim, mx[2]+rim,
				                             mx[3]+rimP));

			}
			from = MyMPI.get_left();

		} else if (dir==1) { // y-direction

			if(!extended) {

				RecvBuff3D.resize(Index::set(     -rim,      -rim,      -rimP),
				                  Index::set(mx[0]+rim, mx[2]+rim, mx[3]+rimP));

			} else {

				RecvBuff4D.resize(Index::set(     -rim,   1,      -rim,
				                                  -rimP),
				                  Index::set(mx[0]+rim, rim, mx[2]+rim,
				                             mx[3]+rimP));

			}
			from = MyMPI.get_front();

		} else { // z-direction

			if(!extended) {

				RecvBuff3D.resize(Index::set(     -rim,      -rim,      -rimP),
				                  Index::set(mx[0]+rim, mx[1]+rim, mx[3]+rimP));

			} else {

				RecvBuff4D.resize(Index::set(     -rim,      -rim,   1,
				                                  -rimP),
				                  Index::set(mx[0]+rim, mx[1]+rim, rim,
				                             mx[3]+rimP));

			}
			from = MyMPI.get_bottom();

		}
		
	}

	// Do the actual communication
	if(SendRight || RecvLeft) {

		// Set correct size:
		if(dir==0) {
			size = sizex;
		} else if (dir==1) {
			size = sizey;
		} else {
			size = sizez;
		}

		if(!extended) {
			do_MpiSendRecv(SendBuff3D, RecvBuff3D, from, into, size,
			               SendRight, RecvLeft);
		} else {
			do_MpiSendRecv(SendBuff4D, RecvBuff4D, from, into, size,
			               SendRight, RecvLeft);
		}
	}

	// Now assign the data if necessary:
	if(RecvLeft) {

		if(dir==0) { // x-direction

			if(!extended) {
				for(int ip = -rimP; ip <= mx[3]+rimP; ++ip) {
					for(int iz = -1; iz <= mx[2]+1; iz++) {
						for(int iy = -1; iy <= mx[1]+1; iy++) {
							data(-1,iy,iz,ip) = RecvBuff3D(iy,iz,ip);
						}
					}
				}
			} else {
				for(int ip = -rimP; ip <= mx[3]+rimP; ++ip) {
					for(int iz = -1; iz <= mx[2]+1; iz++) {
						for(int iy = -1; iy <= mx[1]+1; iy++) {
							for(int ix = 1; ix<=rim; ix++) {
								data(-ix,iy,iz,ip) = RecvBuff4D(ix,iy,iz,ip);
							}
						}
					}
				}
			}

		} else if(dir==1) { // y-direction

			if(!extended) {
				for(int ip = -rimP; ip <= mx[3]+rimP; ++ip) {
					for(int iz = -1; iz <= mx[2]+1; iz++) {
						for(int ix = -1; ix <= mx[0]+1; ix++) {
							data(ix, -1,iz,ip) = RecvBuff3D(ix,iz,ip);
						}
					}
				}
			} else {
				for(int ip = -rimP; ip <= mx[3]+rimP; ++ip) {
					for(int iz = -1; iz <= mx[2]+1; iz++) {
						for(int iy = 1; iy<=rim; iy++) {
							for(int ix = -1; ix <= mx[0]+1; ix++) {
								data(ix,-iy,iz,ip) = RecvBuff4D(ix,iy,iz,ip);
							}
						}
					}
				}
			}

		} else { // z-direction

			if(!extended) {
				for(int ip = -rimP; ip <= mx[3]+rimP; ++ip) {
					for(int iy = -1; iy <= mx[1]+1; iy++) {
						for(int ix = -1; ix <= mx[0]+1; ix++) {
							data(ix, iy, -1,ip) = RecvBuff3D(ix,iy,ip);
						}
					}
				}
			} else {
				for(int ip = -rimP; ip <= mx[3]+rimP; ++ip) {
					for(int iz = 1; iz<=rim; iz++) {
						for(int iy = -1; iy <= mx[1]+1; iy++) {
							for(int ix = -1; ix <= mx[0]+1; ix++) {
								data(ix, iy, -iz,ip) = RecvBuff4D(ix,iy,iz,ip);
							}
						}
					}
				}
			}

		}
	}

}


void BoundaryHandler4D::do_MpiSendRecv(NumMatrix<double,4> &Send,
                                       NumMatrix<double,4> &Recv,
                                       int from, int into, int size,
                                       bool do_Send, bool do_Receive)
{

	int numRequests = 0;
	if(do_Send) numRequests++;
	if(do_Receive) numRequests++;


	// Transfer of data:
	// initialize
	MPI_Request requests[numRequests];
	MPI_Status statusrl[numRequests];
	for(int ireq=0; ireq<numRequests; ireq++) {
		requests[ireq] = MPI_REQUEST_NULL;
	}
    
	// receive data
	// message tag -- must not be less than 0!
	int ireq(0);
	if(do_Receive) {
		int tag = from; 

		MPI_Irecv((double *)Recv, size, MPI_DOUBLE, from , tag,
		          MyMPI.comm3d, &requests[ireq]);
		ireq++;
	}
	
	if(do_Send) {
		int tag = MyMPI.get_rank();
    
		MPI_Isend((double *)Send, size, MPI_DOUBLE, into, tag,
		          MyMPI.comm3d, &requests[ireq]);
	}


	/* wait for all communication to complete */
	
	if(numRequests > 0) {
		MPI_Waitall(numRequests, requests, statusrl);
	}
    
}


void BoundaryHandler4D::do_MpiSendRecv(NumMatrix<double,3> &Send,
                                       NumMatrix<double,3> &Recv,
                                       int from, int into, int size,
                                       bool do_Send, bool do_Receive)
{

	int numRequests = 0;
	if(do_Send) numRequests++;
	if(do_Receive) numRequests++;


	// Transfer of data:
	// initialize
	MPI_Request requests[numRequests];
	MPI_Status statusrl[numRequests];
	for(int ireq=0; ireq<numRequests; ireq++) {
		requests[ireq] = MPI_REQUEST_NULL;
	}
    
	// receive data
	// message tag -- must not be less than 0!
	int ireq(0);
	if(do_Receive) {
		int tag = from; 

		MPI_Irecv((double *)Recv, size, MPI_DOUBLE, from , tag,
		          MyMPI.comm3d, &requests[ireq]);
		ireq++;
	}
	
	if(do_Send) {
		int tag = MyMPI.get_rank();
    
		MPI_Isend((double *)Send, size, MPI_DOUBLE, into, tag,
		          MyMPI.comm3d, &requests[ireq]);
	}


	/* wait for all communication to complete */
	
	if(numRequests > 0) {
		MPI_Waitall(numRequests, requests, statusrl);
	}
    
}


#endif


