
/* =============================================================================
 *
 * Title:         BiCGStab 3d OpenCL accelerated
 * Author:        Felix Niederwanger
 * Description:   Example and test program for the BiCGStab 3D linear solver
 *                using OpenCL acceleration
 * =============================================================================
 */



#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cmath>
#include <unistd.h>
#include <signal.h>
#include "LinSolver3D.hpp"
#include "BiCGStabCL.hpp"
#include "FlexCL.hpp"
#include "FlexCLMatrix.hpp"
#include "time_ms.h"
#include "main.hpp"

using namespace std;
using namespace flexCL;

// XXX: Remove me!
#define TEST_CASE 0

// Default size
#define SIZE 32

#if 0
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
#endif

/* ==== Program configuration ==== */
static size_t size = SIZE;
static bool verbose = false;
static long runtime = -time_ms();
// Desired OpenCL context
static int opencl_context = OCL_CONTEXT_DEFAULT;

static int testSwitch = TEST_ONE;
/* Tolerance for the computation */
static double tolerance = 1e-6;

/* ==== Function prototypes ================================================ */

#define VERBOSE(x) if(verbose) cout << x << endl;
#define DELETE(x) { if(x!=NULL) delete x; x = NULL; }

static void printHelp(string programName = "bicgstab_cl");
static void sig_handler(int);

/** Check given matrix for illegal values */
static bool checkMatrix(NumMatrix<double,3>&);

/* ==== MAIN PROGRAM FUNCTION ============================================== */

int main(int argc, char** argv) {
    cout << " -- BiCGStab 3D Linear solver (OpenCL) --" << endl;
    signal(SIGINT, sig_handler);
    signal(SIGUSR1, sig_handler);
    //signal(SIGSEGV, sig_handler);

	/* ====== Problem variable declaration ====== */
	Linsolver3D *solver = NULL;

	/* ==== Handle program parameter ==== */
	for(int i=1;i<argc;i++) {
		const string arg = string(argv[i]);
		const bool isLast = i >= argc-1;
		try {
			if(arg == "-h" || arg == "--help") {
				printHelp(string(argv[0]));
				return EXIT_SUCCESS;
			} else if(arg == "-v" || arg == "--verbose") {
				verbose = true;
			} else if(arg == "--gpu")
				opencl_context = OCL_CONTEXT_GPU;
			else if(arg == "--cpu")
				opencl_context = OCL_CONTEXT_CPU;
			else if(arg == "-n" || arg == "--np") {
				if(isLast) throw "Missing argument: Problem size";
				size = atoi(argv[++i]);
			} else if(arg == "-t" || arg == "--test") {
				if(isLast) throw "Missing argument: Test case";
				testSwitch = atoi(argv[++i]);
			} else
				throw "Illegal argument";
		} catch(const char* msg) {
			cerr << msg << endl;
			return EXIT_FAILURE;
		}
	}

	/* ==== OpenCL Initialisation ========================================== */
    VERBOSE("  OpenCL initialisation ... ");
    OpenCL openCl;
	Context *oclContext = NULL;
    try {
		switch(opencl_context) {
		case OCL_CONTEXT_CPU:
			oclContext = openCl.createCPUContext();
			break;
		case OCL_CONTEXT_GPU:
			oclContext = openCl.createGPUContext();
			break;
		default:
			oclContext = openCl.createContext();
			break;
		}
		if(oclContext == NULL) throw OpenCLException("No OpenCL device");

		DeviceInfo oclDevice = oclContext->device_info();
		cout << "OpenCL context initialized on device " <<oclDevice.device_id() << ": " << oclDevice.vendor() << " " << oclDevice.name() << endl;

    } catch (OpenCLException &e) {
    	cerr << "Error setting up OpenCL context: " << e.what() << endl;
    	return EXIT_FAILURE;
    }


    /* ==== Problem initialisation ========================================= */
	static size_t mx[3] = {size, size, size };
	static size_t Nx_global[3] { size+1, size+1, size+1 };


    VERBOSE("  Problem initialization (" << mx[0] << "x" << mx[1] << "x" << mx[2] << ") - Test switch " << testSwitch << " ... ");
	grid_manager grid(0., 0., 0., 1., 1., 1., Nx_global[0], Nx_global[1], Nx_global[2], 1);
	BoundaryHandler3D boundaries;
	// Set all boundary conditions to Dirichlet condition
	for(int bound=0; bound<6; ++bound) grid.set_bcType(bound, BOUNDARY_DIRICHLET);
	boundaries.set_bcType(BOUNDARY_DIRICHLET);
	// Initialize matrices
	NumMatrix<double,3> phi;				// Numerical solution phi
	NumMatrix<double,3> phi_exact;			// Analytical solution
	NumMatrix<double,3> rhs;
	NumMatrix<double,3> lambda;
	NumMatrix<double,3> diffTens[4];
	NumArray<double> diff(3);
	// 1 ghost cell in each direction
	phi.resize(Index::set(-1,-1,-1), Index::set(mx[0]+1, mx[1]+1, mx[2]+1));
	phi_exact.resize(Index::set(-1,-1,-1), Index::set(mx[0]+1, mx[1]+1, mx[2]+1));
	rhs.resize(Index::set(-1,-1,-1), Index::set(mx[0]+1, mx[1]+1, mx[2]+1));
	lambda.resize(Index::set(-1,-1,-1), Index::set(mx[0]+1, mx[1]+1, mx[2]+1));
	diffTens[0].resize(Index::set(-1,-1,-1), Index::set(mx[0]+1, mx[1]+1, mx[2]+1));
	diffTens[1].resize(Index::set(-1,-1,-1), Index::set(mx[0]+1, mx[1]+1, mx[2]+1));
	diffTens[2].resize(Index::set(-1,-1,-1), Index::set(mx[0]+1, mx[1]+1, mx[2]+1));
	diffTens[3].resize(Index::set(-1,-1,-1), Index::set(mx[0]+1, mx[1]+1, mx[2]+1));



	const double dPar = 1.0;
	const double dPerp = 0.1;

	// Setting grid values
	try {
		diffTens[0].set_constVal(1.0);
		diffTens[1].set_constVal(1.0);
		diffTens[2].set_constVal(1.0);
		diffTens[3].clear();
		diff(0) = 1.0;
		diff(1) = 1.0;
		diff(2) = 100.0;
		diff(2) = 10.0;
		phi_exact.clear();
		lambda.clear();
		phi.clear();
		rhs.clear();

		const double pi = M_PI;
		for(size_t iz = 0; iz <= mx[2]; ++iz) {
			const double zVal = grid.get_Pos(2,iz);
			for(size_t iy = 0; iy <= mx[1]; ++iy) {
				const double yVal = grid.get_Pos(1,iy);
				for(size_t ix = 0; ix <= mx[0]; ++ix) {
					const double xVal = grid.get_Pos(0,ix);

					// cout << "[" << ix << "," << iy << "," << iz << "] = (" << xVal << "," << yVal << "," << zVal << ")" << endl;

					phi_exact(ix,iy,iz) = sin(M_PI * xVal)*sin(M_PI * yVal)*sin(M_PI * zVal);
					lambda(ix,iy,iz) = 0.2*xVal*sqr(yVal)*zVal;



					switch(testSwitch) {
					case TEST_ONE:
						rhs(ix,iy,iz) = -(sqr(M_PI)*(diff(0) + diff(1) + diff(2)) + lambda(ix,iy,iz))*phi_exact(ix,iy,iz);
						break;
					case TEST_TWO:		// Test 2 (räumliche Diffusion)
					{
						phi_exact(ix,iy,iz) = sin(pi*xVal)*sin(pi*yVal)*sin(pi*zVal);

						diffTens[0](ix,iy,iz) = yVal;
						diffTens[1](ix,iy,iz) = xVal;
						diffTens[2](ix,iy,iz) = zVal;
						rhs(ix,iy,iz) = -(sqr(pi)*(xVal + yVal + zVal) + lambda(ix,iy,iz))*phi_exact(ix,iy,iz) + pi*sin(pi*xVal)*sin(pi*yVal)*cos(pi*zVal);
					}
						break;
					case TEST_THREE:	// Test 3 (räumliche Diffusion mit D_xy)
					{
						double AVal = 0.1;//1.8;
						diffTens[0](ix,iy,iz) = yVal;
						diffTens[1](ix,iy,iz) = xVal;
						diffTens[2](ix,iy,iz) = zVal;
						// DiffTens[3](ix,iy,iz) = AVal*sqr(xVal)*yVal*zVal;
						diffTens[3](ix,iy,iz) = AVal*sqr(xVal)*yVal*zVal;
						const double D_xy = diffTens[3](ix,iy,iz);
						rhs(ix,iy,iz) = -(sqr(pi)*(xVal + yVal + zVal) + lambda(ix,iy,iz))*phi_exact(ix,iy,iz) +
								pi*sin(pi*xVal)*sin(pi*yVal)*cos(pi*zVal) +
								2.*D_xy*sqr(pi)*cos(pi*xVal)*cos(pi*yVal)*sin(pi*zVal)+
								2.*AVal*xVal*yVal*zVal*pi*sin(pi*xVal)*cos(pi*yVal)*sin(pi*zVal) +
								AVal*sqr(xVal)*zVal*pi*cos(pi*xVal)*sin(pi*yVal)*sin(pi*zVal);
					}
						break;
					case TEST_FOUR:
					{
						phi_exact(ix,iy,iz) = sin(pi*xVal)*sin(pi*yVal)*sin(pi*zVal);
						const double angle = atan2(yVal, xVal);
						diffTens[0](ix,iy,iz) = (dPar*sqr(sin(angle)) + dPerp*sqr(cos(angle)));
						diffTens[1](ix,iy,iz) = (dPar*sqr(cos(angle)) + dPerp*sqr(sin(angle)));
						diffTens[2](ix,iy,iz) = dPerp;
						diffTens[3](ix,iy,iz) = (dPerp - dPar)*sin(angle)*cos(angle);

						const double Dxx = diffTens[0](ix,iy,iz);
						const double Dyy = diffTens[1](ix,iy,iz);
						const double Dzz = diffTens[2](ix,iy,iz);
						const double Dxy = diffTens[3](ix,iy,iz);
						const double sqrRad = sqr(xVal) + sqr(yVal);
						const double dphidx = -yVal/sqrRad;
						const double dphidy = xVal/sqrRad;
						const double dDxxDx = 2.*(dPar - dPerp)*sin(angle)*cos(angle)*dphidx;
						const double dDyyDy = 2.*(dPerp - dPar)*sin(angle)*cos(angle)*dphidy;
						const double dDxyDx = (dPerp - dPar)*(sqr(cos(angle)) - sqr(sin(angle)))*dphidx;
						const double dDxyDy = (dPerp - dPar)*(sqr(cos(angle)) - sqr(sin(angle)))*dphidy;

						rhs(ix,iy,iz) =
								(dDyyDy + dDxyDx)*pi*sin(pi*xVal)*cos(pi*yVal)*sin(pi*zVal) +
								(dDxxDx + dDxyDy)*pi*cos(pi*xVal)*sin(pi*yVal)*sin(pi*zVal) +
								2.0*Dxy*sqr(pi)*cos(pi*xVal)*cos(pi*yVal)*sin(pi*zVal) -
								((Dxx + Dyy + Dzz)*sqr(pi) + lambda(ix,iy,iz))*phi_exact(ix,iy,iz);
					}
					break;

					default:
						throw "Illegal test case";
					}

				}
			}
		}

	} catch(const char* msg) {
		cerr << "Error setting up grid values: " << msg << endl;
		return EXIT_FAILURE;
	}

	/* Pre-run test to check if any illegal values are in the matrices */
	try {
		if(!checkMatrix(phi)) throw "Matrix phi";
		if(!checkMatrix(phi_exact)) throw "Matrix phi_exact";
		if(!checkMatrix(diffTens[0])) throw "Matrix diffTens[0]";
		if(!checkMatrix(diffTens[1])) throw "Matrix diffTens[1]";
		if(!checkMatrix(diffTens[2])) throw "Matrix diffTens[2]";
		if(!checkMatrix(diffTens[3])) throw "Matrix diffTens[3]";
		if(!checkMatrix(lambda)) throw "Matrix lambda";
		if(!checkMatrix(rhs)) throw "Matrix rhs";

		cout << "  Pre-run checks completed." << endl;

	} catch (const char* msg) {
		cerr << "Pre-run check of the matrices failed: " << msg << endl;
		exit(5);
	}

	/* ==== Initialisation of the linear solver ============================ */
	VERBOSE("  Setting up solver ... ");
	BiCGStabSolver *bicgsolver = NULL;
	try {
		bicgsolver = new BiCGStabSolver(grid, tolerance, 2, oclContext);
		bicgsolver->setVerbose(verbose);
		bicgsolver->setupContext();
		solver = bicgsolver;
	} catch (CompileException &e) {
		// OpenCL compilation error
		cerr << "BiCGStab kernel compilation error: " << e.what() << endl;
		cerr << endl << e.compile_output() << endl;
		return EXIT_FAILURE;
	} catch (OpenCLException &e) {
		cerr << "OpenCL exception: " << e.what() << endl;
		return EXIT_FAILURE;
	}



	/* ==== Solver ========================================================= */
	VERBOSE("Problem setup complete.");
	long calc_runtime = -time_ms();
	cout << endl << "=======================================================" << endl;
	cout << "  Running calculation ... " << endl;

	try {

		if(testSwitch == TEST_ONE) {
			solver->solve(boundaries, phi, rhs, lambda, diff[0], diff[1], diff[2], 8);
		} else if (testSwitch == TEST_TWO) {
			solver->solve(boundaries, phi, rhs, lambda, diffTens[0], diffTens[1], diffTens[2], diffTens[3], 8);
		} else if(testSwitch == TEST_THREE || testSwitch == TEST_FOUR) {
			solver->solve(boundaries, phi, rhs, lambda, diffTens[0], diffTens[1], diffTens[2], diffTens[3], 8, true);
		} else {
			cerr << "UNKNOWN TEST SWITCH: " << testSwitch << endl;
			return EXIT_FAILURE;
		}


		calc_runtime += time_ms();
		cout << "  Computation complete. Calculation time: " << calc_runtime << " ms" << endl;

		/* ==== Error estimate ================================================= */
		//const double coeff[3] = {1.0/sqr(dx),1.0/sqr(dy),1.0/sqr(dz)};
		//const double coeff_xy = 1./(2.*dx*dy);
		double error(0.), num(0.);
		for(size_t iz = 0; iz <= mx[2]; ++iz) {
			for(size_t iy = 0; iy <= mx[1]; ++iy) {
				for(size_t ix = 0; ix <= mx[0]; ++ix) {
					error += sqr((phi_exact(ix,iy,iz)) - phi(ix,iy,iz));
					num += 1.0;
				}
			}
		}
		double l2err = sqrt(error/num);
		cout << " l2 error: " << l2err << "  (" << error << "/" << num << ")" << endl;

		// Detailed look at solution
		// Now let's have a look:
		if(verbose) {
			cout << "Detailed look at solution: " << endl;
			for(size_t iz = 0; iz <= mx[2]; ++iz) {
				for(size_t iy = 0; iy <= mx[1]; ++iy) {
					for(size_t ix = 0; ix <= mx[0]; ++ix) {
						// Just print data in center
						if(ix==mx[0]/2 && iz==mx[2]/2) {
							cout << "\t at y = " << iy << ": (" << phi_exact(ix,iy,iz) << " - " << phi(ix,iy,iz) << ") = ";
							cout << phi_exact(ix,iy,iz)-phi(ix,iy,iz) << endl;
						}
					}
				}
			}
		}

		// Search for maximum error
		double max_error = -1.0;

		for(size_t iz = 0; iz <= mx[2]; ++iz) {
			for(size_t iy = 0; iy <= mx[1]; ++iy) {
				for(size_t ix = 0; ix <= mx[0]; ++ix) {
					const double c_error = fabs(phi_exact(ix,iy,iz)-phi(ix,iy,iz));
					if(max_error <= 0.0) max_error = c_error;
					else max_error = fmax(max_error, c_error);
				}
			}
		}

		cout << "Maximum error on total grid: " << max_error << endl;

	} catch (OpenCLException &e) {
		cerr << "OpenCL exception thrown: " << e.what() << " - " << e.opencl_error_string() << endl;
		DELETE(solver);
		DELETE(oclContext);
		return EXIT_FAILURE;
	} catch (NumException &e) {
		cerr << "Numerical exception occurred: " << e.what() << endl;
		DELETE(solver);
		DELETE(oclContext);
		return EXIT_FAILURE;
	} catch (const char *msg) {
		cerr << "Exception thrown: " << msg << endl;
		DELETE(solver);
		DELETE(oclContext);
		return EXIT_FAILURE;
	} catch (...) {
		cerr << "Unknown exception thrown." << endl;
		DELETE(solver);
		DELETE(oclContext);
		return EXIT_FAILURE;
	}
	cout << "=======================================================" << endl << endl;

	/* ==== CLEANUP ======================================================== */
	long iterations = bicgsolver->iterations();
	VERBOSE("Cleanup ... ");
	DELETE(solver);
	DELETE(oclContext);

	// Goodbye message
	runtime += time_ms();
	cout << "Total runtime: " << runtime << " ms (" << iterations << " iterations)" << endl;
	if (verbose) {
		double avg_runtime = (double)runtime / (double)iterations;
		cout << "\tAverage: " << avg_runtime << " ms/iterations" << endl;
	}
	cout << "Bye" << endl;
    return EXIT_SUCCESS;
}





static void printHelp(string programName) {
    cout << "  SYNOPSIS" << endl;
    cout << "    " << programName << " [OPTIONS]" << endl << endl;
    cout << "  OPTIONS" << endl;
    cout << "    -h   --help                   Show program help" << endl;
    cout << "    -v   --verbose                Verbose mode" << endl;
    cout << "    -n N --np N                   Set problem size to N" << endl;
    cout << "         --cpu                    Use CPU context (OpenCL)" << endl;
    cout << "         --gpu                    Use GPU context (OpenCL)" << endl;
    cout << "    -t --test TEST                Set test case to TEST" << endl;
}


static void sig_handler(int sig_no) {
	switch(sig_no) {
	case SIGINT:
	case SIGUSR1:
		cerr << "User termination request." << endl;
		exit(EXIT_FAILURE);
		return;
	case SIGUSR2:
		cout << "User signal received." << endl;
		// Do nothing. Placeholder for easter egg
		break;
	case SIGSEGV:
		cerr << "Segmentation fault, Terminating program NOW!" << endl;
		exit(EXIT_FAILURE);
		return;
	}
}


static bool checkMatrix(NumMatrix<double,3> &matrix) {
	// Matrix range
	int mx[3][2];
	for(int i=0;i<3;i++) {
		mx[i][0] = matrix.getLow(i);
		mx[i][1] = matrix.getHigh(i);
	}

	for(int iz = mx[2][0]; iz <= mx[2][1]; ++iz) {
		for(int iy = mx[1][0]; iy <= mx[1][1]; ++iy) {
			for(int ix = mx[0][0]; ix <= mx[0][1]; ++ix) {
				const double value = matrix(ix,iy,iz);

				if (::isnan(value) || ::isinf(value)) return false;
			}
		}
	}

	return true;
}
