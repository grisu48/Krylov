
/* =============================================================================
 * 
 * Title:         BiCGStab 3d OpenCL accelerated
 * Author:        Felix Niederwanger
 * Description:   Example and test program for the BiCGStab 3D linear solver
 *                using OpenCL acceleration
 * =============================================================================
 */
 
 
 
#include <iostream>
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


// Default size
#define SIZE 32


/* ==== Program configuration ==== */
static size_t size = SIZE;
static bool verbose = false;
static long runtime = -time_ms();
// Desired OpenCL context
static int opencl_context = OCL_CONTEXT_DEFAULT;

static int testSwitch = TEST_ONE;
/* Tolerance for the computation */
static double tolerance = 1e-9;

/* ==== Function prototypes ================================================ */

#define VERBOSE(x) if(verbose) cout << x << endl;
#define DELETE(x) { if(x!=NULL) delete x; x = NULL; }

static void printHelp(string programName = "bicgstab_cl");
static void sig_handler(int);


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
			else if(arg == "-n") {
				if(isLast) throw "Missing argument: Problem size";
				size = atoi(argv[++i]);
			}
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


    VERBOSE("  Problem initialization ... (" << mx[0] << "x" << mx[1] << "x" << mx[2] << ")");
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
	phi.resize(Index::set(-1,-1,-1), Index::set(mx[0]+1, mx[1]+1, mx[2]+1));
	phi_exact.resize(Index::set(-1,-1,-1), Index::set(mx[0]+1, mx[1]+1, mx[2]+1));
	rhs.resize(Index::set(-1,-1,-1), Index::set(mx[0]+1, mx[1]+1, mx[2]+1));
	lambda.resize(Index::set(-1,-1,-1), Index::set(mx[0]+1, mx[1]+1, mx[2]+1));
	diffTens[0].resize(Index::set(-1,-1,-1), Index::set(mx[0]+1, mx[1]+1, mx[2]+1));
	diffTens[1].resize(Index::set(-1,-1,-1), Index::set(mx[0]+1, mx[1]+1, mx[2]+1));
	diffTens[2].resize(Index::set(-1,-1,-1), Index::set(mx[0]+1, mx[1]+1, mx[2]+1));
	diffTens[3].resize(Index::set(-1,-1,-1), Index::set(mx[0]+1, mx[1]+1, mx[2]+1));
	//const double dx = grid.get_delx(0);
	//const double dy = grid.get_delx(1);
	//const double dz = grid.get_delx(2);

	// Setting grid values
	try {
		diffTens[0].set_constVal(1.0);
		diffTens[1].set_constVal(1.0);
		diffTens[2].set_constVal(1.0);
		diffTens[3].clear();
		phi_exact.clear();
		lambda.clear();
		phi.clear();
		rhs.clear();

		for(size_t iz = 0; iz <= mx[2]; ++iz) {
			const double zVal = grid.get_Pos(2,iz);
			for(size_t iy = 0; iy <= mx[1]; ++iy) {
				const double yVal = grid.get_Pos(1,iy);
				for(size_t ix = 0; ix <= mx[0]; ++ix) {
					const double xVal = grid.get_Pos(0,ix);

					phi_exact(ix,iy,iz) = sin(M_PI * xVal)*sin(M_PI * yVal)*sin(M_PI * zVal);
					lambda(ix,iy,iz) = 0.2*xVal*sqr(yVal)*zVal;



					switch(testSwitch) {
					case TEST_ONE:
						rhs(ix,iy,iz) = -(sqr(M_PI)*(diff(0) + diff(1) + diff(2)) + lambda(ix,iy,iz))*phi_exact(ix,iy,iz);
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

	/* ==== Initialisation of the linear solver ============================ */
	VERBOSE("  Setting up solver ... ");
	try {
		BiCGStabSolver *bicgsolver = new BiCGStabSolver(grid, tolerance, 2, oclContext);
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
		solver->solve(boundaries, phi, rhs, lambda, diff[0], diff[1], diff[2], 8);

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
	} catch (OpenCLException &e) {
		cerr << "OpenCL exception thrown: " << e.what() << " - " << e.opencl_error_string() << endl;
		// Just exit, I am lazy (missing cleanup)
		return EXIT_FAILURE;
	} catch (const char *msg) {
		cerr << "Exception thrown: " << msg << endl;
		// Just exit, I am lazy (missing cleanup)
		return EXIT_FAILURE;
	}
	cout << "=======================================================" << endl << endl;

	/* ==== CLEANUP ======================================================== */
	VERBOSE("Cleanup ... ");
	DELETE(solver);
	DELETE(oclContext);

	// Goodbye message
	runtime += time_ms();
	cout << "Total runtime: " << runtime << " ms" << endl;
	cout << "Bye" << endl;
    return EXIT_SUCCESS;
}





static void printHelp(string programName) {
    cout << "  SYNOPSIS" << endl;
    cout << "    " << programName << " [OPTIONS]" << endl << endl;
    cout << "  OPTIONS" << endl;
    cout << "    -h   --help                   Show program help" << endl;
    cout << "    -v   --verbose                Verbose mode" << endl;
    cout << "    -n N                          Set problem size to N" << endl;
    cout << "         --cpu                    Use CPU context (OpenCL)" << endl;
    cout << "         --gpu                    Use GPU context (OpenCL)" << endl;
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
