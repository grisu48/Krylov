
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
#include <algorithm>

#include <cstdlib>
#include <cmath>
#include <ctime>
#include <unistd.h>
#include <signal.h>
#include <time.h>

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
/** Test switch */
static int testSwitch = TEST_ONE;
/** Tolerance for the computation */
static double tolerance = 1e-6;
/** Randomize input parameters */
static bool randomize = false;

/* ==== Function prototypes ================================================ */

#define VERBOSE(x) if(verbose) cout << x << endl;
#define DELETE(x) { if(x!=NULL) delete x; x = NULL; }

static void printHelp(string programName = "bicgstab_cl");
static void sig_handler(int);

/** Check given matrix for illegal values */
static bool checkMatrix(NumMatrix<double,3>&);

/** Random double float */
static inline double randomf(double min = 0.0, double max = 1.0);

/** Get a random seed */
static long randomSeed(void);

/** Arithmetic average */
static double average(const vector<long> &values);

/** Calculate geometric average */
static double geometric_average(const std::vector<long> &values);

/** Calculate standart deviation */
static double stdev(const std::vector<long> &values);

/* ==== MAIN PROGRAM FUNCTION ============================================== */

int main(int argc, char** argv) {
	signal(SIGINT, sig_handler);
	signal(SIGUSR1, sig_handler);
	signal(SIGTERM, sig_handler);
	//signal(SIGSEGV, sig_handler);

	// Program variables
	int mode = RUN_NORMAL;

	// Runtime variables
	long ocl_setup_time = 0;
	long total_init_time = 0;
	long problem_init_time = 0;
	long solver_setup_time = 0;
	double max_error = -1.0;
	double l2err = 0.0;
	int iterations = 1;


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
			else if(arg == "--acc" || arg == "--accelerator")
				opencl_context = OCL_CONTEXT_ACCL;
			else if(arg == "-i" || arg == "--iterations") {
				if(isLast) throw "Missing argument: Number of iterations";
				iterations = atoi(argv[++i]);
			} else if(arg == "-n" || arg == "--np") {
				if(isLast) throw "Missing argument: Problem size";
				size = atoi(argv[++i]);
			} else if(arg == "-t" || arg == "--test") {
				if(isLast) throw "Missing argument: Test case";
				testSwitch = atoi(argv[++i]);
			} else if(arg == "-p" || arg == "--precision" || arg == "--tolerance") {
				if(isLast) throw "Missing argument: Tolerance";
				const double tol = atof(argv[++i]);
				if(tol <= 0) throw "Illegal argument: Tolerance zero or negative";
				tolerance = tol;
			} else if(arg == "-r" || arg == "--random" || arg == "--randomize") {
				randomize = true;
			} else if(arg == "-s" || arg == "--stats" || arg == "--statistical") {
				mode = RUN_STATS;
			} else {
				cerr << "Illegal argument: " << arg << "." << endl;
				cerr << "  Type " << argv[0] << " --help if you need help" << endl;
				return EXIT_FAILURE;
			}
		} catch(const char* msg) {
			cerr << msg << endl;
			return EXIT_FAILURE;
		}
	}

	// Initialize random generator
	if(randomize) {
		time_t seed = randomSeed();
		if((mode & RUN_NORMAL) != 0)
			cout << "\tInitialize random generator. Seed = " << seed << endl;
		else if ((mode & RUN_STATS) != 0) {
			cout << "# Random Seed = " << seed << endl;
		}
		srand(seed);
	}


	if((mode & RUN_NORMAL) != 0)
		cout << " -- BiCGStab 3D Linear solver (OpenCL) --" << endl;

	/* ==== OpenCL Initialisation ========================================== */
	ocl_setup_time = -time_ms();
	total_init_time = -time_ms();
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
		case OCL_CONTEXT_ACCL:
			oclContext = openCl.createContext(CL_DEVICE_TYPE_ACCELERATOR);
			break;
		default:
			oclContext = openCl.createContext();
			break;
		}
		if(oclContext == NULL) {
			cerr << "OpenCL context intialisation failed: No such device found" << endl;
			return EXIT_FAILURE;
		}

		DeviceInfo oclDevice = oclContext->device_info();
		if((mode & RUN_NORMAL) != 0)
			cout << "OpenCL context initialized on device " <<oclDevice.device_id() << ": " << oclDevice.vendor() << " " << oclDevice.name() << endl;

	} catch (OpenCLException &e) {
		cerr << "Error setting up OpenCL context: " << e.what() << endl;
		return EXIT_FAILURE;
	}
	ocl_setup_time += time_ms();


	/* ==== Problem initialisation ========================================= */
	problem_init_time = -time_ms();
	static size_t mx[3] = {size, size, size };
	static size_t Nx_global[3] { size+1, size+1, size+1 };



	for(int iteration=1;iteration<=iterations;iteration++) {
		long iterationRuntime = -time_ms();
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

		// Randomized values
		double lambda_factor = 0.2;
		double diffTensFactor[4];

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

			// Randomize
			if(randomize) {
				lambda_factor = randomf(0.1, 100.0);
				if((mode & RUN_NORMAL) != 0)
					cout << "\tRandom lambda factor = " << lambda_factor << endl;
				for(int i=0;i<4;i++){
					diffTensFactor[i] = randomf(0.1,10.0);
					if((mode & RUN_NORMAL) != 0)
						cout << "\tRandom diffTens[" << i << "] = " << diffTensFactor[i] << endl;
				}
			} else {
				lambda_factor = 0.2;
				for(int i=0;i<4;i++) diffTensFactor[i] = 1.0;
			}

			for(size_t iz = 0; iz <= mx[2]; ++iz) {
				const double zVal = grid.get_Pos(2,iz);
				for(size_t iy = 0; iy <= mx[1]; ++iy) {
					const double yVal = grid.get_Pos(1,iy);
					for(size_t ix = 0; ix <= mx[0]; ++ix) {
						const double xVal = grid.get_Pos(0,ix);

						// cout << "[" << ix << "," << iy << "," << iz << "] = (" << xVal << "," << yVal << "," << zVal << ")" << endl;


						phi_exact(ix,iy,iz) = sin(M_PI * xVal)*sin(M_PI * yVal)*sin(M_PI * zVal);
						lambda(ix,iy,iz) = lambda_factor*xVal*sqr(yVal)*zVal;



						switch(testSwitch) {
						case TEST_ONE:
							rhs(ix,iy,iz) = -(sqr(M_PI)*(diff(0) + diff(1) + diff(2)) + lambda(ix,iy,iz))*phi_exact(ix,iy,iz);
							break;
						case TEST_TWO:		// Test 2 (räumliche Diffusion)
						{
							phi_exact(ix,iy,iz) = sin(pi*xVal)*sin(pi*yVal)*sin(pi*zVal);

							diffTens[0](ix,iy,iz) = diffTensFactor[0] * yVal;
							diffTens[1](ix,iy,iz) = diffTensFactor[1] * xVal;
							diffTens[2](ix,iy,iz) = diffTensFactor[2] * zVal;
							rhs(ix,iy,iz) = -(sqr(pi)*(xVal + yVal + zVal) + lambda(ix,iy,iz))*phi_exact(ix,iy,iz) + pi*sin(pi*xVal)*sin(pi*yVal)*cos(pi*zVal);
						}
						break;
						case TEST_THREE:	// Test 3 (räumliche Diffusion mit D_xy)
						{
							double AVal = 0.1;//1.8;
							diffTens[0](ix,iy,iz) = diffTensFactor[0] * yVal;
							diffTens[1](ix,iy,iz) = diffTensFactor[1] * xVal;
							diffTens[2](ix,iy,iz) = diffTensFactor[2] * zVal;
							diffTens[3](ix,iy,iz) = diffTensFactor[3] * AVal*sqr(xVal)*yVal*zVal;
							const double D_xy = diffTens[3](ix,iy,iz);
							rhs(ix,iy,iz) = -(sqr(pi)*(xVal + yVal + zVal) + lambda(ix,iy,iz))*phi_exact(ix,iy,iz) +
									pi*sin(pi*xVal)*sin(pi*yVal)*cos(pi*zVal) +
									2.*D_xy*sqr(pi)*cos(pi*xVal)*cos(pi*yVal)*sin(pi*zVal)+
									2.*AVal*xVal*yVal*zVal*pi*sin(pi*xVal)*cos(pi*yVal)*sin(pi*zVal) +
									AVal*sqr(xVal)*zVal*pi*cos(pi*xVal)*sin(pi*yVal)*sin(pi*zVal);
						}
						break;
						case TEST_FOUR:		// Test 4 (räumliche Diffusion)
						{
							phi_exact(ix,iy,iz) = sin(pi*xVal)*sin(pi*yVal)*sin(pi*zVal);
							const double angle = atan2(yVal, xVal);

							diffTens[0](ix,iy,iz) = diffTensFactor[0] * (dPar*sqr(sin(angle)) + dPerp*sqr(cos(angle)));
							diffTens[1](ix,iy,iz) = diffTensFactor[1] * (dPar*sqr(cos(angle)) +dPerp*sqr(sin(angle)));
							diffTens[2](ix,iy,iz) = diffTensFactor[2] * dPerp;
							diffTens[3](ix,iy,iz) = diffTensFactor[3] * (dPerp - dPar)*sin(angle)*cos(angle);

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
						case TEST_FIVE:		// Test 5 (räumliche Diffusion)
						{
							phi_exact(ix,iy,iz) = sin(pi*xVal)*sin(pi*yVal)*sin(pi*zVal);
							if(ix==mx[0] || iy==mx[1] || iz==mx[2]) {
								phi_exact(ix,iy,iz) = 0.;
							}

							diffTens[0](ix,iy,iz) = diffTensFactor[0];
							diffTens[1](ix,iy,iz) = diffTensFactor[1];
							//diffTens[2](ix,iy,iz) = 1. + 0.00000001*xVal;
							diffTens[2](ix,iy,iz) = diffTensFactor[2];
							diffTens[3](ix,iy,iz) = 0.0;

							rhs(ix,iy,iz) = -(sqr(pi)*(1. + 0.00000001*xVal + 1. + 1.) +
									lambda(ix,iy,iz))*phi_exact(ix,iy,iz);
							//rhs(ix,iy,iz) = -(sqr(pi)*(1. + 1. + 1.) + lambda(ix,iy,iz))*phi_exact(ix,iy,iz);
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
		problem_init_time += time_ms();

		/* Pre-run test to check if any illegal values are in the matrices */
		try {
			if(!checkMatrix(phi)) throw "Matrix phi";
			if(!checkMatrix(phi_exact)) throw "Matrix phi_exact";
			if(!checkMatrix(diffTens[0])) throw "Matrix diffTens[0]";
			if(!checkMatrix(diffTens[1])) throw "Matrix diffTens[1]";
			if(!checkMatrix(diffTens[2])) throw "Matrix diffTens[2]";
			if(!checkMatrix(diffTens[3])) throw "Matrix diffTens[3]";
			if(!checkMatrix(lambda)) throw "Matrix lambda";

			// XXX: In test case four we have some illegal values at first iteration
			//      So just ignore the check for test switch four. This is dirty
			if (testSwitch == TEST_FOUR) {
				cout.flush();
				cerr << "rhs check disabled for test case 4." << endl;
				cerr.flush();
			} else
				if(!checkMatrix(rhs)) throw "Matrix rhs";

			if((mode & RUN_NORMAL) != 0)
				cout << "  Pre-run checks completed." << endl;

		} catch (const char* msg) {
			cerr << "Pre-run check of the matrices failed: " << msg << endl;
			exit(5);
		}

		/* ==== Initialisation of the linear solver ============================ */
		VERBOSE("  Setting up solver ... ");
		solver_setup_time = -time_ms();
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
		solver_setup_time += time_ms();


		/* ==== Solver ========================================================= */
		VERBOSE("Problem setup complete.");
		total_init_time += time_ms();
		long calc_runtime = -time_ms();
		if((mode & RUN_NORMAL) != 0) {
			cout << endl << "=======================================================" << endl;
			cout << "  Running calculation ... " << endl;
		}


		try {

			if(testSwitch == TEST_ONE) {
				VERBOSE("Solving with diagonal diffusion matrix ... ");
				solver->solve(boundaries, phi, rhs, lambda, diff[0], diff[1], diff[2], 8);
			} else if (testSwitch == TEST_TWO || testSwitch == TEST_FIVE) {
				VERBOSE("Solving with arbitrary diffusion matrix ... ");
				solver->solve(boundaries, phi, rhs, lambda, diffTens[0], diffTens[1], diffTens[2], diffTens[3], 8);
			} else if(testSwitch == TEST_THREE || testSwitch == TEST_FOUR) {
				VERBOSE("Solving with arbitrary diffusion matrix with off-diagonal elements ... ");
				solver->solve(boundaries, phi, rhs, lambda, diffTens[0], diffTens[1], diffTens[2], diffTens[3], 8, true);
			} else {
				cerr << "UNKNOWN TEST SWITCH: " << testSwitch << endl;
				return EXIT_FAILURE;
			}


			calc_runtime += time_ms();

			if((mode & RUN_NORMAL) != 0)
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
			l2err = sqrt(error/num);
			if((mode & RUN_NORMAL) != 0)
				cout << " l2 error: " << l2err << "  (" << error << "/" << num << ")" << endl;

			// Detailed look at solution
			// Now let's have a look:
#if 0
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
#endif

			// Search for maximum error
			for(size_t iz = 0; iz <= mx[2]; ++iz) {
				for(size_t iy = 0; iy <= mx[1]; ++iy) {
					for(size_t ix = 0; ix <= mx[0]; ++ix) {
						const double c_error = fabs(phi_exact(ix,iy,iz)-phi(ix,iy,iz));
						if(max_error <= 0.0) max_error = c_error;
						else max_error = fmax(max_error, c_error);
					}
				}
			}

			if((mode & RUN_NORMAL) != 0)
				cout << "Maximum error found on grid: " << max_error << endl;

		} catch (OpenCLException &e) {
			cerr << "OpenCL exception thrown: " << e.what() << " - " << e.opencl_error_string() << endl;
			delete solver;
			delete oclContext;
			return EXIT_FAILURE;
		} catch (NumException &e) {
			cerr << "Numerical exception occurred: " << e.what() << endl;
			delete solver;
			delete oclContext;
			return EXIT_FAILURE;
		} catch (const char *msg) {
			cerr << "Exception thrown: " << msg << endl;
			delete solver;
			delete oclContext;
			return EXIT_FAILURE;
		} catch (...) {
			cerr << "Unknown exception thrown." << endl;
			delete solver;
			delete oclContext;
			return EXIT_FAILURE;
		}
		if((mode & RUN_NORMAL) != 0)
			cout << "=======================================================" << endl << endl;

		/* ==== CLEANUP ======================================================== */
		const long iterations = bicgsolver->iterations();
		const long steptimeMin = bicgsolver->steptimeMin();
		const long steptimeMax = bicgsolver->steptimeMax();


		// Statistics
		iterationRuntime  += time_ms();

		vector<long> runtimes = bicgsolver->stepRuntimes();
		const double avg_runtime = round(average(runtimes)*10.0)/10.0;
		const double stdev_normal = round(stdev(runtimes)*10.0)/10.0;
		const double avg_geometric = geometric_average(runtimes);


		if(verbose) {
			cout << " ==== Time statistics ==== " << endl;
			cout << "\tCalculation time               : " << calc_runtime << " ms (" << iterations << " iterations)" << endl;
			cout << "\tIteration runtime              : " << iterationRuntime << " ms" << endl;

			cout << "\t  Minimum step time            : " << steptimeMin << " ms" << endl;
			cout << "\t  Maximum step time            : " << steptimeMax << " ms" << endl;
			cout << "\t  Average time (arithmetic)    : " << avg_runtime << " +/-" << stdev_normal << " ms/iterations" << endl;
			cout << "\t  Average time (geometric)     : " << avg_geometric << " ms/iterations" << endl;
			cout << endl;
			cout << "\tOpenCL setup time              : " << ocl_setup_time << " ms" << endl;
			cout << "\tTotal initialisation time      : " << total_init_time << " ms" << endl;
			cout << "\tProblem initialisation time    : " << problem_init_time << " ms" << endl;
			cout << "\tSolver setup time              : " << solver_setup_time << " ms" << endl;

			cout << endl << " ==== Numerics ==== " << endl;
			cout << "\tTolerance: " << tolerance << endl;

			if (randomize) {
				cout << endl;
				cout << " ==== Randomized input ==== " << endl;
				cout <<"\tLambda factor: " << lambda_factor << endl;
				for(int i=0;i<4;i++)
					cout << "\tdiffTensFactor[" << i << "] = " << diffTensFactor[i] << endl;
			}
		} else {

			if (mode == RUN_STATS) {
				static bool firstRun = true;
				if(firstRun) {
					cout << "# Test\tSize\tIterations\tTotal runtime\tCalculation time\tMin step time\tMax step time\tAverage (arithmetic)\tAverage (geometric)\tTolerance\tL2 error\tMax error" << endl;
					firstRun = false;
				}

				int test = 0;
				switch(testSwitch) {
				case TEST_ONE:
					test = 1;
					break;
				case TEST_TWO:
					test = 2;
					break;
				case TEST_THREE:
					test = 3;
					break;
				case TEST_FOUR:
					test = 4;
					break;
				case TEST_FIVE:
					test = 5;
					break;
				}

				const char sep = '\t';
				cout << test << sep << size << sep << iterations << sep << iterationRuntime << sep << calc_runtime <<
						sep << steptimeMin << sep << steptimeMax << sep <<
						avg_runtime << sep << avg_geometric << sep << tolerance << sep << l2err << sep << max_error << endl;
			} else {
				// Default mode
				cout << "\tCalculation time: " << calc_runtime << " ms (" << iterations << " iterations)" << endl;
				cout << "\tTotal runtime:    " << runtime << " ms" << endl;
			}
		}
	}


	VERBOSE("Cleanup ... ");
	delete solver;
	delete oclContext;

	runtime += time_ms();

	// Goodbye, only if not in stats mode
	if (mode != RUN_STATS) {
		cout << "  Total runtime: " << runtime << " ms" << endl;
		cout << "Bye" << endl;
	}
	return EXIT_SUCCESS;
}





static void printHelp(string programName) {
	cout << "  SYNOPSIS" << endl;
	cout << "    " << programName << " [OPTIONS]" << endl << endl;
	cout << "  OPTIONS" << endl;
	cout << "    -h   --help                   Show program help" << endl;
	cout << "    -v   --verbose                Verbose mode" << endl;
	cout << "    -n N --np N                   Set problem size to N" << endl;
	cout << "    -i N --iterations N           Set number of iterations to N (default: 1)" << endl;
	cout << "         --cpu                    Use CPU context (OpenCL)" << endl;
	cout << "         --gpu                    Use GPU context (OpenCL)" << endl;
	cout << "         --acc                    Use accelerator context (OpenCL)" << endl;
	cout << "         --accelerator            Use accelerator context (OpenCL)" << endl;
	cout << "    -t   --test TEST              Set test case to TEST" << endl;
	cout << "    -p   --tolerance TOL          Set tolerance to TOL" << endl;
	cout << "    -r   --random                 Randomize matrix" << endl;
	cout << "    -s   --stats                  Statistical output (less than normal but normalized as CSV)" << endl;
}


static void sig_handler(int sig_no) {
	switch(sig_no) {
	case SIGTERM:
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

static double randomf(double min, double max) {
	const double diff = max-min;
	return min + (double)rand()/(double)(RAND_MAX/diff);
}

static long randomSeed(void) {
	struct timespec spec;
	clock_gettime(CLOCK_REALTIME, &spec);
	return spec.tv_sec ^ spec.tv_nsec;

}

static double average(const vector<long> &values) {
	const size_t size = values.size();
	if(size == 0) return 0.0;
	vector<long> runtimes(values);

	double avg = 0.0;
	for(unsigned int i=0;i<size;i++)
		avg += values[i];
	avg /= size;

	return avg;
}

static double geometric_average(const vector<long> &values) {
	const size_t size = values.size();
	if(size == 0) return 0.0;

	// Copy array, because we need them sorted
	vector<long> runtimes(values);
	std::sort(runtimes.begin(), runtimes.end());
	return (double)runtimes[size/2];
}

static inline double sqr(double x) { return (x*x); }

static double stdev(const std::vector<long> &values) {
	const size_t size = values.size();
	if(size == 0) return 0.0;
	vector<long> runtimes(values);

	const double avg = average(values);
	double value = 0.0;
	for(unsigned int i=0;i<size;i++) {
		const double current = values[i];
		value += sqr(current - avg);
	}
	value /= (double)sqr(size);

	return sqrt(value);
}
