
/* =============================================================================
 *
 * Title:         OpenCL matrix operations example program
 * Author:        Felix Niederwanger
 * Description:   Example and test program for the OpenCL matrix operations
 *
 * =============================================================================
 */

#include <iostream>
#include <cstdlib>
#include <sstream>
#include <string>
#include <time.h>
#include <pthread.h>
#include <unistd.h>
#include <signal.h>

#include "FlexCL.hpp"
#include "FlexCLMatrix.hpp"
#include "time_ms.h"

using namespace std;
using namespace flexCL;

/** Problem size */
static size_t size = 8;
static size_t rim = 0;
static size_t local_mem_size = 40 * 1024L;
static bool local_mem_set = false;

#define _DEV_TYPE_DEFAULT 0
#define _DEV_TYPE_CPU 1
#define _DEV_TYPE_GPU 2
/** Device type to be used */
static int type = 0;
static long iterations = 250L;		// Desired number of iterations


// Number of threads for single-core runs
// CURRENTLY NOT WORKING!!
#define THREADS 0


#define SEPARATOR "================================================================================\n"
static volatile bool running = true;
static volatile bool terminating = false;
static bool benchmark = false;



static double random_double() {
	double result = (double)rand();
	result /= (double)RAND_MAX;
	return result;
}

static inline double matrix_func(int ix, int iy, int iz) {
	return (abs(ix)%5)*random_double()+(abs(iy)%5)*random_double()+(abs(iz)%5)*random_double();
}

static string mem_string(size_t size) {
	stringstream result;

	if(size > 1024) {
		size_t kB = size/1024;
		if(kB > 1024) {
			size_t MB = kB / 1024;
			if(MB > 1024) {
				float GB = (float)MB / 1024.0;
				result << GB << " GB";
			} else {
				result << MB << " MB";
			}
		} else {
			result << kB << " kB";
		}
	} else
		result << size << " Bytes";
	return result.str();
}

static void printHelp(string progname = "matrix_cl") {
	cout << "Matrix demo program, 2015 Felix Niederwanger" << endl << endl;
	cout << "SYNPOSIS: " << progname << " [OPTIONS]" << endl;
	cout << "OPTIONS:" << endl;
	cout << "  -h   | --help           Print this help message" << endl;
	cout << "  -n N | --size N         Define size of the problem to be N^3" << endl;
	cout << "         --gpu            Use GPU device context" << endl;
	cout << "         --cpu            Use CPU device context" << endl;
	cout << "         --local N        Set local memory to N bytes" << endl;
	cout << "  -i N | --iterations N   Set the number of test iterations to N" << endl;
	cout << "  -b   | --benchmark      Enable benchmark test (disabled by default)" << endl;
	cout << "       | --rim N          Set number of RIM cells" << endl;
	cout << endl;
}
#if THREADS > 1

struct {
	Matrix3d<double>* matrix;
	size_t start;
	size_t end;
} typedef MATRIX_THREAD_FILL_ARG ;

static void matrix_fill_thread(MATRIX_THREAD_FILL_ARG* arg) {
	for(size_t i=arg->start;i<=arg->end;i++)
		arg->matrix->set(i, matrix_func(i));
}

#endif

/*
static void diagonalizeMatrix(Matrix3d &matrix) {
	ssize_t size[3] = { (ssize_t)matrix.size(0), (ssize_t)matrix.size(1), (ssize_t)matrix.size(2) };
		const ssize_t rim = matrix.rim();
		for(ssize_t ix=-rim;ix<size[0]+rim;ix++) {
			for(ssize_t iy=-rim;iy<size[1]+rim;iy++) {
				for(ssize_t iz=-rim;iz<size[2]+rim;iz++) {
					bool isDiagonal = ix==iy;// && iy==iz;
					matrix(ix,iy,iz) = isDiagonal?1.0:0.0;
				}
			}
	}
}
*/

static void randomizeMatrix(Matrix3d &matrix) {
#if THREADS <= 1
	ssize_t size[3] = { (ssize_t)matrix.size(0), (ssize_t)matrix.size(1), (ssize_t)matrix.size(2) };
	const ssize_t rim = matrix.rim();
	for(ssize_t ix=-rim;ix<size[0]+rim;ix++) {
		for(ssize_t iy=-rim;iy<size[1]+rim;iy++) {
			for(ssize_t iz=-rim;iz<size[2]+rim;iz++) {
				matrix(ix,iy,iz) = matrix_func(ix,iy,iz); //(ix*ix) + ix*iy + iz;
			}
		}
	}
#else
	size_t size = matrix.size();

	// Create threads
	pthread_t threads[THREADS];
	MATRIX_THREAD_FILL_ARG args[THREADS];
	size_t start = 0;
	size_t index_per_threads = size/THREADS;
	for(int i=0;i<THREADS;i++) {
		args[i].matrix = &matrix;
		args[i].start = start;
		start += index_per_threads;
		if(i >= THREADS-1)
			start = size;
		args[i].end = start;

		void* arg = &args[i];
		int rc = pthread_create(&threads[i], NULL, (void* (*)(void*))matrix_fill_thread, arg);
		if(rc != 0) throw "Thread creation failed";
	}

	// Join all threads
	for(int i=0;i<THREADS;i++) {
		pthread_join(threads[i], NULL);
	}
#endif
}

static void sig_handler(int sig_no) {
	switch(sig_no) {
	case SIGINT:
	case SIGUSR1:
		cerr << "User cancel request" << endl;
		if(!running) {
			// Exit
			exit(EXIT_FAILURE);
		}
		running = false;
		terminating = true;
		cerr << "Now canceling. Send signal again to terminate program." << endl;
		break;
	case SIGTERM:
		running = false;
		terminating = true;
		exit(EXIT_FAILURE);
		break;
	}
}




int main(int argc, char** argv) {
	srand(time(NULL));

	/** Parse program arguments */
	try {
		for(int i=1;i<argc;i++) {
			string arg = string(argv[i]);
			const bool isLast = i >= argc-1;
			if(arg == "--size" || arg == "-n") {
				if(isLast) throw "Missing argument: Size";
				size = atoi(argv[++i]);
			} else if(arg == "--help" || arg == "-h") {
				printHelp(string(argv[0]));
				return EXIT_SUCCESS;
			} else if(arg == "--gpu") {
				type = _DEV_TYPE_GPU;
			} else if(arg == "--cpu") {
				type = _DEV_TYPE_CPU;
			} else if(arg == "--local") {
				if(isLast) throw "Missing argument: Local memory";
				local_mem_size = (size_t)atol(argv[++i]);
				local_mem_set = true;
			} else if(arg == "--iterations" || arg == "-i") {
				if(isLast) throw "Missing argument: Iterations";
				iterations = atol(argv[++i]);
			} else if(arg == "--benchmark" || arg == "-b") {
				benchmark = true;
			} else if(arg == "--rim") {
				if(isLast) throw "Missing argument: rim";
				rim = (size_t)atol(argv[++i]);
			} else
				throw string("Illegal argument: " + string(arg)).c_str();
		}
	} catch(const char* msg) {
		cerr << msg << endl;
		return EXIT_FAILURE;
	}


	OpenCL opencl;
	Context *context;
	long time;
	try {
		switch(type) {
		case _DEV_TYPE_CPU:
			context = opencl.createCPUContext();
			break;
		case _DEV_TYPE_GPU:
			context = opencl.createGPUContext();
			break;
		default:
			context = opencl.createContext();
			break;
		}

		DeviceInfo devInfo = context->device_info();
		cout << "Using context: " << devInfo.name() << " (" << devInfo.device_id() << ")" << endl;
		cout << "  Global memory: " << mem_string((size_t)devInfo.globalMemSize()) << endl;
		if(!local_mem_set) {
			size_t local_mem = (size_t)devInfo.localMemSize();
			if(local_mem > 0)
				local_mem_size = local_mem;
			else
				cerr << "  WARNING: Failed to query local memory size" << endl;
		}
		cout << "  Local memory: " << mem_string(local_mem_size) << endl;
	} catch (OpenCLException &e) {
		cerr << "OpenCL exception: " << e.what() << " - " << e.opencl_error_string() << endl;
		return EXIT_FAILURE;
	}
	cout << "Setting up matrix (" << size << "^3) ... "; cout.flush();
	Matrix3d matrix(size,size,size, rim, "matrix one");
	try {
		time = -time_ms();
		randomizeMatrix(matrix);
		time += time_ms();
		cout << "done (" << time << " ms)" << endl;
	} catch(const char* msg) {
		cerr << msg << endl;
		return EXIT_FAILURE;
	}

	//matrix.printSlice(SIZE/2);


	cout << "Setting up matrices on device ... "; cout.flush();
	CLMatrix3d *m1;
	CLMatrix3d *m2;
	try {
		m1 = matrix.transferToDevice(context);
		m2 = matrix.transferToDevice(context);
		m1->initializeContext();
		m2->initializeContext();

		m1->setLocalMemorySize(local_mem_size);
		m2->setLocalMemorySize(local_mem_size);

		cout << "done" << endl;
	} catch (CompileException &e) {
		cerr << "Compilation failed: " << e.what() << endl;
		cerr << e.compile_output() << endl;
		return EXIT_FAILURE;
	} catch (OpenCLException &e) {
		cerr << "OpenCL exception : " << e.what() << " - " << e.opencl_error_string() << endl;
		return EXIT_FAILURE;
	}

	try {
		context->join();
		// Transfer data (just for benchmarking)
		cerr.flush();
		cout << endl << "   ..... Preliminary tests .....   " << endl;
		cout << "Transferring data to device ... "; cout.flush();
		time = -time_ms();
		m1->transferToDevice(matrix.raw());
		m2->transferToDevice(matrix.raw());
		context->join();
		time += time_ms();
		cout << "done (" << time << " ms)" << endl;


		m1->dotProduct();


		cout << "Matrix-matrix multiplication ... "; cout.flush();
		time = -time_ms();
		m2->mul(m2);
		context->join();
		time += time_ms();
		cout << "done (" << time << " ms)" << endl;
		//matrix.printSlice(SIZE/2);
		//cout << endl;

		cout << "Matrix multiplication (with factor) ... "; cout.flush();
		time = -time_ms();
		m2->mul(2.0);
		context->join();
		time += time_ms();
		cout << "done (" << time << " ms)" << endl;
		cout << "Transferring data to host ... "; cout.flush();
		time = -time_ms();
		m2->transferToHost(matrix.raw(), true);
		context->join();
		time += time_ms();
		cout << "done (" << time << " ms)" << endl;
		//matrix.printSlice(SIZE/2);
		//cout << endl;


		// Dot product
		cout << "Dot product" << endl;
		m2->transferToDevice(matrix.raw());
		context->join();
		double dotProduct = 0.0;
		time = -time_ms();
		dotProduct = m2->dotProduct();
		context->join();
		time += time_ms();

		cout << "  Returned result = " << dotProduct << " (within " << time << " ms)" << endl;

		// Classical dot product
		if(::isnan(dotProduct)) {
			cout.flush();
			cerr << "Returned dot product is NAN" << endl;
			cerr.flush();
			return EXIT_FAILURE;
		} else {
			time = -time_ms();
			double dotProduct_CPU = matrix.dotProduct(matrix);
			time += time_ms();
			cout << "  Expected result = " << dotProduct_CPU << " (within " << time << " ms)" << endl;

			const double delta = fabs(dotProduct - dotProduct_CPU);
			double delta_relative = fabs(delta / dotProduct_CPU);
			if(fabs(dotProduct_CPU) < 1e-5) delta_relative = fabs(delta);
			if(delta_relative > 1e-5) {
				cerr << "  ERROR: Delta > 0.001% (Delta = " << (delta_relative*100.0) << "%)" << endl;
				cerr << "Initial tests failed." << endl;
				// Lazy exit
				exit(EXIT_FAILURE);
			} else
				cout << "  Delta value = " << (delta_relative*100.0) << "% (below tolerance) : OK" << endl;
		}


		// Now the testing begins
		signal(SIGINT, sig_handler);
		signal(SIGUSR1, sig_handler);
		signal(SIGUSR2, sig_handler);

		cout << endl << endl << SEPARATOR;
		long remaining = iterations;
		long iteration = 0;
		double result[2];
		double t_delta;
		vector<string> errors;
		while(running && remaining-- >0 && !terminating) {
			Matrix3d test_matrix(size,size,size, rim, "test_matrix");
			Matrix3d test1(size,size,size, rim, "test1");
			Matrix3d test2(size,size,size, rim, "test2");

			long runtime = -time_ms();
			stringstream sstatus;
			sstatus << "\rTesting iteration " << iteration++;
			string status = sstatus.str();
			string lotsOfSpaces = "          ";


			try {
				cout << status << " (Generating matrix) ... " << lotsOfSpaces; cout.flush();
				randomizeMatrix(test_matrix);
				test1.copyFrom(test_matrix);
				test2.copyFrom(test_matrix);
				delete m1; m1 = test_matrix.transferToDevice(context);
				delete m2; m2 = test_matrix.transferToDevice(context);
				context->join();		// We join to ensure the memory allocation is completed

				cout << status << " (Randomize matrix) ... " << lotsOfSpaces; cout.flush();
				randomizeMatrix(test_matrix);
				test1.copyFrom(test_matrix);
				cout << status << " (Transfer to device) ... " << lotsOfSpaces; cout.flush();
				m1->transferToDevice(test1.raw());


				// Test 1
				const int test1Count = 13;
				cout << status << " (Test 1: Basic arithmetic) ... " << lotsOfSpaces; cout.flush();
				test1.copyFrom(test_matrix);
				m1->transferToDevice(test1.raw());
				m2->transferToDevice(test1.raw());
				test1.mul(test1Count);
				for(int ii=1;ii<test1Count;ii++)
					m1->add(m2);
				m1->transferToHost(test2.raw());
				if(test1 != test2) {
					/*
					size_t differences = test1.compare(test2);
					cerr << differences << " differences found" << endl;
					cout << endl; test1.printSlice(size/2); cout << endl;
					cout << endl; test2.printSlice(size/2); cout << endl;
					cout << endl;
					test1.printDifferences(test2, cout, true);
					exit(EXIT_FAILURE);
					*/
					throw "Test 1 failed (add/mul equivalent)";
				}

				// Test 2: Dot product
				cout << status << " (dot product) ... " << lotsOfSpaces; cout.flush();
				m1->transferToDevice(test_matrix.raw(), false);
				result[0] = m1->dotProduct();
				test1.copyFrom(test_matrix);
				result[1] = test1.dotProduct();

				t_delta = fabs(result[0] - result[1]);
				if( ::isnan(result[0] || t_delta > 1e-3)) {
					stringstream error;

					error << "Test 2 failed (dotProduct) [";
					error << result[0] << " != " << result[1] << ", delta = " << t_delta << "]";

					throw error.str();
				}

				// Test 3
				cout << status << " (Multiply with matrix) ... " << lotsOfSpaces; cout.flush();
				//diagonalizeMatrix(test1);
				m1->transferToDevice(test1.raw());
				m2->transferToDevice(test2.raw());
				test1.mul(test2);
				m1->mul(m2);
				m1->transferToHost(test2.raw(), true);
				if(test1 != test2) throw "Test3 failed (Matrix-Matrix mul)";


				// Test 4
				cout << status << " (dotProduct with 0) ... " << lotsOfSpaces; cout.flush();
				m1->clear();
				result[0] = m1->dotProduct();
				result[1] = m1->dotProduct(m2);
				if(::isnan(result[0]) || result[0] != 0.0) throw "Test4.1 failed (dotProduct with 0)";
				if(::isnan(result[1]) || result[1] != 0.0) throw "Test4.2 failed (dotProduct with 0)";

				// Test 5
				cout << status << " (combined matrix operations) ... " << lotsOfSpaces; cout.flush();
				const double factor = random_double();
				test1.copyFrom(test_matrix);
				test2.copyFrom(test_matrix);
				m1->transferToDevice(test1.raw(), true);
				m2->transferToDevice(test2.raw(), true);
				m1->addMultiplied(m2, factor);
				test1.add(test2.mul(factor));
				m1->transferToHost(test2.raw(), true);
				if(test1 != test2) throw "Test5 failed (addMultiplied)";
				test1.copyFrom(test_matrix);
				test2.copyFrom(test_matrix);
				m1->transferToDevice(test1.raw(), true);
				m2->transferToDevice(test2.raw(), true);
				m1->subMultiplied(m2, factor);
				test1.copyFrom(test_matrix);
				test2.copyFrom(test_matrix);
				test1.sub(test2.mul(factor));
				m1->transferToHost(test2.raw(), true);
				if(test1 != test2) throw "Test5 failed (subMultiplied)";

				// Test 6 - LOTS of random matrix operations in a row
				int operations = (int)(random_double()*1000);
				cout << status << " (" << operations << " random operations) ... " << lotsOfSpaces; cout.flush();
				test1.copyFrom(test_matrix);
				test2.copyFrom(test_matrix);
				m1->transferToDevice(test1.raw());
				m2->transferToDevice(test2.raw());
				while(operations-- > 0) {
					const int operation = (int)(random_double()*5);
					const double rand_number = random_double();

					switch(operation) {
					case 0:
						m1->add(m2);
						test1.add(test2);
						break;
					case 1:
						m1->sub(m2);
						test1.sub(test2);
						break;
					case 2:
						m1->mul(m2);
						test1.mul(test2);
						break;
					case 3:
						m1->add(rand_number*100.0);
						test1.add(rand_number*100.0);
						break;
					case 4:
						m1->sub(rand_number*100.0);
						test1.sub(rand_number*100.0);
						break;
					case 5:
						m1->mul(rand_number);
						test1.mul(rand_number);
						break;
					default:
						//cerr << "ILLEGAL OPERATION" << endl;
						break;
					}
				}
				context->join();
				m1->transferToHost(test2.raw(), true);
				if(test1 != test2) throw "Test6 failed (multiple consequent operations)";


				// Test 7 - RIM fields (if existing)
				if(rim > 0) {
					cout << status << " (RIM fields) ... " << lotsOfSpaces; cout.flush();
					if(m1->rim() != rim) throw "Test 7 failed (m1.rim != rim)";

					m1->setConstantValue(1.0);
					m1->setRim(5.0);

					// The dot product in this case sohuld be exactly the number of cells
					double returned_product = m1->dotProduct();
					size_t cells = (size_t)returned_product;
					if(cells != m1->size()) {
						stringstream serror;
						serror << "Test 7 failed (returned dot Product does not match number of cells - ";
						ssize_t delta = cells - m1->size();
						serror << "expected: " << m1->size() << ", returned: " << cells << " (" << returned_product << "), delta = " << delta << ")";


						string error = serror.str();
						throw error;
					}

				}


				// Test 8 -- This is actually a benchmark
				float benchmark_factor = 0.0;
				if(benchmark) {
					cout << status << " (Running benchmark - CL) ... " << lotsOfSpaces; cout.flush();
					const int multiplications = 10;
					const int additions = 10;
					const int dotProducts = 10;
					// First, handle operations on the openCL device
					m1->transferToDevice(test_matrix.raw(), true);
					m2->transferToDevice(test_matrix.raw(), true);
					long cl_runtime = -time_ms();
					for(int ii=0;ii<additions;ii++) {
						m1->add(m2);
						for(int jj=0;jj<multiplications;jj++) {
							m2->mul(1.001);
							for(int kk=0;kk<dotProducts;kk++) {
								m2->dotProduct();
								m1->dotProduct();
							}
						}
					}
					context->join();
					cl_runtime += time_ms();
					// Now handle operations on CPU
					cout << status << " (Running benchmark - CPU) ... " << lotsOfSpaces; cout.flush();
					test1.copyFrom(test_matrix);
					test2.copyFrom(test_matrix);
					long cpu_runtime = -time_ms();
					for(int ii=0;ii<additions;ii++) {
						test1.add(test2);
						for(int jj=0;jj<multiplications;jj++) {
							test2.mul(1.001);
							for(int kk=0;kk<dotProducts;kk++) {
								test2.dotProduct();
								test1.dotProduct();
							}
						}
					}
					cpu_runtime += time_ms();
					benchmark_factor = (float)(cpu_runtime) / (float)cl_runtime;
				}

				// Done testing.
				runtime += time_ms();
				if(benchmark)
					cout << status << " ... all tests passed (" << runtime << " ms, benchmark factor = " << benchmark_factor << ")";
				else
					cout << status << " ... all tests passed (" << runtime << " ms)";
				cout << lotsOfSpaces << endl; cout.flush();
				context->join();

			} catch (const char* msg) {
				cout << " failed" << endl; cout.flush();
				cerr << msg << endl;
				cerr.flush();
				errors.push_back(string(msg));
			} catch (string &msg) {
				cout << " failed" << endl; cout.flush();
				cerr << msg << endl;
				cerr.flush();
				errors.push_back(string(msg));
			}

		}

		if(terminating) {
			cout.flush();
			cerr << "Terminated." << endl;
			cerr.flush();
		}

		cout << endl;
		cout << iteration << " iterations - " << errors.size() << " errors" << endl;
		if(errors.size() > 0) {
			cout.flush();
			cerr.flush();
			for(unsigned int i=0;i<errors.size();i++)
				cerr << "Error " << i << ": " << errors.at(i) << endl;
			cerr.flush();
			cout << iteration << " iterations - " << errors.size() << " errors" << endl;
		}

		cout << SEPARATOR;
		cout.flush();
	} catch (OpenCLException &e) {
		cerr << "OpenCL Exception: " << e.what() << " - " << e.opencl_error_string() << endl;
		delete m1;
		delete m2;
		return EXIT_FAILURE;
	}

	cout << "Cleanup ... " << endl;
	delete m1;
	delete m2;


	cout << "Done" << endl;
}

