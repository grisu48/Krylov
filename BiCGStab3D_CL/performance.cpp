
/* =============================================================================
 *
 * Title:         OpenCL matrix operations performance measuring program
 * Author:        Felix Niederwanger
 * Description:   Example and test program for the OpenCL matrix operations
 *
 * =============================================================================
 */

#include <iostream>
#include <cstdlib>
#include <sstream>
#include <iomanip>
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


#define _DEV_TYPE_DEFAULT 0
#define _DEV_TYPE_CPU 1
#define _DEV_TYPE_GPU 2



#define SEPARATOR "================================================================================\n"

static double random_double() {
	double result = (double)rand();
	result /= (double)RAND_MAX;
	return result;
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
	cout << "Matrix performance measurement program, 2015 Felix Niederwanger" << endl << endl;
	cout << "SYNPOSIS: " << progname << " [OPTIONS]" << endl;
	cout << "OPTIONS:" << endl;
	cout << "  -h   | --help           Print this help message" << endl;
	cout << "  -n N | --size N         Define size of the problem to be N^3" << endl;
	cout << "         --gpu            Use GPU device context" << endl;
	cout << "         --cpu            Use CPU device context" << endl;
	cout << "         --local N        Set local memory to N bytes" << endl;
	cout << "       | --rim N          Set number of RIM cells" << endl;
	cout << "  -v   | --verbose        Verbose output" << endl;
	cout << endl;
}


static inline double matrix_func(int ix, int iy, int iz) {
	return (abs(ix)%5)*random_double()+(abs(iy)%5)*random_double()+(abs(iz)%5)*random_double();
}

static void randomizeMatrix(Matrix3d &matrix) {
	ssize_t size[3] = { (ssize_t)matrix.size(0), (ssize_t)matrix.size(1), (ssize_t)matrix.size(2) };
	const ssize_t rim = matrix.rim();
	for(ssize_t ix=-rim;ix<size[0]+rim;ix++) {
		for(ssize_t iy=-rim;iy<size[1]+rim;iy++) {
			for(ssize_t iz=-rim;iz<size[2]+rim;iz++) {
				matrix(ix,iy,iz) = matrix_func(ix,iy,iz); //(ix*ix) + ix*iy + iz;
			}
		}
	}
}


int main(int argc, char** argv) {
	/** Device type to be used */
	int type = 0;
	bool verbose = false;
	long iterations = 1000L;

	/** Problem size */
	size_t size = 64;
	size_t rim = 0;
	size_t local_mem_size = 40 * 1024L;
	bool local_mem_set = false;


	// Desired stats - The first is the kernel time, the second each is the real time in milliseconds
	double transfer_time;
	double addition_time[2];
	double subtraction_time[2];
	double mutliplication_time[2];
	double dot_product_itself_time[2];
	double dot_product_foreign_time[2];
	double dot_product_time;


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
			} else if(arg == "--rim") {
				if(isLast) throw "Missing argument: rim";
				rim = (size_t)atol(argv[++i]);
			} else if(arg == "--verbose" || arg == "-v") {
				verbose = true;
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
		context->createProfilingCommandQueue();

		if(verbose) {
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
		}
	} catch (OpenCLException &e) {
		cerr << "OpenCL exception: " << e.what() << " - " << e.opencl_error_string() << endl;
		return EXIT_FAILURE;
	}
	if(verbose) { cout << "Setting up matrix (" << size << "^3) ... "; cout.flush(); }
	time = -time_ms();
	Matrix3d matrix(size,size,size, rim, "matrix");
	randomizeMatrix(matrix);
	time += time_ms();
	if(verbose) { cout << "done (" << time << " ms)" << endl; cout.flush(); }


	if(verbose) { cout << "Transferring matrices to device ... "; cout.flush(); }
	else cout << "Running tests ... " << endl;

	CLMatrix3d *m1;
	CLMatrix3d *m2;
	try {
		time = -time_ms();
		m1 = matrix.transferToDevice(context);
		m2 = matrix.transferToDevice(context);
		time += time_ms();
		transfer_time = time;
		m1->initializeContext();
		m2->initializeContext();
		m1->setProfiling(true);
		m2->setProfiling(true);

		m1->setLocalMemorySize(local_mem_size);
		m2->setLocalMemorySize(local_mem_size);

		if(verbose) cout << "done (" << time << " ms)" << endl;
	} catch (CompileException &e) {
		cerr << "Compilation failed: " << e.what() << endl;
		cerr << e.compile_output() << endl;
		return EXIT_FAILURE;
	} catch (OpenCLException &e) {
		cerr << "OpenCL exception : " << e.what() << " - " << e.opencl_error_string() << endl;
		return EXIT_FAILURE;
	}


	if(verbose) cout << "Number of iterations: " << iterations << endl;
	/* ====  Doing performance tests ==== */
	long runtime;
	// Addition
	if(verbose) { cout << "Addition ... "; cout.flush(); }
	context->join();
	runtime = 0L;
	time = -time_ms();
	for(long i=0;i<iterations;i++) {
		m1->add(m2);
		runtime += m1->lastKernelRuntime();
	}
	context->join();
	time += time_ms();
	addition_time[0] = runtime / 1e6;
	addition_time[1] = time;
	m1->transferToDevice(matrix, true);
	m2->transferToDevice(matrix, true);
	if(verbose) { cout << time << " ms" << endl; cout.flush(); }
	// Subtraction
	if(verbose) { cout << "Subtraction ... "; cout.flush(); }
	context->join();
	time = -time_ms();
	runtime = 0L;
	for(long i=0;i<iterations;i++) {
		m1->sub(m2);
		runtime += m1->lastKernelRuntime();
	}
	context->join();
	time += time_ms();
	subtraction_time[0] = runtime / 1e6;
	subtraction_time[1] = time;
	m1->transferToDevice(matrix, true);
	m2->transferToDevice(matrix, true);
	if(verbose) { cout << time << " ms" << endl; cout.flush(); }
	// Multiplication
	if(verbose) { cout << "Multiplication ... "; cout.flush(); }
	context->join();
	time = -time_ms();
	runtime = 0L;
	for(long i=0;i<iterations;i++) {
		m1->mul(m2);
		runtime += m1->lastKernelRuntime();
	}
	context->join();
	time += time_ms();
	mutliplication_time[0] = runtime / 1e6;
	mutliplication_time[1] = time;
	m1->transferToDevice(matrix, true);
	m2->transferToDevice(matrix, true);
	if(verbose) { cout << time << " ms" << endl; cout.flush(); }
	// Dot product
	if(verbose) { cout << "Dot-Product ... "; cout.flush(); }
	context->join();
	time = -time_ms();
	runtime = 0L;
	context->join();
	for(long i=0;i<iterations;i++) {
		m1->dotProduct();
		runtime += m1->lastKernelRuntime();
	}
	context->join();
	time += time_ms();
	dot_product_itself_time[0] = runtime / 1e6;
	dot_product_itself_time[1] = time;
	time = -time_ms();
	runtime = 0L;
	for(long i=0;i<iterations;i++) {
		m1->dotProduct(m2);
		runtime += m1->lastKernelRuntime();
	}
	context->join();
	time += time_ms();
	dot_product_foreign_time[0] = runtime / 1e6;
	dot_product_foreign_time[1] = time;
	m1->transferToDevice(matrix, true);
	m2->transferToDevice(matrix, true);
	if(verbose) { cout << time << " ms" << endl; cout.flush(); }

	// Just for fun
	if(verbose) { cout << "Dot-Product (vanilla)... "; cout.flush(); }
	m1->setProfiling(false);
	m2->setProfiling(false);
	m1->transferToDevice(matrix, true);
	m2->transferToDevice(matrix, true);
	context->join();
	runtime = -time_ms();
	for(long i=0;i<iterations;i++)
		m1->dotProduct();
	context->join();
	runtime +=time_ms();
	dot_product_time = runtime;
	if(verbose) { cout << runtime << " ms" << endl; cout.flush(); }



	cout << SEPARATOR;
	cout << "  --------RESULTS--------" << endl;
	cout << "                          :   PROFILE  | WALL CLOCK " << endl;
	double index[2];
	index[0] = 0.0;
	index[1] = ((double)transfer_time / (double)(size*size*size))*1e5;
	cout << "    Transfer time         : " << setw(10) << index[0] << " | " << setw(10) << index[1] << endl;
	index[0] = (double)addition_time[0]/(double)iterations;
	index[1] = (double)addition_time[1]/(double)iterations;
	cout << "    Addition              : " << setw(10) << index[0] << " | " << setw(10) << index[1] << endl;
	index[0] = (double)subtraction_time[0]/(double)iterations;
	index[1] = (double)subtraction_time[1]/(double)iterations;
	cout << "    Subtraction           : " << setw(10) << index[0] << " | " << setw(10) << index[1] << endl;
	index[0] = (double)mutliplication_time[0]/(double)iterations;
	index[1] = (double)mutliplication_time[1]/(double)iterations;
	cout << "    Multiplication        : " << setw(10) << index[0] << " | " << setw(10) << index[1] << endl;
	index[0] = (double)dot_product_itself_time[0]/(double)iterations;
	index[1] = (double)dot_product_itself_time[1]/(double)iterations;
	cout << "  -- Performance indices -- " << endl;
	cout << "    dotProduct (itself)   : " << setw(10) << index[0] << " | " << setw(10) << index[1] << endl;
	index[0] = (double)dot_product_foreign_time[0]/(double)iterations;
	index[1] = (double)dot_product_foreign_time[1]/(double)iterations;
	cout << "    dotProduct (foreign)  : " << setw(10) << index[0] << " | " << setw(10) << index[1] << endl;
	cout << "    dotProduct (vanilla)  : " << setw(10) << dot_product_time << " | " << setw(10) << ((double)dot_product_time/(double)iterations) << endl;
	cout << "  -- lower index means better result -- " << endl;
	cout << SEPARATOR;





	return EXIT_SUCCESS;
}
