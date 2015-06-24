/*
 * matrix_mul.cpp
 *
 *  Created on: Apr 2, 2015
 *      Author: phoenix
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
static size_t size = 32;
static size_t rim = 5;


static double random_double() {
	double result = (double)rand();
	result /= (double)RAND_MAX;
	return result;
}

static inline double matrix_func(int ix, int iy, int iz) {
	return (abs(ix)%5)*random_double()+(abs(iy)%5)*random_double()+(abs(iz)%5)*random_double();
}

#if 0
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
#endif

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
	cout << "Matrix multiplication test program" << endl;
	long runtime = -time_ms();

	Matrix3d matrix(size, size, size, rim);
	Matrix3d matrix2(size, size, size, rim);
	Matrix3d matrix_compare(size, size, size, rim);
	OpenCL opencl;
	Context* context = NULL;
	try {
		context = opencl.createCPUContext();

		randomizeMatrix(matrix);
		matrix2.copyFrom(matrix);
		matrix_compare.copyFrom(matrix);
		CLMatrix3d *clMatrix = matrix.transferToDevice(context);
		CLMatrix3d *clMatrix2 = matrix2.transferToDevice(context);
		cout << "  Size = " << clMatrix->size(0) << "x" << clMatrix->size(1) << "x" << clMatrix->size(2) << " (rim = " << clMatrix->rim() << ") " << endl;
		cout << "Matrix ready. Starting process ... " << endl;

		long operations = 100L;
		long errors = 0;
		for(long i=0;i<operations;i++) {
			cout << "Operation " << i << " of " << operations << " ... "; cout.flush();

			//delete clMatrix;
			//clMatrix = matrix_compare.transferToDevice(context);

			clMatrix->add(clMatrix2);
			clMatrix->dotProduct();
			matrix.add(matrix2);

			clMatrix->transferToHost(matrix_compare);
			if(matrix_compare != matrix) {
				size_t diffs = matrix_compare.compare(matrix);
				cout << "failed - differences at " << diffs << " cells" << endl;
				matrix_compare.printDifferences(matrix);
				errors++;
			} else {
				cout << "done" << endl;
			}
		}

		cout << endl;
		cout << operations << " done - " << errors << " errors" << endl;
	} catch(CompileException &e) {
		cerr << "OpenCL compile exception: " << e.what() << " - " << e.opencl_error_string() << endl;
		cerr << e.compile_output() << endl;
		return EXIT_FAILURE;

	} catch (OpenCLException &e) {
		cerr << "OpenCL exception: " << e.what() << " - " << e.opencl_error_string() << endl;
		return EXIT_FAILURE;
	}

	if(context != NULL) delete context;
	runtime +=time_ms();
	cout << "\rRuntime: " << runtime << " ms" << endl;
	cout << "Bye" << endl;;
}
