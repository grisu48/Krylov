/* =============================================================================
 *
 * Title:         FlexCL Matrix
 * Author:        Felix Niederwanger
 * Description:   OpenCL Matrix library. Include this file as source file,
 *                since this uses a template class
 * =============================================================================
 */
#include "FlexCL.hpp"
#include <iostream>
#include <iomanip>
#include "FlexCLMatrix.hpp"

#ifndef _FLEXCL_LIBRARY_MATRIX_CPP_
#define _FLEXCL_LIBRARY_MATRIX_CPP_


/* ==== USER CONFIGURATION SECTION ==== */

// Perform additional checks to verify the stability of the code
#ifndef _FLEXCL_ADDITIONAL_CHECKS
#define _FLEXCL_ADDITIONAL_CHECKS 1
#endif
// Assumed local size, if the device query fails
#ifndef _FLEXCL_ASSUMED_LOCALMEM_
#define _FLEXCL_ASSUMED_LOCALMEM_ 16*1024L
#endif

// Enable profiling (low-level)
#ifndef _FLEXCL_MATRIX_PROFILING_
#define _FLEXCL_MATRIX_PROFILING_ 0
#endif

/* ==== STATIC CONFIGURATION ==== */

#ifndef _FLEXCL_MATRIX_KERNEL_FILENAME
#define _FLEXCL_MATRIX_KERNEL_FILENAME "flex_matrix.cl"
#endif

// Tolerance for numeric comparisons
#ifndef _FLEXCL_MATRIX_NUMERIC_TOLERANCE_
#define _FLEXCL_MATRIX_NUMERIC_TOLERANCE_ 1e-6
#endif

/* Deep numeric comparison (compares also infinity and nan values) */
// Switch this on if you need this comparisons, disable to increase performance
#ifndef _FLEXCL_MATRIX_NUMERIC_DEEP_COMPARISON
#define _FLEXCL_MATRIX_NUMERIC_DEEP_COMPARISON 0
#endif






/* ==== ADDITIONAL SETUP ==== */

// Use plain array operations to increase performance
#ifndef _FLEXCL_MATRIX_ARRAY_OPERATIONS
#define _FLEXCL_MATRIX_ARRAY_OPERATIONS 1
#endif

#if _FLEXCL_MATRIX_PROFILING_ == 1
#include "time_ms.h"
#endif


static inline bool fequals(double x, double y) {
#if _FLEXCL_MATRIX_NUMERIC_DEEP_COMPARISON == 1
	if(::isnan(x) && ::isnan(y)) return true;
	if(::isinf(x) && ::isinf(y)) return true;
#endif
	return (fabs(x-y)/(abs(fmax(x,y))+_FLEXCL_MATRIX_NUMERIC_TOLERANCE_) < _FLEXCL_MATRIX_NUMERIC_TOLERANCE_);
}

#define DELETE(x) { if(x!=NULL) delete x; x = NULL }
#define DEL_ARRAY(x) { if(x!=NULL) delete[] x; x = NULL; }

#endif




using namespace std;


namespace flexCL {


Matrix3d::Matrix3d(size_t mx, size_t my, size_t mz, size_t rim, string name) {
	this->_mx[0] = mx;
	this->_mx[1] = my;
	this->_mx[2] = mz;
	this->_rim = rim;
	const size_t size = (mx+2*rim)*(my+2*rim)*(mz+2*rim);
	this->data = new double[size];
	this->_name = string(name);
	this->clear();
}
Matrix3d::~Matrix3d() {
	DEL_ARRAY(this->data);
}

size_t Matrix3d::index(int x, int y, int z) {
	size_t mx = _mx[0] + 2*_rim;
	size_t my = _mx[1] + 2*_rim;
#if _FLEXCL_MATRIX_BOUNDS_CHECK == 1
	const int rim = (int)_rim;
	if(x < -rim) throw "Matrix x-index < 0";
	if(y < -rim) throw "Matrix y-index < 0";
	if(z < -rim) throw "Matrix z-index < 0";
#endif
	const size_t _x = _rim + x;
	const size_t _y = _rim + y;
	const size_t _z = _rim + z;
#if _FLEXCL_MATRIX_BOUNDS_CHECK == 1
	size_t mz = _mx[2] + 2*_rim;
	if(_x >= mx) throw "Matrix x-index > bounds";
	if(_y >= my) throw "Matrix y-index > bounds";
	if(_z >= mz) throw "Matrix z-index > bounds";
#endif
	size_t index =_z*mx*my + _y*mx + _x;
#if _FLEXCL_MATRIX_BOUNDS_CHECK == 1
	size_t max = this->sizeTotal();
	if(index >= max)
		throw "Matrix index (overall) > bounds";
#endif
	return index;
}

void Matrix3d::boundsCheck(Matrix3d *matrix) {
	if(matrix->_mx[0] != this->_mx[0]) throw "Matrix size mismatch (mx[0])";
	if(matrix->_mx[1] != this->_mx[1]) throw "Matrix size mismatch (mx[1])";
	if(matrix->_mx[2] != this->_mx[2]) throw "Matrix size mismatch (mx[1])";
	if(matrix->_rim != this->_rim) throw "Matrix size mismatch (rim)";
}

bool Matrix3d::equals(Matrix3d* matrix, bool includeRim) {
	// Boundaries must be the same
	if(matrix->_mx[0] != this->_mx[0]) return false;
	if(matrix->_mx[1] != this->_mx[1]) return false;
	if(matrix->_mx[2] != this->_mx[2]) return false;
	if(includeRim) if(matrix->_rim != this->_rim) return false;

	if(includeRim) {
		size_t _size = this->sizeTotal();
		for(size_t i=0;i<_size;i++)
			if(!fequals(matrix->data[i], this->data[i])) return false;
	} else {
		for(size_t ix=0;ix<_mx[0];ix++)
			for(size_t iy=0;iy<_mx[1];iy++)
				for(size_t iz=0;iz<_mx[2];iz++) {
					const int index = this->index(ix,iy,iz);
					if(!fequals(this->data[index], matrix->data[index])) return false;
				}

	}
	return true;

}

bool Matrix3d::hasNanValues(bool includeRim) {
	if(includeRim) {
		size_t _size = this->sizeTotal();
		for(size_t i=0;i<_size;i++) {
			const double value = this->data[i];
			if(::isnan(value))
				return true;
		}
	} else {
		for(size_t ix=0;ix<_mx[0];ix++)
			for(size_t iy=0;iy<_mx[1];iy++)
				for(size_t iz=0;iz<_mx[2];iz++) {
					const int i = this->index(ix,iy,iz);
					const double value = this->data[i];
					if(::isnan(value))
						return true;
				}
	}
	return false;
}

size_t Matrix3d::compare(Matrix3d* matrix, bool includeRim) {
#if _FLEXCL_ADDITIONAL_CHECKS == 1 || _FLEXCL_MATRIX_BOUNDS_CHECK == 1
	this->boundsCheck(matrix);
#endif
	size_t result = 0;

	if(includeRim) {
		size_t _size = this->sizeTotal();
		for(size_t i=0;i<_size;i++)
			if(!fequals(matrix->data[i], this->data[i])) result++;
	} else {
		for(size_t ix=0;ix<_mx[0];ix++)
			for(size_t iy=0;iy<_mx[1];iy++)
				for(size_t iz=0;iz<_mx[2];iz++) {
					const int i = this->index(ix,iy,iz);
						if(!fequals(matrix->data[i], this->data[i])) {
							result++;
						}
				}
	}

	return result;
}

void Matrix3d::copyFrom(Matrix3d &matrix) {
#if _FLEXCL_ADDITIONAL_CHECKS == 1 || _FLEXCL_MATRIX_BOUNDS_CHECK == 1
	this->boundsCheck(matrix);
#endif

	size_t _size = this->size();
	for(size_t i=0;i<_size;i++)
		this->data[i] = matrix.data[i];
}


Matrix3d& Matrix3d::mul(double factor, bool includeRim) {
	if(includeRim) {
		size_t size = this->sizeTotal();
		for(size_t i=0;i<size;i++)
			data[i] *= factor;
	} else {
		for(size_t ix=0;ix<_mx[0];ix++)
			for(size_t iy=0;iy<_mx[1];iy++)
				for(size_t iz=0;iz<_mx[2];iz++) {
					const int i = this->index(ix,iy,iz);
					data[i] *= factor;
				}
	}

	return *this;
}
Matrix3d& Matrix3d::add(double summand, bool includeRim) {
	if(includeRim) {
		size_t size = this->sizeTotal();
		for(size_t i=0;i<size;i++)
			data[i] += summand;
	} else {
		for(size_t ix=0;ix<_mx[0];ix++)
			for(size_t iy=0;iy<_mx[1];iy++)
				for(size_t iz=0;iz<_mx[2];iz++) {
					const int i = this->index(ix,iy,iz);
					data[i] += summand;
				}
	}

	return *this;
}
Matrix3d& Matrix3d::sub(double minuend, bool includeRim) {
	if(includeRim) {
		size_t size = this->sizeTotal();
		for(size_t i=0;i<size;i++)
			data[i] -= minuend;
	} else {
		for(size_t ix=0;ix<_mx[0];ix++)
			for(size_t iy=0;iy<_mx[1];iy++)
				for(size_t iz=0;iz<_mx[2];iz++) {
					const int i = this->index(ix,iy,iz);
					data[i] -= minuend;
				}
	}

	return *this;
}

Matrix3d& Matrix3d::div(double divisor, bool includeRim) {
	const int factor = 1.0/divisor;
	return this->mul(factor, includeRim);
}

Matrix3d& Matrix3d::mul(Matrix3d *matrix, bool includeRim) {
#if _FLEXCL_ADDITIONAL_CHECKS == 1 || _FLEXCL_MATRIX_BOUNDS_CHECK == 1
	this->boundsCheck(matrix);
#endif

	if(includeRim) {
		size_t size = this->sizeTotal();
		for(size_t i=0;i<size;i++)
			data[i] *= matrix->data[i];
	} else {
		for(size_t ix=0;ix<_mx[0];ix++)
			for(size_t iy=0;iy<_mx[1];iy++)
				for(size_t iz=0;iz<_mx[2];iz++) {
					const int i = this->index(ix,iy,iz);
					data[i] *= matrix->data[i];
				}
	}

	return *this;
}
Matrix3d& Matrix3d::add(Matrix3d *matrix, bool includeRim) {
#if _FLEXCL_ADDITIONAL_CHECKS == 1 || _FLEXCL_MATRIX_BOUNDS_CHECK == 1
	this->boundsCheck(matrix);
#endif

	if(includeRim) {
		size_t size = this->sizeTotal();
		for(size_t i=0;i<size;i++)
			data[i] += matrix->data[i];
	} else {
		for(size_t ix=0;ix<_mx[0];ix++)
			for(size_t iy=0;iy<_mx[1];iy++)
				for(size_t iz=0;iz<_mx[2];iz++) {
					const int i = this->index(ix,iy,iz);
					data[i] += matrix->data[i];
				}
	}

	return *this;
}
Matrix3d& Matrix3d::sub(Matrix3d *matrix, bool includeRim) {
#if _FLEXCL_ADDITIONAL_CHECKS == 1 || _FLEXCL_MATRIX_BOUNDS_CHECK == 1
	this->boundsCheck(matrix);
#endif

	if(includeRim) {
		size_t size = this->sizeTotal();
		for(size_t i=0;i<size;i++)
			data[i] -= matrix->data[i];
	} else {
		for(size_t ix=0;ix<_mx[0];ix++)
			for(size_t iy=0;iy<_mx[1];iy++)
				for(size_t iz=0;iz<_mx[2];iz++) {
					const int i = this->index(ix,iy,iz);
					data[i] -= matrix->data[i];
				}
	}

	return *this;
}
Matrix3d& Matrix3d::div(Matrix3d *matrix, bool includeRim) {
#if _FLEXCL_ADDITIONAL_CHECKS == 1 || _FLEXCL_MATRIX_BOUNDS_CHECK == 1
	this->boundsCheck(matrix);
#endif

	if(includeRim) {
		size_t size = this->sizeTotal();
		for(size_t i=0;i<size;i++)
			data[i] /= matrix->data[i];
	} else {
		for(size_t ix=0;ix<_mx[0];ix++)
			for(size_t iy=0;iy<_mx[1];iy++)
				for(size_t iz=0;iz<_mx[2];iz++) {
					const int i = this->index(ix,iy,iz);
					data[i] /= matrix->data[i];
				}
	}

	return *this;
}

double Matrix3d::dotProduct(Matrix3d *matrix, bool includeRim) {
#if _FLEXCL_ADDITIONAL_CHECKS == 1 || _FLEXCL_MATRIX_BOUNDS_CHECK == 1
	this->boundsCheck(matrix);
#endif

	double result = 0.0;
	if(matrix == NULL) matrix = this;

	if(includeRim) {
		size_t size = this->sizeTotal();
		for(size_t i=0;i<size;i++)
			result += (this->data[i] * matrix->data[i]);
	} else {
		for(size_t ix=0;ix<_mx[0];ix++)
			for(size_t iy=0;iy<_mx[1];iy++)
				for(size_t iz=0;iz<_mx[2];iz++) {
					const int i = this->index(ix,iy,iz);
					result += (this->data[i] * matrix->data[i]);
				}
	}

	return result;
}

void Matrix3d::clear(void) {
	size_t n = this->sizeTotal();
	for(size_t i=0;i<n;i++) data[i] = 0;
}


void Matrix3d::printSlice(size_t z, std::ostream &out, bool includeRim) {
	if(includeRim) {
		for(size_t ix=_rim;ix<_mx[0]+_rim;ix++) {
			for(size_t iy=_rim;iy<_mx[1]+_rim;iy++)
				out << data[index(ix,iy,z)] << '\t';
			out << endl;
		}
	} else {
		for(size_t ix=0;ix<_mx[0];ix++) {
			for(size_t iy=0;iy<_mx[1];iy++)
				out << data[index(ix,iy,z)] << '\t';
			out << endl;
		}
	}
}

void Matrix3d::printDifferences(Matrix3d* matrix, std::ostream &out, bool includeRim) {
	if(includeRim) {
		ssize_t size[3] { (ssize_t)(_mx[0]+_rim), (ssize_t)(_mx[1]+_rim), (ssize_t)(_mx[2]+_rim) };
		for(ssize_t ix=-_rim;ix<size[0];ix++)
			for(ssize_t iy=-_rim;iy<size[0];iy++)
				for(ssize_t iz=-_rim;iz<size[0];iz++) {
					const int index = this->index(ix,iy,iz);
					if(!fequals(this->data[index], matrix->data[index])) {
						cout << "matrix[" << setw(4) << ix << " " << setw(4) << iy << " " << setw(4) << iz << "] (" << this->data[index] << " != " << matrix->data[index] << ")" << endl;
					}
				}
	} else {
		for(size_t ix=0;ix<_mx[0];ix++)
			for(size_t iy=0;iy<_mx[1];iy++)
				for(size_t iz=0;iz<_mx[2];iz++) {
					const int index = this->index(ix,iy,iz);
					if(!fequals(this->data[index], matrix->data[index])) {
						cout << "matrix[" << setw(4) << ix << " " << setw(4) << iy << " " << setw(4) << iz << "] (" << this->data[index] << " != " << matrix->data[index] << ")" << endl;
					}
				}

	}
}

CLMatrix3d* Matrix3d::transferToDevice(Context *context) {
	CLMatrix3d *result = new CLMatrix3d(context, this->_mx[0], this->_mx[1], this->_mx[2], this->data, this->_rim);
	result->setName(this->_name);
	result->initializeContext();
	return result;
}

















CLMatrix_d::CLMatrix_d(Context* context) {
	this->_mem = NULL;
	this->_context = context;
	this->_rim = 0;
	this->_status = _FLEXCL_MATRIX_STATUS_UNINITIALIZED_;
	DeviceInfo info = context->device_info();
	this->_localMemSize = (size_t)info.localMemSize();
	this->_maxWorkGroupSize = (size_t)info.maxWorkGroupSize();
}

CLMatrix_d::~CLMatrix_d() {
	this->release();
}

void CLMatrix_d::initializeContext() {
#if _FLEXCL_MATRIX_PROFILING_ == 1
	if(!this->_context->isProfiling())
		this->_context->createProfilingCommandQueue();
#endif

	// Load program
	this->_program = this->_context->createProgramFromSourceFile(_FLEXCL_MATRIX_KERNEL_FILENAME);
	// Load kernels
	this->_kern_Reduction_Local = this->_program->createKernel("reduction_local");
	this->_kern_ArraySet = this->_program->createKernel("array_set");

	this->_kern_ArrayAdd = this->_program->createKernel("array_add");
	this->_kern_ArraySub = this->_program->createKernel("array_sub");
	this->_kern_ArrayMul = this->_program->createKernel("array_mul");
	this->_kern_ArrayDiv = this->_program->createKernel("array_div");
	this->_kern_ArrayArrayAdd = this->_program->createKernel("array_array_add");
	this->_kern_ArrayArraySub = this->_program->createKernel("array_array_sub");
	this->_kern_ArrayArrayMul = this->_program->createKernel("array_array_mul");
	this->_kern_ArrayArrayDiv = this->_program->createKernel("array_array_div");
	this->_kern_ArrayArrayAddMul = this->_program->createKernel("array_array_add_mul");
	this->_kern_ArrayArraySubMul = this->_program->createKernel("array_array_sub_mul");

	this->_status = _FLEXCL_MATRIX_STATUS_READY_;
}

void CLMatrix_d::release() {
	if(this->_mem != NULL) {
		_context->releaseBuffer(this->_mem);
		this->_mem = NULL;
	}
	if(this->_program != NULL) {
		_context->releaseProgram(this->_program);
		this->_program = NULL;

		// Releasing the program releases automatically all created kernels
		this->_kern_Transpose = NULL;
		this->_kern_MatrixMatrixMul = NULL;
		this->_kern_MatrixMul = NULL;
		this->_kern_MatrixMatrixAdd = NULL;
		this->_kern_MatrixMatrixAddMul = NULL;
		this->_kern_MatrixAdd = NULL;
		this->_kern_MatrixMatrixSub = NULL;
		this->_kern_MatrixMatrixSubMul = NULL;
		this->_kern_MatrixSub = NULL;
		this->_kern_Reduction_Local = NULL;
		this->_kern_ArraySet = NULL;
	}
}

void CLMatrix_d::aquireBuffer(size_t size) {
	if(this->_mem != NULL) _context->releaseBuffer(this->_mem);
	this->_mem = this->_context->createBuffer(size);
}

void CLMatrix_d::clArraySet(cl_mem array, size_t size, size_t offset, double value) {
	this->_kern_ArraySet->setArgument(0, array);
	this->_kern_ArraySet->setArgument(1, offset);
	this->_kern_ArraySet->setArgument(2, value);
	this->_kern_ArraySet->enqueueNDRange(size);
}
static void array_arithmetic_operation(Kernel* kernel, cl_mem a1, double a2, cl_mem dst, size_t size) {
	kernel->setArgument(0, a1);
	kernel->setArgument(1, a2);
	kernel->setArgument(2, dst);
	kernel->setArgument(3, size);
	kernel->enqueueNDRange(size);
}

static void array_arithmetic_operation(Kernel* kernel, cl_mem a1, cl_mem a2, cl_mem dst, size_t size) {
	kernel->setArgument(0, a1);
	kernel->setArgument(1, a2);
	kernel->setArgument(2, dst);
	kernel->setArgument(3, size);
	kernel->enqueueNDRange(size);
}

static void array_arithmetic_operation(Kernel* kernel, cl_mem a1, cl_mem a2, cl_mem dst, size_t size, double factor) {
	kernel->setArgument(0, a1);
	kernel->setArgument(1, a2);
	kernel->setArgument(2, dst);
	kernel->setArgument(3, size);
	kernel->setArgument(4, factor);
	kernel->enqueueNDRange(size);
}

void CLMatrix_d::clArrayAdd(cl_mem array1, double value, cl_mem dst, size_t size) {
	array_arithmetic_operation(this->_kern_ArrayAdd, array1, value, dst, size);
}
void CLMatrix_d::clArraySub(cl_mem array1, double value, cl_mem dst, size_t size) {
	array_arithmetic_operation(this->_kern_ArraySub, array1, value, dst, size);
}
void CLMatrix_d::clArrayMul(cl_mem array1, double value, cl_mem dst, size_t size) {
	array_arithmetic_operation(this->_kern_ArrayMul, array1, value, dst, size);
}
void CLMatrix_d::clArrayDiv(cl_mem array1, double value, cl_mem dst, size_t size) {
	array_arithmetic_operation(this->_kern_ArrayDiv, array1, value, dst, size);
}


void CLMatrix_d::clArrayArrayAdd(cl_mem array1, cl_mem array2, cl_mem dst, size_t size) {
	array_arithmetic_operation(this->_kern_ArrayArrayAdd, array1, array2, dst, size);
}
void CLMatrix_d::clArrayArraySub(cl_mem array1, cl_mem array2, cl_mem dst, size_t size) {
	array_arithmetic_operation(this->_kern_ArrayArraySub, array1, array2, dst, size);
}
void CLMatrix_d::clArrayArrayMul(cl_mem array1, cl_mem array2, cl_mem dst, size_t size) {
	array_arithmetic_operation(this->_kern_ArrayArrayMul, array1, array2, dst, size);
}
void CLMatrix_d::clArrayArrayDiv(cl_mem array1, cl_mem array2, cl_mem dst, size_t size) {
	array_arithmetic_operation(this->_kern_ArrayArrayDiv, array1, array2, dst, size);
}

void CLMatrix_d::clArrayAddMul(cl_mem array1, cl_mem array2, cl_mem dst, size_t size, double value) {
	array_arithmetic_operation(this->_kern_ArrayArrayAddMul, array1, array2, dst, size, value);
}
void CLMatrix_d::clArraySubMul(cl_mem array1, cl_mem array2, cl_mem dst, size_t size, double value) {
	array_arithmetic_operation(this->_kern_ArrayArraySubMul, array1, array2, dst, size, value);
}


void CLMatrix_d::setConstantValue(double value) {
	clArraySet(this->_mem, this->sizeTotal(), 0, value);
}

double CLMatrix_d::reduction_double(cl_mem buffer, size_t size, size_t offset) {
#if _FLEXCL_MATRIX_PROFILING_ == 1
	unsigned long profiling_time = -time_ms();
#endif

	// The number of computation units should be a multiple of this value to fully utilize the hardware
	//const size_t computationUnits = this->_kern_Reduction_Local->getPreferredWorkGroupSizeMultiple();
	const size_t maxWorkGroupSize = ::max(this->_maxWorkGroupSize, this->_kern_Reduction_Local->getKernelWorkGroupSize());


	/* Determine the number of working groups and working items by the available local memory and the maximum work group size */

	// Local memory size for double - We must round down to the next integer
	size_t localMemSize = (size_t)(::floor((double)this->_localMemSize/(double)sizeof(double)) * sizeof(double));
	// Local size is the number of elements per scan block
	size_t localSize = localMemSize / sizeof(double);

	if(localSize > maxWorkGroupSize) {		// The bottleneck is the number of working items per group
		localSize = maxWorkGroupSize;
		localMemSize = localSize * sizeof(double);
	}

	// Global size extended to a multiple of the local Size
	const size_t globalSize = (size_t)(::ceil((double)size/(double)localSize))*localSize;

	// Number of working groups
	size_t numWorkGroups = (size_t)::ceil((double)globalSize / (double)localSize);


#if _FLEXCL_ADDITIONAL_CHECKS == 1
		// Check if global size is a multiple of local size
		size_t factor = globalSize / localSize;
		if(localSize*factor != globalSize) {
			cerr << "REDUCTION :: Global size not a multiple of local size !!" << endl;
		}
#endif
#if _FLEXCL_MATRIX_PROFILING_ == 1
	_context->join();
	profiling_time = -time_ms();
#endif

	// Result buffer
	cl_mem result_buf = _context->createBuffer(sizeof(double)*numWorkGroups);

	try {
		// Setup kernel
		_kern_Reduction_Local->setArgument(0, buffer);
		_kern_Reduction_Local->setArgumentLocalMem(1, localMemSize);
		_kern_Reduction_Local->setArgument(2, size);
		_kern_Reduction_Local->setArgument(3, offset);
		_kern_Reduction_Local->setArgument(4, result_buf);
		size_t globalWorkSize[1] = { globalSize };
		size_t localWorkSize[1] = { localSize };
		_kern_Reduction_Local->enqueueNDRange(1, globalWorkSize, localWorkSize);

		_context->join();

		// Old relict of performance testing
#if 0
		long runtime = _kern_Reduction_Local->runtime() / 1e3;
		static long min_runtime = runtime;
		static long max_runtime = runtime;
		static long avg_runtime = runtime;
		static long counter = 0;
		min_runtime = ::min(min_runtime, runtime);
		max_runtime = ::max(max_runtime, runtime);
		avg_runtime = (0.95*runtime) + (0.05 * avg_runtime);
		counter++;
		cerr << "reduction " << counter << " within " << runtime << " µs (min = " << min_runtime << ", max = " << max_runtime << ", avg = " << avg_runtime << ") for " << size << " elements" << endl;
#endif


		// Result readout, do not forget to release buffer!
		double *result_buffer = new double[numWorkGroups];
		_context->readBuffer(result_buf, sizeof(double)*numWorkGroups, result_buffer, true);
		_context->releaseBuffer(result_buf);
		result_buf = NULL;

#if _FLEXCL_MATRIX_PROFILING_ == 1
		_context->join();
		profiling_time += time_ms();
		unsigned long runtime = _kern_Reduction_Local->runtime()*1e-6;
		cerr << "Reduction(" << buffer << "," << size << "," << offset << ")  --  " << runtime << " ms (CPU time: " << profiling_time << " ms)" << endl;
		profiling_time = -time_ms();
#endif
		double result = 0.0;
		for(size_t i=0;i<numWorkGroups;i++)
			result += result_buffer[i];
		delete[] result_buffer;
#if _FLEXCL_MATRIX_PROFILING_ == 1
		profiling_time += time_ms();
		cerr << "Reduction rest calculation took " << profiling_time << " ms" << endl;
#endif
		return result;

	} catch(...) {
		// Clean exception handling: Release OpenCL buffer
		if(result_buf != NULL) _context->releaseBuffer(result_buf);
		throw;
	}

}





CLMatrix3d::CLMatrix3d(Context* context, size_t* size, double* data, size_t rim) : CLMatrix3d(context, size[0], size[1], size[2], data, rim) {}

CLMatrix3d::CLMatrix3d(Context* context, size_t mx, size_t my, size_t mz, double* data, size_t rim) : CLMatrix_d(context) {
	this->_mx[0] = mx;
	this->_mx[1] = my;
	this->_mx[2] = mz;
	this->_rim = rim;
	this->_runtime = 0L;
	this->_profiling = false;
	this->_lastExecutedKernel = NULL;
	size_t size = this->sizeTotal();
	CLMatrix_d::aquireBuffer(sizeof(double)*size);
	if(data != NULL)
		this->transferToDevice(data);
}

CLMatrix3d::~CLMatrix3d() {}



void CLMatrix3d::transferToDevice(double* data, bool blocking) {
	if(this->_mem == NULL) throw OpenCLException("Matrix memory already released");
	const size_t size = sizeof(double) * this->sizeTotal();
	if(blocking) this->_context->join();
	this->_context->writeBuffer(this->_mem, size, data, blocking);
	this->_lastExecutedKernel = NULL;
	if(blocking) this->_context->join();
}

Matrix3d* CLMatrix3d::transferToHost(void) {
	Matrix3d *result = new Matrix3d(this->_mx[0], this->_mx[1], this->_mx[2], this->_rim);
	this->transferToHost(result->raw(), true);
	//_context->join();
	return result;
}

void CLMatrix3d::transferToHost(double* dst, bool blocking) {
	if(this->_mem == NULL) throw OpenCLException("Matrix memory already released");
	const size_t size = sizeof(double) * this->sizeTotal();
	if(blocking) this->_context->join();
	this->_context->readBuffer(this->_mem, size, dst, blocking);
	this->_lastExecutedKernel = NULL;
	if(blocking) this->_context->join();
}

void CLMatrix3d::transferToHost(Matrix3d *matrix, bool blocking) {
	if(matrix == NULL) throw OpenCLException("Matrix must not be NULL");
	// Bounds check
	this->boundsCheck(matrix);
	this->transferToHost(matrix->raw(), blocking);
}

void CLMatrix3d::transferToDevice(Matrix3d *matrix, bool blocking) {
	if(matrix == NULL) throw OpenCLException("Matrix must not be NULL");
	// Bounds check
	this->boundsCheck(matrix);
	this->transferToDevice(matrix->raw(), blocking);
}

void CLMatrix3d::boundsCheck(CLMatrix3d *matrix) {
	if(matrix->_mx[0] != this->_mx[0]) throw OpenCLException("CLMatrix bound check failed (mx[0])");
	if(matrix->_mx[1] != this->_mx[1]) throw OpenCLException("CLMatrix bound check failed (mx[1])");
	if(matrix->_mx[2] != this->_mx[2]) throw OpenCLException("CLMatrix bound check failed (mx[2])");
	if(matrix->_rim != this->_rim) throw OpenCLException("CLMatrix bound check failed (rim)");
}

void CLMatrix3d::boundsCheck(Matrix3d *matrix) {
	if(matrix->size(0) != this->_mx[0]) throw OpenCLException("CLMatrix bound check failed (mx[0])");
	if(matrix->size(1) != this->_mx[1]) throw OpenCLException("CLMatrix bound check failed (mx[1])");
	if(matrix->size(2) != this->_mx[2]) throw OpenCLException("CLMatrix bound check failed (mx[2])");
	if(matrix->rim() != this->_rim) throw OpenCLException("CLMatrix bound check failed (rim)");
}


void CLMatrix3d::initializeContext(void) {
	CLMatrix_d::initializeContext();

	//this->_kern_Transpose = this->_program->createKernel("matrix_transpose_3d");
	this->_kern_MatrixMatrixMul = this->_program->createKernel("matrix_matrix_mul_3d");
	this->_kern_MatrixMatrixMulSetRim = this->_program->createKernel("matrix_matrix_mul_rim_3d");
	this->_kern_MatrixMul = this->_program->createKernel("matrix_mul_3d");
	this->_kern_MatrixAdd = this->_program->createKernel("matrix_add_3d");
	this->_kern_MatrixMatrixAdd = this->_program->createKernel("matrix_matrix_add_3d");
	this->_kern_MatrixMatrixAddMul = this->_program->createKernel("matrix_matrix_add_mul_3d");
	this->_kern_MatrixSub = this->_program->createKernel("matrix_sub_3d");
	this->_kern_MatrixMatrixSub = this->_program->createKernel("matrix_matrix_sub_3d");
	this->_kern_MatrixMatrixSubMul = this->_program->createKernel("matrix_matrix_sub_mul_3d");

	this->_kern_ClearRim = this->_program->createKernel("matrix_set_rim");
}

bool CLMatrix3d::equalDimensions(CLMatrix3d *matrix) {
	if(matrix == NULL) return false;
	for(int i=0;i<3;i++)
		if(matrix->_mx[i] != this->_mx[i]) return false;
	if(matrix->_rim != this->_rim) return false;

	return true;
}

void CLMatrix3d::setRim(cl_mem mem, double value) {
	if(this->_rim <= 0) return;

	this->_kern_ClearRim->setArgument(0, mem);
	this->_kern_ClearRim->setArgument(1, this->_rim);
	this->_kern_ClearRim->setArgument(2, value);
	this->_kern_ClearRim->enqueueNDRange(this->_mx[0]+2*_rim, this->_mx[1]+2*_rim, this->_mx[2]+2*_rim);
	this->_lastExecutedKernel = _kern_ClearRim;

	if(this->profiling()) {
		this->_context->join();
		this->_runtime = this->_lastExecutedKernel->runtime();
	}
}


CLMatrix3d& CLMatrix3d::add(double summand, cl_mem dst) {
#if _FLEXCL_MATRIX_ARRAY_OPERATIONS == 1
	clArrayAdd(this->_mem, summand, dst, this->sizeTotal());
	this->_lastExecutedKernel = _kern_ArrayAdd;
#else
	this->_kern_MatrixAdd->setArgument(0, this->_mem);
	this->_kern_MatrixAdd->setArgument(1, dst);
	this->_kern_MatrixAdd->setArgument(2, summand);
	this->_kern_MatrixAdd->setArgument(3, (long)this->_rim);
	this->_kern_MatrixAdd->enqueueNDRange(this->_mx[0]+2*_rim, this->_mx[1]+2*_rim, this->_mx[2]+2*_rim);
	this->_lastExecutedKernel = _kern_MatrixAdd;
#endif

#if _FLEXCL_MATRIX_PROFILING_ == 1
	this->_context->join();
#if _FLEXCL_MATRIX_ARRAY_OPERATIONS == 1
	unsigned long runtime_ms = this->_kern_ArrayMul->runtime() * 1e-6;
#else
	unsigned long runtime_ms = this->_kern_MatrixAdd->runtime() * 1e-6;
#endif
	cerr << "CLMatrix3d:: MatrixAdd -- " << runtime_ms << " ms" << endl;
#endif

	if(this->profiling()) {
		this->_context->join();
		this->_runtime = this->_lastExecutedKernel->runtime();
	}

	return *this;
}

CLMatrix3d& CLMatrix3d::add(CLMatrix3d *matrix, cl_mem dst) {
// __kernel void matrix_matrix_add_3d(__global REAL* m1, __global REAL* m2, __global REAL* dst, size_t mx, size_t my, size_t mz, size_t rim) {
	if(matrix == NULL) matrix = this;

#if _FLEXCL_ADDITIONAL_CHECKS == 1 || _FLEXCL_MATRIX_BOUNDS_CHECK == 1
	this->boundsCheck(matrix);
#endif
#if _FLEXCL_MATRIX_ARRAY_OPERATIONS == 1
	clArrayArrayAdd(this->_mem, matrix->_mem, dst, this->sizeTotal());
	this->_lastExecutedKernel = _kern_ArrayArrayAdd;
#else
	this->_kern_MatrixMatrixAdd->setArgument(0, this->_mem);
	this->_kern_MatrixMatrixAdd->setArgument(1, matrix->_mem);
	this->_kern_MatrixMatrixAdd->setArgument(2, dst);
	this->_kern_MatrixMatrixAdd->setArgument(3, (long)this->_rim);
	this->_kern_MatrixMatrixAdd->enqueueNDRange(this->_mx[0]+2*_rim, this->_mx[1]+2*_rim, this->_mx[2]+2*_rim);
	this->_lastExecutedKernel = _kern_MatrixMatrixAdd;
#endif

#if _FLEXCL_MATRIX_PROFILING_ == 1
	this->_context->join();
#if _FLEXCL_MATRIX_ARRAY_OPERATIONS == 1
	unsigned long runtime_ms = this->_kern_ArrayArrayAdd->runtime() * 1e-6;
#else
	unsigned long runtime_ms = this->_kern_MatrixMatrixAdd->runtime() * 1e-6;
#endif
	cerr << "CLMatrix3d:: MatrixMatrixAdd -- " << runtime_ms << " ms" << endl;
#endif

	if(this->profiling()) {
		this->_context->join();
		this->_runtime = this->_lastExecutedKernel->runtime();
	}

	return *this;
}

CLMatrix3d& CLMatrix3d::addMultiplied(CLMatrix3d *matrix, cl_mem dst, double factor) {
	if(matrix == NULL) matrix = this;

#if _FLEXCL_ADDITIONAL_CHECKS == 1 || _FLEXCL_MATRIX_BOUNDS_CHECK == 1
	this->boundsCheck(matrix);
#endif
#if _FLEXCL_MATRIX_ARRAY_OPERATIONS == 1
	clArrayAddMul(this->_mem, matrix->_mem, dst, this->sizeTotal(), factor);
	this->_lastExecutedKernel = _kern_ArrayArrayAddMul;
#else
	this->_kern_MatrixMatrixAddMul->setArgument(0, this->_mem);
	this->_kern_MatrixMatrixAddMul->setArgument(1, matrix->_mem);
	this->_kern_MatrixMatrixAddMul->setArgument(2, dst);
	this->_kern_MatrixMatrixAddMul->setArgument(3, (long)this->_rim);
	this->_kern_MatrixMatrixAddMul->setArgument(4, factor);
	this->_kern_MatrixMatrixAddMul->enqueueNDRange(this->_mx[0]+2*_rim, this->_mx[1]+2*_rim, this->_mx[2]+2*_rim);
	this->_lastExecutedKernel = _kern_MatrixMatrixAddMul;
#endif

#if _FLEXCL_MATRIX_PROFILING_ == 1
	this->_context->join();
#if _FLEXCL_MATRIX_ARRAY_OPERATIONS == 1
	unsigned long runtime_ms = this->_kern_ArrayArrayAddMul->runtime() * 1e-6;
#else
	unsigned long runtime_ms = this->_kern_MatrixMatrixAddMul->runtime() * 1e-6;
#endif
	cerr << "CLMatrix3d:: MatrixMatrixAdd -- " << runtime_ms << " ms" << endl;
#endif

	if(this->profiling()) {
		this->_context->join();
		this->_runtime = this->_lastExecutedKernel->runtime();
	}

	return *this;
}

CLMatrix3d& CLMatrix3d::subMultiplied(CLMatrix3d *matrix, cl_mem dst, double factor) {
	if(matrix == NULL) matrix = this;

#if _FLEXCL_ADDITIONAL_CHECKS == 1 || _FLEXCL_MATRIX_BOUNDS_CHECK == 1
	this->boundsCheck(matrix);
#endif
#if _FLEXCL_MATRIX_ARRAY_OPERATIONS == 1
	clArraySubMul(this->_mem, matrix->_mem, dst, this->sizeTotal(), factor);
	this->_lastExecutedKernel = _kern_ArrayArraySubMul;
#else
	this->_kern_MatrixMatrixSubMul->setArgument(0, this->_mem);
	this->_kern_MatrixMatrixSubMul->setArgument(1, matrix->_mem);
	this->_kern_MatrixMatrixSubMul->setArgument(2, dst);
	this->_kern_MatrixMatrixSubMul->setArgument(3, (long)this->_rim);
	this->_kern_MatrixMatrixSubMul->setArgument(4, factor);
	this->_kern_MatrixMatrixSubMul->enqueueNDRange(this->_mx[0]+2*_rim, this->_mx[1]+2*_rim, this->_mx[2]+2*_rim);
	this->_lastExecutedKernel = _kern_MatrixMatrixSubMul;
#endif

#if _FLEXCL_MATRIX_PROFILING_ == 1
	this->_context->join();
#if _FLEXCL_MATRIX_ARRAY_OPERATIONS == 1
	unsigned long runtime_ms = this->_kern_ArrayArraySubMul->runtime() * 1e-6;
#else
	unsigned long runtime_ms = this->_kern_MatrixMatrixSubMul->runtime() * 1e-6;
#endif
	cerr << "CLMatrix3d:: MatrixMatrixAdd -- " << runtime_ms << " ms" << endl;
#endif

	if(this->profiling()) {
		this->_context->join();
		this->_runtime = this->_lastExecutedKernel->runtime();
	}

	return *this;
}


CLMatrix3d& CLMatrix3d::sub(double minuend, cl_mem dst) {
#if _FLEXCL_MATRIX_ARRAY_OPERATIONS == 1
	clArraySub(this->_mem, minuend, dst, this->sizeTotal());
#else
	this->_kern_MatrixSub->setArgument(0, this->_mem);
	this->_kern_MatrixSub->setArgument(1, dst);
	this->_kern_MatrixSub->setArgument(2, minuend);
	this->_kern_MatrixSub->setArgument(3, (long)this->_rim);
	this->_kern_MatrixSub->enqueueNDRange(this->_mx[0]+2*_rim, this->_mx[1]+2*_rim, this->_mx[2]+2*_rim);
	this->_lastExecutedKernel = _kern_MatrixSub;
#endif

#if _FLEXCL_MATRIX_PROFILING_ == 1
	this->_context->join();
#if _FLEXCL_MATRIX_ARRAY_OPERATIONS == 1
	unsigned long runtime_ms = this->_kern_ArraySub->runtime() * 1e-6;
#else
	unsigned long runtime_ms = this->_kern_MatrixSub->runtime() * 1e-6;
#endif
	cerr << "CLMatrix3d:: MatrixSub -- " << runtime_ms << " ms" << endl;
#endif

	if(this->profiling()) {
		this->_context->join();
		this->_runtime = this->_lastExecutedKernel->runtime();
	}

	return *this;

}

CLMatrix3d& CLMatrix3d::sub(CLMatrix3d *matrix, cl_mem dst) {
	if(matrix == NULL) matrix = this;

#if _FLEXCL_ADDITIONAL_CHECKS == 1 || _FLEXCL_MATRIX_BOUNDS_CHECK == 1
	this->boundsCheck(matrix);
#endif
#if _FLEXCL_MATRIX_ARRAY_OPERATIONS == 1
	clArrayArraySub(this->_mem, matrix->_mem, dst, this->sizeTotal());
	this->_lastExecutedKernel = _kern_ArrayArraySub;
#else
	this->_kern_MatrixMatrixSub->setArgument(0, this->_mem);
	this->_kern_MatrixMatrixSub->setArgument(1, matrix->_mem);
	this->_kern_MatrixMatrixSub->setArgument(2, dst);
	this->_kern_MatrixMatrixSub->setArgument(3, (long)this->_rim);
	this->_kern_MatrixMatrixSub->enqueueNDRange(this->_mx[0]+2*_rim, this->_mx[1]+2*_rim, this->_mx[2]+2*_rim);
	this->_lastExecutedKernel = _kern_MatrixMatrixSub;
#endif

#if _FLEXCL_MATRIX_PROFILING_ == 1
	this->_context->join();
#if _FLEXCL_MATRIX_ARRAY_OPERATIONS == 1
	unsigned long runtime_ms = this->_kern_ArrayArraySub->runtime() * 1e-6;
#else
	unsigned long runtime_ms = this->_kern_MatrixMatrixSub->runtime() * 1e-6;
#endif
	cerr << "CLMatrix3d:: MatrixMatrixSub -- " << runtime_ms << " ms" << endl;
#endif

	if(this->profiling()) {
		this->_context->join();
		this->_runtime = this->_lastExecutedKernel->runtime();
	}

	return *this;
}


CLMatrix3d& CLMatrix3d::mul(double factor, cl_mem dst) {
#if _FLEXCL_MATRIX_ARRAY_OPERATIONS == 1
	clArrayMul(this->_mem, factor, dst, this->sizeTotal());
	this->_lastExecutedKernel = _kern_ArrayMul;
#else
	this->_kern_MatrixMul->setArgument(0, this->_mem);
	this->_kern_MatrixMul->setArgument(1, dst);
	this->_kern_MatrixMul->setArgument(2, factor);
	this->_kern_MatrixMul->setArgument(3, (long)this->_rim);
	this->_kern_MatrixMul->enqueueNDRange(this->_mx[0]+2*_rim, this->_mx[1]+2*_rim, this->_mx[2]+2*_rim);
	this->_lastExecutedKernel = _kern_MatrixMul;
#endif

#if _FLEXCL_MATRIX_PROFILING_ == 1
	this->_context->join();
#if _FLEXCL_MATRIX_ARRAY_OPERATIONS == 1
	unsigned long runtime_ms = this->_kern_ArrayMul->runtime() * 1e-6;
#else
	unsigned long runtime_ms = this->_kern_MatrixMul->runtime() * 1e-6;
#endif
	cerr << "CLMatrix3d:: MatrixMultiplication -- " << runtime_ms << " ms" << endl;
#endif

	if(this->profiling()) {
		this->_context->join();
		this->_runtime = this->_lastExecutedKernel->runtime();
	}

	return *this;

}

CLMatrix3d& CLMatrix3d::mul(CLMatrix3d *matrix, cl_mem dst) {
	if(matrix == NULL) matrix = this;
#if _FLEXCL_ADDITIONAL_CHECKS == 1 || _FLEXCL_MATRIX_BOUNDS_CHECK == 1
	this->boundsCheck(matrix);
#endif
#if _FLEXCL_MATRIX_ARRAY_OPERATIONS == 1
	clArrayArrayMul(this->_mem, matrix->_mem, dst, this->sizeTotal());
	this->_lastExecutedKernel = _kern_ArrayArrayMul;
#else
	this->_kern_MatrixMatrixMul->setArgument(0, this->_mem);
	this->_kern_MatrixMatrixMul->setArgument(1, matrix->_mem);
	this->_kern_MatrixMatrixMul->setArgument(2, dst);
	this->_kern_MatrixMatrixMul->setArgument(3, (long)this->_rim);
	this->_kern_MatrixMatrixMul->enqueueNDRange(this->_mx[0]+2*_rim, this->_mx[1]+2*_rim, this->_mx[2]+2*_rim);
	this->_lastExecutedKernel = _kern_MatrixMatrixMul;
#endif

#if _FLEXCL_MATRIX_PROFILING_ == 1
	this->_context->join();
#if _FLEXCL_MATRIX_ARRAY_OPERATIONS == 1
	unsigned long runtime_ms = this->_kern_ArrayArrayMul->runtime() * 1e-6;
#else
	unsigned long runtime_ms = this->_kern_MatrixMatrixMul->runtime() * 1e-6;
#endif
	cerr << "CLMatrix3d:: MatrixMatrixMultiplication -- " << runtime_ms << " ms" << endl;
#endif

	if(this->profiling()) {
		this->_context->join();
		this->_runtime = this->_lastExecutedKernel->runtime();
	}

	return *this;
}

void CLMatrix3d::mulRim(CLMatrix3d *matrix, cl_mem dst, double valueRim) {
	if(matrix == NULL) matrix = this;
#if _FLEXCL_ADDITIONAL_CHECKS == 1 || _FLEXCL_MATRIX_BOUNDS_CHECK == 1
	this->boundsCheck(matrix);
#endif
	this->_kern_MatrixMatrixMulSetRim->setArgument(0, this->_mem);
	this->_kern_MatrixMatrixMulSetRim->setArgument(1, matrix->_mem);
	this->_kern_MatrixMatrixMulSetRim->setArgument(2, dst);
	this->_kern_MatrixMatrixMulSetRim->setArgument(3, (long)this->_rim);
	this->_kern_MatrixMatrixMulSetRim->setArgument(4, valueRim);
	this->_kern_MatrixMatrixMulSetRim->enqueueNDRange(this->_mx[0]+2*_rim, this->_mx[1]+2*_rim, this->_mx[2]+2*_rim);
	this->_lastExecutedKernel = _kern_MatrixMatrixMulSetRim;

#if _FLEXCL_MATRIX_PROFILING_ == 1
	this->_context->join();
#if _FLEXCL_MATRIX_ARRAY_OPERATIONS == 1
	unsigned long runtime_ms = this->_kern_ArrayArrayMul->runtime() * 1e-6;
#else
	unsigned long runtime_ms = this->_kern_MatrixMatrixMulSetRim->runtime() * 1e-6;
#endif
	cerr << "CLMatrix3d:: MatrixMatrixMultiplicationRim -- " << runtime_ms << " ms" << endl;
#endif

	if(this->profiling()) {
		this->_context->join();
		this->_runtime = this->_lastExecutedKernel->runtime();
	}
}

double CLMatrix3d::dotProduct(CLMatrix3d *matrix) {
	if(!this->isInitialized()) throw OpenCLException("CLMatrix has not yet been initialized");
	if(matrix == NULL) matrix = this;

#if _FLEXCL_ADDITIONAL_CHECKS == 1 || _FLEXCL_MATRIX_BOUNDS_CHECK == 1
	this->boundsCheck(matrix);
#endif

	// Declarations
	const size_t size = this->sizeTotal();
	cl_mem buffer;				// Reduction buffer on device

	// Create memory buffer for dot product
	buffer = this->_context->createBuffer(sizeof(double)*size);
	try {
		long runtime = 0L;
		const bool profiling = this->profiling();

		if(profiling) this->_context->join();

		// Copy the matrix into the buffer and do multiplication with this matrix
		this->_context->copyBuffer(buffer, matrix->_mem, sizeof(double)*size);

		// MulRim = Multiplication with RIM cleanup
		this->mulRim(matrix, buffer);
		if(profiling) {
			this->_context->join();
			runtime += this->_lastExecutedKernel->runtime();
		}

		// In the buffer is now the multiplied matrix. We need to sum it up using reduction
		double result = this->reduction_double(buffer, size);
		this->_context->releaseBuffer(buffer);
		this->_lastExecutedKernel = this->_kern_Reduction_Local;
		if(profiling) {
			this->_context->join();
			runtime += this->_lastExecutedKernel->runtime();
			this->_runtime = runtime;
		}
		buffer = NULL;
		return result;

	} catch (...) {
		// Clean exception handling: Release OpenCL buffer
		if(buffer != NULL) this->_context->releaseBuffer( buffer );
		throw;
	}
}


bool CLMatrix3d::profiling(void) { return this->_profiling; }
void CLMatrix3d::setProfiling(bool enabled) {
	if(enabled && !_context->isProfiling()) throw OpenCLException("Cannot enable profiling for CLMatrix3d if it is not supported by the context");
	this->_profiling = enabled;
}
long CLMatrix3d::lastKernelRuntime(void) {
	if(!profiling()) return -1L;
	else return this->_runtime;
}

double CLMatrix3d::l2Norm(void) {
	double squared = this->dotProduct(this);
	return sqrt(squared);

}

void CLMatrix3d::copyFrom(CLMatrix3d *matrix) {
	if(!this->isInitialized()) throw OpenCLException("CLMatrix has not yet been initialized");

#if _FLEXCL_ADDITIONAL_CHECKS == 1 || _FLEXCL_MATRIX_BOUNDS_CHECK == 1
	this->boundsCheck(matrix);
#endif

	const size_t size = sizeof(double)*this->sizeTotal();
	this->_context->copyBuffer(this->_mem, matrix->_mem, size);

}

}  // Namespace flexCL

