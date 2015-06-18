/* =============================================================================
 *
 * Title:         FlexCL Matrix
 * Author:        Felix Niederwanger
 * Description:   OpenCL Matrix library. Include this file as source file,
 *                since this uses a template class
 * =============================================================================
 */

#include "FlexCL.hpp"

#include <vector>
#include <string>
#include <cmath>

#ifndef _FLEXCL_LIBRARY_MATRIX_HPP_
#define _FLEXCL_LIBRARY_MATRIX_HPP_

// Static defines
#define _FLEXCL_MATRIX_STATUS_UNINITIALIZED_ 0
#define _FLEXCL_MATRIX_STATUS_READY_ 1


namespace flexCL {


class CLMatrix_d;
class CLMatrix3d;


/**
 * 3-dimension double Matrix in RAM memory.
 */
class Matrix3d {
private:
	/** Data */
	double* data;
	/** Defined number of matrix cells without RIM cells */
	size_t _mx[3];
	/** RIM cells in each dimension */
	size_t _rim;

	std::string _name;

#if _FLEXCL_MATRIX_BOUNDS_CHECK != 0
	inline
#endif
	size_t index(int x, int y, int z);

	void boundsCheck(Matrix3d &matrix) { this->boundsCheck(&matrix); }
	void boundsCheck(Matrix3d *matrix);

public:
	Matrix3d(size_t mx, size_t my, size_t mz, size_t rim = 0, std::string name = "");
	virtual ~Matrix3d();

	/** Size of the matrix without RIM cells */
	size_t size(void) { return this->_mx[0] * this->_mx[1] * this->_mx[2]; }
	/** Total size of the matrix */
	size_t sizeTotal(void) { return (_mx[0]+2*_rim)*(_mx[1]+2*_rim)*(_mx[2]+2*_rim); }
	/** Size in the given dimension wihtout RIM cells */
	size_t size(int dim) { return this->_mx[dim]; }
	size_t rim() { return this->_rim; }

	bool hasNanValues(bool includeRim = true);
	bool isNull(bool includeRim = true);

	double& get(int x, int y, int z) { return data[index(x,y,z)]; }
	double& operator()(int x, int y, int z) { return data[index(x,y,z)]; }
	double& operator[](int index) { return data[index]; }
	void set(int index, double value) { this->data[index] = value; }
	void set(int x, int y, int z, double value) { this->data[index(x,y,z)] = value; }

	bool operator==(Matrix3d* matrix) { return this->equals(matrix); }
	bool operator==(Matrix3d& matrix) { return this->equals(matrix); }
	bool operator!=(Matrix3d* matrix) { return !(this->equals(matrix)); }
	bool operator!=(Matrix3d& matrix) { return !(this->equals(matrix)); }
	bool equals(Matrix3d& matrix, bool includeRim = false) { return this->equals(&matrix, includeRim); }
	bool equals(Matrix3d* matrix, bool includeRim = false);


	/**
	 * returns the number of cells that do not match in the different matrices
	 */
	size_t compare(Matrix3d* matrix, bool includeRim = false);
	/**
	 * returns the number of cells that do not match in the different matrices
	 */
	size_t compare(Matrix3d& matrix, bool includeRim = false) { return this->compare(&matrix, includeRim); }

	void copyFrom(Matrix3d &matrix);


	Matrix3d& mul(double factor, bool includeRim = true);
	Matrix3d& add(double summand, bool includeRim = true);
	Matrix3d& sub(double minuend, bool includeRim = true);
	Matrix3d& div(double divisor, bool includeRim = true);

	Matrix3d& mul(Matrix3d *matrix, bool includeRim = true);
	Matrix3d& add(Matrix3d *matrix, bool includeRim = true);
	Matrix3d& sub(Matrix3d *matrix, bool includeRim = true);
	Matrix3d& div(Matrix3d *matrix, bool includeRim = true);

	Matrix3d& mul(Matrix3d &matrix, bool includeRim = true) { return this->mul(&matrix, includeRim); }
	Matrix3d& add(Matrix3d &matrix, bool includeRim = true) { return this->add(&matrix, includeRim); }
	Matrix3d& sub(Matrix3d &matrix, bool includeRim = true) { return this->sub(&matrix, includeRim); }
	Matrix3d& div(Matrix3d &matrix, bool includeRim = true) { return this->div(&matrix, includeRim); }


	Matrix3d& operator*(double factor) { return this->mul(factor); }
	Matrix3d& operator+(double summand) { return this->add(summand); }
	Matrix3d& operator*(Matrix3d &matrix) { return this->mul(matrix); }
	Matrix3d& operator+(Matrix3d &matrix) { return this->add(matrix); }
	Matrix3d& operator-(Matrix3d &matrix) { return this->sub(matrix); }

	//double dotProduct(Matrix3d &matrix, bool includeRim = false) { return this->dotProduct(&matrix, includeRim); }
	double dotProduct(Matrix3d *matrix, bool includeRim = false);
	double dotProduct(Matrix3d &matrix, bool includeRim = false) { return this->dotProduct(&matrix, includeRim); }
	double dotProduct() { return this->dotProduct(this, false); }

	/** Get the absolute maximum value of the matrix */
	double maxNorm(bool includeRim = false);
	/** L2 norm of the matrix */
	double l2Norm(bool includeRim = false);

	/** Set the whole matrix to zero */
	void clear(void);

	/**
	 * Prints the given slice at the z coordinate to the given output stream or cout, if not defined
	 */
	void printSlice(size_t z, std::ostream &out = std::cout, bool includeRim = false);

	/** Prnts the differences and their indices */
	void printDifferences(Matrix3d* matrix, std::ostream &out = std::cout, bool includeRim = false);
	void printDifferences(Matrix3d& matrix, std::ostream &out = std::cout, bool includeRim = false) { this->printDifferences(&matrix, out, includeRim); }

	/** Access to the raw data array of the matrix.
	 * DO NOT USE THIS UNLESS YOU KNOW WHAT YOU ARE DOING
	 * */
	double* raw(void) { return data; }
	std::string name() { return this->_name; }


	CLMatrix3d* transferToDevice(Context *context);

};


/**
 * Matrix in OpenCL
 */
class CLMatrix_d {
protected:
	/** Memory of the matrix */
	cl_mem _mem;
	/** Initialisation status */
	int _status;
	/** OpenCL context */
	flexCL::Context* _context;

	/** Name of the matrix (mostly used for debugging purposes) */
	std::string _name;

	/** Local memory size */
	size_t _localMemSize;
	/** Maximum work group size */
	size_t _maxWorkGroupSize;

	/** RIM cells in each dimension */
	size_t _rim;

	/** Loaded opencl program */
	flexCL::Program *_program = NULL;
	flexCL::Kernel *_kern_Transpose = NULL;
	flexCL::Kernel *_kern_MatrixMatrixMul = NULL;
	flexCL::Kernel *_kern_MatrixMatrixMulSetRim = NULL;
	flexCL::Kernel *_kern_MatrixMul = NULL;
	flexCL::Kernel *_kern_MatrixMatrixAdd = NULL;
	flexCL::Kernel *_kern_MatrixMatrixAddMul = NULL;
	flexCL::Kernel *_kern_MatrixAdd = NULL;
	flexCL::Kernel *_kern_MatrixMatrixSub = NULL;
	flexCL::Kernel *_kern_MatrixMatrixSubMul = NULL;
	flexCL::Kernel *_kern_MatrixSub = NULL;

	flexCL::Kernel *_kern_Reduction_Local = NULL;
	flexCL::Kernel *_kern_Reduction_Local_max = NULL;

	flexCL::Kernel *_kern_ArraySet = NULL;
	flexCL::Kernel *_kern_ArrayAbs = NULL;
	flexCL::Kernel *_kern_ArrayAdd = NULL;
	flexCL::Kernel *_kern_ArraySub = NULL;
	flexCL::Kernel *_kern_ArrayMul = NULL;
	flexCL::Kernel *_kern_ArrayDiv = NULL;
	flexCL::Kernel *_kern_ArrayArrayAdd = NULL;
	flexCL::Kernel *_kern_ArrayArraySub = NULL;
	flexCL::Kernel *_kern_ArrayArrayMul = NULL;
	flexCL::Kernel *_kern_ArrayArrayDiv = NULL;
	flexCL::Kernel *_kern_ArrayArrayAddMul = NULL;
	flexCL::Kernel *_kern_ArrayArraySubMul = NULL;


	void clArraySet(cl_mem array, size_t size, size_t offset = 0, double value = 0.0);
	void clArrayAbs(cl_mem array, size_t size, size_t offset = 0);


	void clArrayArrayAdd(cl_mem array1, cl_mem array2, cl_mem dst, size_t size);
	void clArrayArraySub(cl_mem array1, cl_mem array2, cl_mem dst, size_t size);
	void clArrayArrayMul(cl_mem array1, cl_mem array2, cl_mem dst, size_t size);
	void clArrayArrayDiv(cl_mem array1, cl_mem array2, cl_mem dst, size_t size);
	void clArrayAdd(cl_mem array1, double value, cl_mem dst, size_t size);
	void clArraySub(cl_mem array1, double value, cl_mem dst, size_t size);
	void clArrayMul(cl_mem array1, double value, cl_mem dst, size_t size);
	void clArrayDiv(cl_mem array1, double value, cl_mem dst, size_t size);
	void clArrayAddMul(cl_mem array1, cl_mem array2, cl_mem dst, size_t size, double value);
	void clArraySubMul(cl_mem array1, cl_mem array2, cl_mem dst, size_t size, double value);

	double reduction_double(cl_mem buffer, size_t size, size_t offset = 0);
	double reduction_max_double(cl_mem buffer, size_t size, size_t offset = 0);
	double array_max(cl_mem buffer, size_t size, size_t offset = 0);

	/** Aquire buffer with the given number of double cells  */
	void aquireBuffer(size_t size);
	/** Release memory from OpenCL device */
	void release();





public:
	CLMatrix_d(Context*);
	virtual ~CLMatrix_d();

	virtual void initializeContext(void);

	size_t rim(void) { return this->_rim; }
	/** Set the complete matrix to 0 */
	void clear(void) { this->setConstantValue(0.0); }
	void setConstantValue(double value);

	bool isInitialized(void) { return this->_status == _FLEXCL_MATRIX_STATUS_READY_; }
	size_t localMemorySize(void) { return this->_localMemSize; }
	void setLocalMemorySize(size_t size) { this->_localMemSize = size; }
	void setName(std::string name) { this->_name = name; }

	cl_mem& clMem(void) { return this->_mem; }
	Context* clContext(void) { return this->_context; }
	std::string name() { return this->_name; }


	/** Transfers the given pointer to the OpenCL device */
	virtual void transferToDevice(double* data, bool blocking = true) = 0;
	/** Transfers the data from the OpenCL device to the given pointer at the host */
	virtual void transferToHost(double* dst, bool blocking = true) = 0;

	/** Total size of the matrix excluding RIM cells */
	virtual size_t size() = 0;
	/** Total size of the matrix including RIM cells */
	virtual size_t sizeTotal() = 0;


};





class CLMatrix3d : public CLMatrix_d {
protected:
	/** Number of cells for the matrix (EXCLUDING RIM cells) */
	size_t _mx[3];

	/** Kernel that was last executed */
	flexCL::Kernel *_lastExecutedKernel;

	/** If profiling is enabled (high level profiling) */
	bool _profiling;
	/** Runtime in nanoseconds of the last execution, if profiling is enabled*/
	long _runtime;

	/** Kernel for clearing the RIM fields */
	flexCL::Kernel *_kern_ClearRim = NULL;






	/** Matrix index function. MUST be the same than in the kernel! */
	size_t matrix_index(size_t x, size_t y, size_t z);


	bool equalDimensions(CLMatrix3d &matrix) { return this->equalDimensions(&matrix); }
	bool equalDimensions(CLMatrix3d *matrix);


	/** Set RIM cells to 0*/
	void setRim(cl_mem mem, double value);



	void boundsCheck(CLMatrix3d &matrix) { this->boundsCheck(&matrix); }
	void boundsCheck(CLMatrix3d *matrix);
	void boundsCheck(Matrix3d &matrix) { this->boundsCheck(&matrix); }
	void boundsCheck(Matrix3d *matrix);

	/** Multiplication and set RIM to value */
	void mulRim(CLMatrix3d *matrix, cl_mem dst, double valueRim = 0.0);

public:
	CLMatrix3d(Context*, size_t* size, double* data = NULL, size_t rim = 0);
	CLMatrix3d(Context*, size_t mx, size_t my, size_t mz, double* data = NULL, size_t rim = 0);
	virtual ~CLMatrix3d();


	virtual void initializeContext(void);

	virtual void transferToDevice(double* data, bool blocking = true);
	virtual void transferToDevice(Matrix3d *matrix, bool blocking = true);
	virtual void transferToDevice(Matrix3d &matrix, bool blocking = true) { this->transferToDevice(&matrix, blocking); }
	virtual void transferToHost(double* dst, bool blocking = true);
	virtual void transferToHost(Matrix3d *matrix, bool blocking = true);
	virtual void transferToHost(Matrix3d &matrix, bool blocking = true) { this->transferToHost(&matrix, blocking); }
	Matrix3d* transferToHost(void);

	/** Size of the matrix in given dimension (EXCLUDING RIM cells) */
	virtual size_t size(int dim) { return this->_mx[dim]; }
	/** Total size of the matrix excluding RIM cells */
	virtual size_t size() { return this->_mx[0] * this->_mx[1] * this->_mx[2]; }
	/** Total size of the matrix including RIM cells */
	virtual size_t sizeTotal() { return (this->_mx[0]+2*_rim) * (this->_mx[1]+2*_rim) * (this->_mx[2]+2*_rim); }
	virtual size_t mx(int dim) { return this->_mx[dim]; }


	/** Returns true if profiling is enabled */
	bool profiling(void);
	/** Enable or disable high-level profiling */
	void setProfiling(bool);
	/** If high level profiling is enabled this returns the runtime of the last operation on the device in nanoseconds */
	long lastKernelRuntime(void);


	double dotProduct(CLMatrix3d *matrix);
	double dotProduct(CLMatrix3d &matrix) { return this->dotProduct(&matrix); }
	double dotProduct() { return this->dotProduct(this); }
	double l2Norm(void);
	double maxNorm(bool includeRim = false);

	/** Get a certain value. Very slow, do NOT use this if not absolutely necessary! */
	//double operator()(int x, int y, int z);

	/** Set RIM cells*/
	void clearRim() { this->setRim(this->_mem, 0.0); }
	void setRim(double value = 0.0) { this->setRim(this->_mem, value); }

	CLMatrix3d& add(double summand) { return this->add(summand ,this->_mem); }
	CLMatrix3d& add(CLMatrix3d *matrix) { return this->add(matrix, this->_mem); }
	CLMatrix3d& add(double summand, cl_mem dst);
	CLMatrix3d& add(CLMatrix3d *matrix, cl_mem dst);

	CLMatrix3d& addMultiplied(CLMatrix3d &matrix, double factor) { return this->addMultiplied(&matrix, this->_mem, factor); }
	CLMatrix3d& subMultiplied(CLMatrix3d &matrix, double factor) { return this->subMultiplied(&matrix, this->_mem, factor); }
	CLMatrix3d& addMultiplied(CLMatrix3d *matrix, double factor) { return this->addMultiplied(matrix, this->_mem, factor); }
	CLMatrix3d& subMultiplied(CLMatrix3d *matrix, double factor) { return this->subMultiplied(matrix, this->_mem, factor); }
	CLMatrix3d& addMultiplied(CLMatrix3d *matrix, cl_mem dst, double factor);
	CLMatrix3d& subMultiplied(CLMatrix3d *matrix, cl_mem dst, double factor);

	CLMatrix3d& sub(double minuend) { return this->sub(minuend ,this->_mem); }
	CLMatrix3d& sub(CLMatrix3d *matrix) { return this->sub(matrix, this->_mem); }
	CLMatrix3d& sub(double minuend, cl_mem dst);
	CLMatrix3d& sub(CLMatrix3d *matrix, cl_mem dst);

	CLMatrix3d& mul(double factor) { return this->mul(factor ,this->_mem); }
	CLMatrix3d& mul(CLMatrix3d *matrix) { return this->mul(matrix, this->_mem); }
	CLMatrix3d& mul(double factor, cl_mem dst);
	CLMatrix3d& mul(CLMatrix3d *matrix, cl_mem dst);

	CLMatrix3d& div(double dividend) { return this->mul(1.0/dividend); }
	CLMatrix3d& div(double dividend, cl_mem dst) { return this->mul(1.0/dividend, dst); }

	virtual void copyFrom(CLMatrix3d *matrix);
	virtual void copyFrom(CLMatrix3d &matrix) { this->copyFrom(&matrix); }

};


}  // Namespace flexCL


#endif
