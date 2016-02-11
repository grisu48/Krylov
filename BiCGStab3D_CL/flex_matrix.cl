/* =============================================================================
 *
 * Title:         BiCGStab 3d OpenCL accelerated Kernel file (File for OpenCL)
 * Author:        Felix Niederwanger
 * Description:   This is a OpenCL program file and contains all functions for
 *                the BiCGStab solver
 * =============================================================================
 */

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// Range check for the matrix index. Turn off in productive code to increase performance, especially on GPU!
#define BOUNDS_CHECK 0
// Turn on Verbosity (in means of some printf) -- DISABLE ON GPU, since it won't be supported by the kernel!
#define VERBOSE 0

// Turn this on, if you need a global memory fence on your architecture. This switch DECREASES THE PERFORMANCE SIGNIFICANTLY
#define ARITHMETIC_GLOBAL_MEM_FENCE 0

#define REAL double
#define size_t long





// Index of a matrix element. (x,y,z) are the desired coodrinates,
// mx is the size of the matrix
inline size_t matrix_index_3d(int x, int y, int z, size_t rim) {
	// TOTAL size. THIS IS POTENTIALLY DANGEROUS!!
	const size_t mx = get_global_size(0);
	const size_t my = get_global_size(1);
	
	x += rim;
	y += rim;
	z += rim;
	
	//const size_t mz = size[2]+2*rim;
	size_t result = (z)*(mx*my) + (y) * mx + (x);
#if BOUNDS_CHECK == 1
	const size_t mz = get_global_size(1);
	const size_t total = mx*my*mz;
	
	if(result < 0)
		printf("FlexCL_Matrix:: index < 0\n");
	if(result >= total)
		printf("FlexCL_Matrix:: index >= total\n");
	
	if(x < 0) printf("FlexCL_Matrix::x < 0\n");
	if(y < 0) printf("FlexCL_Matrix::y < 0\n");
	if(z < 0) printf("FlexCL_Matrix::z < 0\n");
	if(x>mx)  printf("FlexCL_Matrix::x > mx\n");
	if(y>my)  printf("FlexCL_Matrix::y > my\n");
	if(z>mz)  printf("FlexCL_Matrix::z > mz\n");
	
	//printf("index(%2d,%2d,%2d) = %d     < %d\n", x-rim,y-rim,z-rim, result, total);
#endif
	return result;
}




/** Reduction on a double-buffered local array set.
 * */
__kernel void reduction_local( __global const REAL *dA, __local REAL *prods, size_t size, size_t offset, __global REAL *dest ) {
#if VERBOSE == 1
	printf("KERNEL::Reduction_local\n");
#endif
	const int gid = get_global_id( 0 );		// Global ID of the array
	const int tnum = get_local_id( 0 ); 		// Local thread number
	const int wgNum = get_group_id( 0 ); 		// Work-group number
	const int numItems = get_local_size(0);
	
	
	if(gid >= size)
		prods[tnum] = 0.0;
	else
		prods[tnum] = dA[gid+offset];
	
	for(int offset = 1; offset< numItems; offset *=2 )
	{
		const int mask = 2*offset -1;
		barrier(CLK_LOCAL_MEM_FENCE);
		if((tnum&mask)==0)
			prods[tnum] += prods[tnum + offset];
	}
	
	// Done
	barrier(CLK_LOCAL_MEM_FENCE);
	if(tnum ==0)
		dest[wgNum] = prods[0];
	
#if VERBOSE == 1
	printf("KERNEL::Reduction_local completed\n");
#endif
}



/** Maximum Reduction on a double-buffered local array set.
 * */
__kernel void reduction_local_max( __global const REAL *dA, __local REAL *prods, size_t size, size_t offset, __global REAL *dest ) {
#if VERBOSE == 1
	printf("KERNEL::Reduction_local_max\n");
#endif
	const int gid = get_global_id( 0 );		// Global ID of the array
	const int tnum = get_local_id( 0 ); 		// Local thread number
	const int wgNum = get_group_id( 0 ); 		// Work-group number
	const int numItems = get_local_size(0);
	
	
	if(gid >= size)
		prods[tnum] = 0.0;
	else
		prods[tnum] = dA[gid+offset];
	
	for(int offset = 1; offset< numItems; offset *=2 )
	{
		const int mask = 2*offset -1;
		barrier(CLK_LOCAL_MEM_FENCE);
		if((tnum&mask)==0)
			prods[tnum] = max(fabs(prods[tnum]), fabs(prods[tnum + offset]));
	}
	
	// Done
	barrier(CLK_LOCAL_MEM_FENCE);
	if(tnum ==0)
		dest[wgNum] = prods[0];
	
#if VERBOSE == 1
	printf("KERNEL::Reduction_local_max completed\n");
#endif
}


/* ==== Array operations ================================================================ */

__kernel void array_set(__global REAL* array, size_t offset, double value) {
	const int gid = get_global_id(0);
	array[gid + offset] = value;
}

__kernel void array_abs(__global REAL* array, size_t offset) {
	const int gid = get_global_id(0);
	array[gid + offset] = fabs(array[gid + offset]);
}

__kernel void array_add(__global REAL* a1, REAL value, __global REAL* dst, size_t size) {
	const int id = get_global_id(0);
	REAL v = a1[id] + value;
#if ARITHMETIC_GLOBAL_MEM_FENCE == 1
	barrier(CLK_GLOBAL_MEM_FENCE);
#endif
	dst[id] = v;
}
__kernel void array_sub(__global REAL* a1, REAL value, __global REAL* dst, size_t size) {
	const int id = get_global_id(0);
	REAL v = a1[id] - value;
#if ARITHMETIC_GLOBAL_MEM_FENCE == 1
	barrier(CLK_GLOBAL_MEM_FENCE);
#endif
	dst[id] = v;
}
__kernel void array_mul(__global REAL* a1, REAL value, __global REAL* dst, size_t size) {
	const int id = get_global_id(0);
	REAL v = a1[id] * value;
#if ARITHMETIC_GLOBAL_MEM_FENCE == 1
	barrier(CLK_GLOBAL_MEM_FENCE);
#endif
	dst[id] = v;
}
__kernel void array_div(__global REAL* a1, REAL value, __global REAL* dst, size_t size) {
	const int id = get_global_id(0);
	REAL v = a1[id] / value;
#if ARITHMETIC_GLOBAL_MEM_FENCE == 1
	barrier(CLK_GLOBAL_MEM_FENCE);
#endif
	dst[id] = v;
}

__kernel void array_array_add(__global REAL* a1, __global REAL* a2, __global REAL* dst, size_t size) {
	const int id = get_global_id(0);
	REAL v = a1[id] + a2[id];
#if ARITHMETIC_GLOBAL_MEM_FENCE == 1
	barrier(CLK_GLOBAL_MEM_FENCE);
#endif
	dst[id] = v;
}
__kernel void array_array_sub(__global REAL* a1, __global REAL* a2, __global REAL* dst, size_t size) {
	const int id = get_global_id(0);
	REAL v = a1[id] - a2[id];
#if ARITHMETIC_GLOBAL_MEM_FENCE == 1
	barrier(CLK_GLOBAL_MEM_FENCE);
#endif
	dst[id] = v;
}
__kernel void array_array_mul(__global REAL* a1, __global REAL* a2, __global REAL* dst, size_t size) {
	const int id = get_global_id(0);
	REAL v = a1[id] * a2[id];
#if ARITHMETIC_GLOBAL_MEM_FENCE == 1
	barrier(CLK_GLOBAL_MEM_FENCE);
#endif
	dst[id] = v;
}
__kernel void array_array_div(__global REAL* a1, __global REAL* a2, __global REAL* dst, size_t size) {
	const int id = get_global_id(0);
	REAL v = a1[id] / a2[id];
#if ARITHMETIC_GLOBAL_MEM_FENCE == 1
	barrier(CLK_GLOBAL_MEM_FENCE);
#endif
	dst[id] = v;
}


__kernel void array_array_add_mul(__global REAL* a1, __global REAL* a2, __global REAL* dst, size_t size, REAL factor) {
	const int id = get_global_id(0);
	REAL v = a1[id] + (a2[id]*factor);
#if ARITHMETIC_GLOBAL_MEM_FENCE == 1
	barrier(CLK_GLOBAL_MEM_FENCE);
#endif
	dst[id] = v;
}
__kernel void array_array_sub_mul(__global REAL* a1, __global REAL* a2, __global REAL* dst, size_t size, REAL factor) {
	const int id = get_global_id(0);
	REAL v = a1[id] - (a2[id]*factor);
#if ARITHMETIC_GLOBAL_MEM_FENCE == 1
	barrier(CLK_GLOBAL_MEM_FENCE);
#endif
	dst[id] = v;
}


/* ==== Matrix operations =============================================================== */

__kernel void matrix_set_rim(__global REAL* matrix, size_t rim, double value) {
#if VERBOSE == 1
	printf("KERNEL::Matrix set rim\n");
#endif
	const int x = get_global_id(0)-rim;
	const int y = get_global_id(1)-rim;
	const int z = get_global_id(2)-rim;
	const int mx = get_global_size(0)-2*rim;
	const int my = get_global_size(1)-2*rim;
	const int mz = get_global_size(2)-2*rim;
	const size_t index = matrix_index_3d(x,y,z,rim);
	
	
	bool isRim = (x < 0 || y < 0 || z < 0) || (x >= mx || y >= my || z >= mz );

	if(isRim) {
		matrix[index] = value;
		/*
		if(y < my && z < mz && y >= 0 && z >= 0)
			printf("rim[%2d,%2d,%2d,%2d,%2d,%2d, %4ld] = %lf\n",x,y,z,mx,my,mz,index, value);
		*/
	}
#if VERBOSE == 1
	printf("KERNEL::Matrix-set-rim done\n");
#endif
}


/** Matrix multiplication and set the RIM to a given value  */
__kernel void matrix_matrix_mul_rim_3d(__global REAL* m1, __global REAL* m2, __global REAL* dst, size_t rim, double value_rim) {
	const int x = get_global_id(0)-rim;
	const int y = get_global_id(1)-rim;
	const int z = get_global_id(2)-rim;
	const int mx = get_global_size(0)-2*rim;
	const int my = get_global_size(1)-2*rim;
	const int mz = get_global_size(2)-2*rim;
	const size_t index = matrix_index_3d(x,y,z,rim);
	
	
	bool isRim = (x < 0 || y < 0 || z < 0) || (x >= mx || y >= my || z >= mz );
	REAL value;
	if(isRim) {
		value = value_rim;
	} else {
		REAL a1 = m1[index];
		REAL a2 = m2[index];
		value = a1*a2;
	}
	dst[index] = value;
}


/** Multiply two matrices m1,m2 and store the result in dst.
 * Hint: dst can also be one of the given matrices.
 * */
__kernel void matrix_matrix_mul_3d(__global REAL* m1, __global REAL* m2, __global REAL* dst, size_t rim) {
#if VERBOSE == 1
	printf("KERNEL::Matrix-Matrix multiplication\n");
#endif
	const int x = get_global_id(0)-rim;
	const int y = get_global_id(1)-rim;
	const int z = get_global_id(2)-rim;
	const int index = matrix_index_3d(x,y,z,rim);
	
	REAL a1 = m1[index];
	REAL a2 = m2[index];
#if ARITHMETIC_GLOBAL_MEM_FENCE == 1
	barrier(CLK_GLOBAL_MEM_FENCE);
#endif
	dst[index] = a1 * a2;
#if VERBOSE == 1
	printf("KERNEL::Matrix-Matrix done\n");
#endif
}


/** Multiply matrices m1 with factor and store result in dst
 * Hint: dst can also be one of the given matrices.
 * */
__kernel void matrix_mul_3d(__global REAL* m1, __global REAL* dst, REAL factor, size_t rim) {
	const int x = get_global_id(0)-rim;
	const int y = get_global_id(1)-rim;
	const int z = get_global_id(2)-rim;
	const int index = matrix_index_3d(x,y,z,rim);
	
	REAL a = m1[index] * factor;
#if ARITHMETIC_GLOBAL_MEM_FENCE == 1
	barrier(CLK_GLOBAL_MEM_FENCE);
#endif
	dst[index] = a;
}

/**
 * Adds a given summand to all values of m1, store the result to dst
 */
__kernel void matrix_add_3d(__global REAL* m1, __global REAL* dst, REAL summand, size_t rim) {
	const int x = get_global_id(0)-rim;
	const int y = get_global_id(1)-rim;
	const int z = get_global_id(2)-rim;
	const int index = matrix_index_3d(x,y,z,rim);
	
	REAL a = m1[index] + summand;
#if ARITHMETIC_GLOBAL_MEM_FENCE == 1
	barrier(CLK_GLOBAL_MEM_FENCE);
#endif
	dst[index] = a;
}

/** Add two matrices m1,m2 and store the result in dst.
 * Hint: dst can also be one of the given matrices.
 * */
__kernel void matrix_matrix_add_3d(__global REAL* m1, __global REAL* m2, __global REAL* dst, size_t rim) {
	const int x = get_global_id(0)-rim;
	const int y = get_global_id(1)-rim;
	const int z = get_global_id(2)-rim;
	const int index = matrix_index_3d(x,y,z,rim);
	
	REAL a1 = m1[index];
	REAL a2 = m2[index];
#if ARITHMETIC_GLOBAL_MEM_FENCE == 1
	barrier(CLK_GLOBAL_MEM_FENCE);
#endif
	dst[index] = a1 + a2;
}

/** Add two matrices m1,m2 and store the result in dst.
 * Hint: dst can also be one of the given matrices.
 * */
__kernel void matrix_matrix_add_mul_3d(__global REAL* m1, __global REAL* m2, __global REAL* dst, size_t rim, REAL factor) {
	const int x = get_global_id(0)-rim;
	const int y = get_global_id(1)-rim;
	const int z = get_global_id(2)-rim;
	const int index = matrix_index_3d(x,y,z,rim);
	
	REAL a1 = m1[index];
	REAL a2 = m2[index] * factor;
#if ARITHMETIC_GLOBAL_MEM_FENCE == 1
	barrier(CLK_GLOBAL_MEM_FENCE);
#endif
	dst[index] = a1 + a2;
}


/**
 * Adds a given summand to all values of m1, store the result to dst
 */
__kernel void matrix_sub_3d(__global REAL* m1, __global REAL* dst, REAL summand, size_t rim) {
	const int x = get_global_id(0)-rim;
	const int y = get_global_id(1)-rim;
	const int z = get_global_id(2)-rim;
	const int index = matrix_index_3d(x,y,z,rim);
	
	REAL a = m1[index] - summand;
#if ARITHMETIC_GLOBAL_MEM_FENCE == 1
	barrier(CLK_GLOBAL_MEM_FENCE);
#endif
	dst[index] = a;
}

/** Add two matrices m1,m2 and store the result in dst.
 * Hint: dst can also be one of the given matrices.
 * */
__kernel void matrix_matrix_sub_3d(__global REAL* m1, __global REAL* m2, __global REAL* dst, size_t rim) {
	const int x = get_global_id(0)-rim;
	const int y = get_global_id(1)-rim;
	const int z = get_global_id(2)-rim;
	const int index = matrix_index_3d(x,y,z,rim);
	
	REAL a1 = m1[index];
	REAL a2 = m2[index];
#if ARITHMETIC_GLOBAL_MEM_FENCE == 1
	barrier(CLK_GLOBAL_MEM_FENCE);
#endif
	dst[index] = a1 - a2;
}

/** Add two matrices m1,m2 and store the result in dst.
 * Hint: dst can also be one of the given matrices.
 * */
__kernel void matrix_matrix_sub_mul_3d(__global REAL* m1, __global REAL* m2, __global REAL* dst, size_t rim, REAL factor) {
	const int x = get_global_id(0)-rim;
	const int y = get_global_id(1)-rim;
	const int z = get_global_id(2)-rim;
	const int index = matrix_index_3d(x,y,z,rim);
	
	REAL a1 = m1[index];
	REAL a2 = m2[index] * factor;
#if ARITHMETIC_GLOBAL_MEM_FENCE == 1
	barrier(CLK_GLOBAL_MEM_FENCE);
#endif
	dst[index] = a1 - a2;
}
