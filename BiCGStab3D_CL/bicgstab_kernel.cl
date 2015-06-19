
/* =============================================================================
 *
 * Title:         BiCGStab 3d OpenCL accelerated Kernel file (File for OpenCL)
 * Author:        Felix Niederwanger
 * Description:   This is a OpenCL program file and contains all functions for
 *                the BiCGStab solver
 * =============================================================================
 */

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define REAL double
#define ssize_t long
#define size_t unsigned long

// Range check for the matrix index. Turn off in productive code to increase performance, especially on GPU!
#define RANGE_CHECK 0



/******************************************************************************
 * Important notes on the kernel:                                             *
 * Keep in mind, that the matrix has one ghost cell in each dimension!        *
 * It means that most operations need also an index shift of 1.               *
 ******************************************************************************
 */




// Index of a matrix element. (x,y,z) are the desired coodrinates,
// mx is the size of the matrix
inline int matrix_index(int x, int y, int z, size_t mx, size_t my, size_t mz) {
#if RANGE_CHECK == 1
	if(x < 0 || x >= mx) printf("ASSERTION FAILED: (x>0 && x<=mx)");
	if(y < 0 || y >= my) printf("ASSERTION FAILED: (y>0 && y<=my)");
	if(z < 0 || z >= mz) printf("ASSERTION FAILED: (z>0 && z<=mz)");
#endif
	return z*(mx*my) + y * mx + x;
}

inline REAL sqr(REAL x) { return x*x; }


__kernel void boundary(__global REAL* matrix, size_t mx, size_t my, size_t mz, size_t rim) {
	// NOTE: Here mx = mx (NOT mx + 2*rim!!)
	
	// Since the BiCGStab kernel is transparent for RIM cells, we have to add them here.
	const ssize_t x = get_global_id(0) - rim;
	const ssize_t y = get_global_id(1) - rim;
	const ssize_t z = get_global_id(2) - rim;
	
	
	// XXX: OpenCL Divergence in x and z direction :-(
	bool _isBoundary = (x <= 0 || y <= 0 || z <= 0) || (x >= mx || y >= my || z >= mz);
	if(_isBoundary) {
		matrix[matrix_index(x+rim,y+rim,z+rim,mx+2*rim,my+2*rim,mz+2*rim)] = 0.0;
	}
}



__kernel void generateAx_Full(__global REAL* psi, __global REAL* lambda, __global REAL* Dxx, __global REAL* Dyy, __global REAL* Dzz, __global REAL* Dxy, __global REAL* dst, size_t mx, size_t my, size_t mz, size_t rim, REAL deltaX, REAL deltaY, REAL deltaZ) {
	const int x = get_global_id(0)+rim;
	const int y = get_global_id(1)+rim;
	const int z = get_global_id(2)+rim;
	const int index = matrix_index(x,y,z,mx,my,mz);
	REAL result = 0.0;
	
	const REAL coeff[3] = {1.0/sqr(deltaX), 1.0/sqr(deltaY), 1.0/sqr(deltaZ) };
	const REAL coeff_xy = 1.0 / (2.0 * coeff[0] * coeff[1]);
	
	
	// Build matrix
	result  = coeff[0] * Dxx[index] * (psi[matrix_index(x+1,y,z,mx,my,mz)] + psi[matrix_index(x-1,y,z,mx,my,mz)]);
	result += coeff[1] * Dyy[index] * (psi[matrix_index(x,y+1,z,mx,my,mz)] + psi[matrix_index(x,y-1,z,mx,my,mz)]);
	result += coeff[2] * Dzz[index] * (psi[matrix_index(x,y,z+1,mx,my,mz)] + psi[matrix_index(x,y,z+1,mx,my,mz)]);

	result -= ( 2.0 * (coeff[0] * Dxx[index] + coeff[1] * Dyy[index] + coeff[2] * Dzz[index] ));
	result += lambda[index] * psi[index];
	result += coeff_xy * Dxy[index] * ( psi[matrix_index(x+1,y+1,z,mx,my,mz)] - psi[matrix_index(x+1,y-1,z,mx,my,mz)] + psi[matrix_index(x-1,y-1,z,mx,my,mz)] - psi[matrix_index(x-1,y+1,z,mx,my,mz)]);
	
	REAL tmp;
	
	tmp  = (Dxx[matrix_index(x+1,y,z,mx,my,mz)] - Dxx[matrix_index(x-1,y,z,mx,my,mz)]) / (2.0 * deltaX);
	tmp += (Dxy[matrix_index(x,y+1,z,mx,my,mz)] - Dxy[matrix_index(x,y-1,z,mx,my,mz)]) / (2.0 * deltaY);
	tmp *= (psi[matrix_index(x+1,y,z,mx,my,mz)] - psi[matrix_index(x-1,y,z,mx,my,mz)]) / (2.0 * deltaX);
	result += tmp;
	
	tmp  = (Dyy[matrix_index(x,y+1,z,mx,my,mz)] - Dyy[matrix_index(x,y-1,z,mx,my,mz)]) / (2.0 * deltaY);
	tmp += (Dxy[matrix_index(x+1,y,z,mx,my,mz)] - Dxy[matrix_index(x-1,y,z,mx,my,mz)]) / (2.0 * deltaX);
	tmp *= (psi[matrix_index(x,y+1,z,mx,my,mz)] - psi[matrix_index(x,y-1,z,mx,my,mz)]) / (2.0 * deltaY);
	result += tmp;
	
	tmp  = (Dzz[matrix_index(x,y,z+1,mx,my,mz)] - Dzz[matrix_index(x,y,z-1,mx,my,mz)]) / (2.0 * deltaZ);
	tmp *= (psi[matrix_index(x,y,z+1,mx,my,mz)] - psi[matrix_index(x,y,z-1,mx,my,mz)]) / (2.0 * deltaZ);
	result += tmp;
	
	// Done
	dst[index] = result;
}

__kernel void generateAx_NoSpatial(__global REAL* psi, __global REAL* lambda, __global REAL* dst, size_t mx, size_t my, size_t mz, size_t rim, REAL deltaX, REAL deltaY, REAL deltaZ, REAL diffDiagX, REAL diffDiagY, REAL diffDiagZ) {
	// NOTE: Here mx = mx + 2*rim, same goes for my and mz!
	
	// Since the BiCGStab kernel is transparent for RIM cells, we have to add them here.
	const ssize_t x = get_global_id(0)+rim;
	const ssize_t y = get_global_id(1)+rim;
	const ssize_t z = get_global_id(2)+rim;
	const size_t index = matrix_index(x,y,z,mx,my,mz);
	REAL result = 0.0;
	
	// 2015-06-19: Coefficient checked.
	const REAL coeff[3] = { diffDiagX/sqr(deltaX), diffDiagY/sqr(deltaY), diffDiagZ/sqr(deltaZ) };
	
	
	// Build matrix
	result += coeff[0] * (psi[matrix_index(x+1,y  ,z   ,mx,my,mz)] + psi[matrix_index(x-1,y  ,z  ,mx,my,mz)]);
	result += coeff[1] * (psi[matrix_index(x  ,y+1,z   ,mx,my,mz)] + psi[matrix_index(x  ,y-1,z  ,mx,my,mz)]);
	result += coeff[2] * (psi[matrix_index(x  ,   y,z+1,mx,my,mz)] + psi[matrix_index(x  ,y  ,z-1,mx,my,mz)]);

	result -= ( 2.0 * (coeff[0] + coeff[1] + coeff[2]) + lambda[index]) * psi[index];
	
	// Done
	dst[index] = result;
}