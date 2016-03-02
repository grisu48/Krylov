/*
 * main.hpp
 * Additional HEADER file for the BiCGStab_CL solver
 */

#ifndef _BICGSTAB_MAIN_HPP_
#define _BICGSTAB_MAIN_HPP_

/* OpenCL context type */
#define OCL_CONTEXT_DEFAULT 0
#define OCL_CONTEXT_CPU 1
#define OCL_CONTEXT_GPU 2
#define OCL_CONTEXT_ACCL 3


/* Test cases */
#define TEST_ONE 1
#define TEST_TWO 2
#define TEST_THREE 3
#define TEST_FOUR 4
#define TEST_FIVE 5

/* Run modi */
#define RUN_NORMAL 0x1			// Normal run
#define RUN_STATS 0x2			// Statistic run (limited but normalized output)


#endif /* MAIN_HPP_ */
