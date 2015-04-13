#ifndef DUMMY_BOUNDARY_HANDLER
#define DUMMY_BOUNDARY_HANDLER

#include <vector>
#include "matrix.H"

#include "grid_manager.H"
#ifdef parallel
#include "mpi_manager.H"
#endif


class BoundaryHandler {
public:
// #else
	BoundaryHandler(int DIM, int bcType=1, int test_case=-1);
// #endif
	// bc for momentum space
	void do_BCsCellAve(NumMatrix<double,1> &spec,
	                   const grid_1D &TheGrid, int rim,
	                   double &S_lower, double &S_higher,
	                   double delt=1., int type=0) const;
	void set_bcType(int bcType);
	int get_bcType(int dir);
protected:
	// std::vector<int> bc_Type;
	NumArray<int> bc_Type;
	double eps;
	int DIM;
	int test_case;
	bool is_testRun;
};

class BoundaryHandler2D : public BoundaryHandler {
public:
#ifdef parallel
	BoundaryHandler2D(mpi_manager_2D MyMPI, int type=1);
#endif
	BoundaryHandler2D(int type=1);
	// bc for configuration space
	void do_BCs(NumMatrix<double,2> &dist, int rim, int dir=-1,
	            bool keepBoundVals=false);
	void do_BCs(NumMatrix<double,2> &dist,
	            NumMatrix<double,1> &bcVals_xb,
	            NumMatrix<double,1> &bcVals_xe,
	            NumMatrix<double,1> &bcVals_yb,
	            NumMatrix<double,1> &bcVals_ye,
	            int rim, int dir=-1,
	            bool keepBoundVals=false, bool use_ExtBCVal = false);
#ifdef parallel
	void connect_CyclicSystems();
#endif
private:
#ifdef parallel
	void do_bc_MPI(NumMatrix<double,2> &data, NumArray<int> &mx,
	               int dir2D, int rim);
	inline void do_MpiSendRecv(NumMatrix<double,2> &SendBuff3D,
	                           NumMatrix<double,2> &RecvBuff3D,
	                           int from, int into, int size,
	                           bool Send, bool Recv);
	inline void do_MpiSendRecv(NumMatrix<double,1> &SendBuff3D,
	                           NumMatrix<double,1> &RecvBuff3D,
	                           int from, int into, int size,
	                           bool Send, bool Recv);

	mpi_manager_2D MyMPI;
	NumMatrix<double,1> SendBuff1D, RecvBuff1D;
	NumMatrix<double,2> SendBuff2D, RecvBuff2D;
#endif

};

class BoundaryHandler3D : public BoundaryHandler {
public:
#ifdef parallel
	BoundaryHandler3D(mpi_manager_3D MyMPI, int type=1);
#endif
	BoundaryHandler3D(int type=1);
	void do_BCs(NumMatrix<double,3> &dist, int rim, int dir=-1,
	            bool keepBoundVals=false);
private:
#ifdef parallel
	void do_bc_MPI(NumMatrix<double,3> &data, NumArray<int> &mx,
	               int dir, int rim);
	inline void do_MpiSendRecv(NumMatrix<double,3> &SendBuff3D,
	                           NumMatrix<double,3> &RecvBuff3D,
	                           int from, int into, int size,
	                           bool Send, bool Recv);
	inline void do_MpiSendRecv(NumMatrix<double,2> &SendBuff2D,
	                           NumMatrix<double,2> &RecvBuff2D,
	                           int from, int into, int size,
	                           bool Send, bool Recv);
	mpi_manager_3D MyMPI;
	NumMatrix<double,2> SendBuff2D, RecvBuff2D;
	NumMatrix<double,3> SendBuff3D, RecvBuff3D;
#endif
};


class BoundaryHandler4D : public BoundaryHandler {
public:
#ifdef parallel
	BoundaryHandler4D(mpi_manager_3D MyMPI);
#endif
	BoundaryHandler4D();
	void do_BCs(NumMatrix<double,4> &dist, int rim, int dir=-1,
	            bool keepBoundVals=false);
private:
#ifdef parallel
	void do_bc_MPI(NumMatrix<double,4> &data, NumArray<int> &mx,
	               int dir, int rim, int rimP);
	void do_MpiSendRecv(NumMatrix<double,4> &SendBuff4D,
	                    NumMatrix<double,4> &RecvBuff4D,
	                    int from, int into, int size,
	                    bool Send, bool Recv);
	void do_MpiSendRecv(NumMatrix<double,3> &SendBuff3D,
	                    NumMatrix<double,3> &RecvBuff3D,
	                    int from, int into, int size,
	                    bool Send, bool Recv);
	mpi_manager_3D MyMPI;
	NumMatrix<double,3> SendBuff3D, RecvBuff3D;
	NumMatrix<double,4> SendBuff4D, RecvBuff4D;
#endif
};

#endif
