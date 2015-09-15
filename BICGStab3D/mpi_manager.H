#ifndef DUMMY_MPI_MANAGER_H
#define DUMMY_MPI_MANAGER_H

#include "mpi.h"
#include "matrix.H"
#include "grid_manager.H"


class mpi_manager_1D {
public:
 	mpi_manager_1D();
	mpi_manager_1D(int ntasks_user, int mx);
 	mpi_manager_1D(const mpi_manager_1D& mpi);
 	void setup(int ntasks_user, int mx);
	void setup(int ntasks_user, int coords,
 	           int rank, int ntasks,
 	           int left, int right,
 	           MPI_Comm comm1d);
 	grid_manager make_LocalGrid(grid_manager &GlobalGrid);
 	int get_rank() const;
 	int get_left() const;
 	int get_right() const;
	int get_ntasks() const;
	int get_coord() const;
	int get_OtherRankAbs(int distance) const;
	int get_OtherRankRel(int distance) const;
	int get_OtherRankRel_cyclic(int distance) const;
 	bool is_OuterBoundary(int boundary) const;
	void do_MPISendRecv(NumMatrix<double,2> &buff, int Destination);
	void do_MPISendRecv(NumMatrix<double,2> &buff, int Source,
	                    int Destination);
	void Finalise();
private:
	NumArray<int> AllRanks;
	NumMatrix<int,1> Neighbours, NeighboursCyclic;
	void determin_OtherRanks();
 	int coords;
 	int left, right;
	int rank, ntasks, ntasks_user;
	MPI_Comm comm1d;
	int DIM;
};

class mpi_manager_2D {
public:
	mpi_manager_2D(NumArray<int> &nproc, NumArray<int> &mx);
	mpi_manager_2D(const mpi_manager_2D& mpi);
	mpi_manager_2D();
	mpi_manager_1D boil_down(int direction);
	void setup(NumArray<int> &nproc, NumArray<int> &mx);
	void setup(int nproc_x, int nproc_y,
	           int coords_x, int coords_y,
	           int rank, int ntasks,
	           int left, int right,
	           int front, int back,
	           MPI_Comm comm2d);
	grid_manager make_LocalGrid(grid_manager &GlobalGrid);
	int get_rank() const;
	int get_left() const;
	int get_right() const;
	int get_front() const;
	int get_back() const;
	int get_nproc(int dir) const;
	int get_coord(int dir) const;
	int get_OtherRankAbs(int dir, int dist) const;
	int get_OtherRankRel(int dir, int dist) const;
	int get_OtherRankRel_cyclic(int dir, int dist) const;
	bool is_OuterBoundary(int boundary) const;
	void Finalise();
	void determin_OtherRanks();
	void get_SubComms();
	void do_MPISendRecv(NumMatrix<double,2> &buff, int Destination);
	void do_MPISendRecv(NumMatrix<double,2> &buff, int Source,
	                    int Destination);
	MPI_Comm comm2d;
private:
	MPI_Comm comm_line_x, comm_line_y;
	int rank_line_x, rank_line_y;
	NumMatrix<int,2> AllRanks, Neighbours, NeighboursCyclic;
	int nproc[2], coords[2];
	int left, right;
	int front,back;
	int rank, ntasks;
	int DIM;
};

class mpi_manager_3D {
public:
	mpi_manager_3D(NumArray<int> &nproc, NumArray<int> &mx);
	mpi_manager_3D(const mpi_manager_3D& mpi);
	mpi_manager_3D();
	void setup(NumArray<int> &nproc, NumArray<int> &mx);
	mpi_manager_2D boil_down(int plane_normal);
	grid_manager make_LocalGrid(grid_manager &GlobalGrid);
	grid_1D make_LocalGrid1D(grid_1D &GlobalGrid1D, int dir);
	int get_rank() const;
	int get_left() const;
	int get_right() const;
	int get_front() const;
	int get_back() const;
	int get_top() const;
	int get_bottom() const;
	int get_coord(int dir) const;
	int get_nproc(int dir) const;
	int get_OtherRankAbs(int ix, int iy, int iz) const;
	bool is_OuterBoundary(int boundary) const;
	void Finalise();
	MPI_Comm comm3d;
	MPI_Comm comm_plane_xy;
	MPI_Comm comm_plane_xz;
	MPI_Comm comm_plane_yz;

private:
	NumMatrix<int,3> AllRanks, Neighbour;
	int nproc[3], coords[3];
	int ntasks;
	int left, right;
	int front,back;
	int top,bottom;
	int rank;
	int rank_plane_xy, rank_plane_xz, rank_plane_yz;
	int DIM;
};



#endif
