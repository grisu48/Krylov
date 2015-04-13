#include "mpi_manager.H"
#include <iostream>
#include <stdlib.h> 


mpi_manager_3D::mpi_manager_3D() {
	// Determine the rank of the current task
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	DIM = 3;
}

mpi_manager_3D::mpi_manager_3D(NumArray<int> &nproc, NumArray<int> &mx) {
	DIM = 3;
	for(int dir=0; dir<DIM; ++dir) {
		this->nproc[dir] = nproc[dir];
	}
	setup(nproc, mx);
}


mpi_manager_3D::mpi_manager_3D(const mpi_manager_3D &mpi) {
	//! Copy constructor
	/*! Here we just copy all the parameters
	 */
	DIM = mpi.DIM;
	for(int dir=0; dir<DIM; ++dir) {
		nproc[dir] = mpi.nproc[dir];
		coords[dir] = mpi.coords[dir];
	}

	ntasks = mpi.ntasks;
	left = mpi.left;
	right = mpi.right;
	front = mpi.front;
	back = mpi.back;
	top = mpi.top;
	bottom = mpi.bottom;

	// Communucators:
	comm3d = mpi.comm3d;
	comm_plane_xy = mpi.comm_plane_xy;
	comm_plane_xz = mpi.comm_plane_xz;
	comm_plane_yz = mpi.comm_plane_yz;
	
	// Different rank definitions:
	rank = mpi.rank;
	rank_plane_xy = mpi.rank_plane_xy;
	rank_plane_xz = mpi.rank_plane_xz;
	rank_plane_yz = mpi.rank_plane_yz;

	AllRanks = mpi.AllRanks;
	Neighbour = mpi.Neighbour;

}

void mpi_manager_3D::setup(NumArray<int> &nproc, NumArray<int> &mx) {
	
	// Save number of processors in each dimension
	for(int dir=0; dir<DIM; ++dir) {
		this->nproc[dir] = nproc[dir];
	}

	// Determine the rank of the current task
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// Get number of ranks from MPI
	int ntasks;
	MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
	this->ntasks = ntasks;

	// Set the distribution of processes:
	if(ntasks != nproc[0]*nproc[1]*nproc[2]){
		std::cerr << " Wrong number of processes " << std::endl;
		std::cout << ntasks << " " << nproc[0]*nproc[1]*nproc[2] << std::endl;
		Finalise();
	}

	if(rank==0) {
		std::cout << " Number of tasks: " << ntasks << std::endl;
	}

	// Check if grid can be subdevided as desired
	for(int dir = 0; dir < DIM; ++dir) {
		if(mx[dir] < nproc[dir] && nproc[dir] > 1) {
			if(rank == 0) {
				std::cerr << " Wrong grid topology for dimension ";
				std::cerr << dir << std::endl;
				std::cerr << "  mx[" << dir << "]:" << mx[dir] << std::endl;
				std::cerr << " nproc[" << dir << "]:" << nproc[dir] << std::endl;
			}
			Finalise();
		}
	}

	// Check if grid is a power of 2:
	double eps = 1.e-12;
	for(int dir = 0; dir < DIM; ++dir) {
		double exponent = log(mx[dir])/log(2.);
		int i_exponent = static_cast<int>(exponent+eps);

		if(exponent - i_exponent > 2.*eps) {
			if(rank == 0) {
				std::cerr << " Error: grid must be of the form mx = 2^n ";
				std::cerr << std::endl;
				std::cerr << " Exiting " << std::endl;
			}
			Finalise();
		}
	}

	// Grid is not periodic
	int periods[3] = {false, false, false};
	int reorder = false;
	// If all is okay: Create new communicator "comm3d"  
	MPI_Cart_create(MPI_COMM_WORLD, DIM, nproc, periods, reorder, &comm3d);

	// Retrieve the cartesian topology
	if (rank == 0) {
		int TopoType;
		std::cout << " Cart topology:  ";
		MPI_Topo_test(comm3d, &TopoType);
		switch (TopoType) {
		case MPI_UNDEFINED : 
			std::cout << " MPI_UNDEFINED " << std::endl;
			break;
		case MPI_GRAPH     :
			std::cout << "MPI_GRAPH" << std::endl;
			break;
		case MPI_CART      :
			std::cout << "MPI_CART" << std::endl;
			break;
		}
	}
	
	//   Determine rank again for cartesian communicator -> overwrite rank
	MPI_Comm_rank(comm3d, &rank);

	// std::cout << " my rank: " << rank << std::endl;

	// Translate rank to coordinates
	MPI_Cart_coords(comm3d, rank, DIM, coords);

	// // Backwards translation
	// int TranslateRank;
	// MPI_Cart_rank(comm3d, coords, &TranslateRank);

	// Find neighbouring ranks
	// Syntax: comm3d, shift direction, displacement, source, destination
	MPI_Cart_shift(comm3d, 0, 1, &left , &right);
	MPI_Cart_shift(comm3d, 1, 1, &front, &back);
	MPI_Cart_shift(comm3d, 2, 1, &bottom, &top);

	// std::cout << " My rank " << rank << " " << left << " " << right << " " << front << " " << back << " " << bottom << " " << top << std::endl;
	if(rank==0) {
		std::cout << " nearby " << right << " " << back << " " << top << std::endl;
	}

	// Determine ranks of neighbour processes:
	int shiftcoord[DIM];
	int lbound[DIM],ubound[DIM];
	for(int dim=0;dim<DIM;dim++){
		lbound[dim]=-1;
		ubound[dim]= 1;
	}
	Neighbour.resize(lbound,ubound);
	Neighbour.clear();

	for(int dim0=-1; dim0<=1; dim0++){
		shiftcoord[0] = (coords[0]+dim0)%nproc[0];
		if(shiftcoord[0] < 0) shiftcoord[0]+=nproc[0];
		for(int dim1=-1; dim1<=1; dim1++){
			shiftcoord[1] = (coords[1]+dim1)%nproc[1];
			if(shiftcoord[1] < 0) shiftcoord[1]+=nproc[1];
			for(int dim2=-1; dim2<=1; dim2++){
				shiftcoord[2] = (coords[2]+dim2)%nproc[2];
				if(shiftcoord[2] < 0) shiftcoord[2]+=nproc[2];
				MPI_Cart_rank(comm3d, shiftcoord,&Neighbour(dim0,dim1,dim2));
			}
		}
	}
	
	// if(rank==1) {
	// 	for(int dim0=-1; dim0<=1; dim0++){
	// 		for(int dim1=-1; dim1<=1; dim1++){
	// 			for(int dim2=-1; dim2<=1; dim2++){
	// 				std::cout << " neighbour " << dim0 << " " << dim1 << " ";
	// 				std::cout << dim2 << " " << Neighbour(dim0, dim1, dim2);
	// 				std::cout << std::endl;
	// 			}
	// 		}
	// 	}
	// }

	// Determine absolute position of any rank:
	AllRanks.resize(Index::set(0,0,0),
	               Index::set(nproc[0]-1,nproc[1]-1,nproc[2]-1));
	
	for(int dim0=0; dim0<nproc[0]; ++dim0) {
		for(int dim1=0; dim1<nproc[1]; ++dim1) {
			for(int dim2=0; dim2<nproc[2]; ++dim2) {
				int coord[3] = {dim0, dim1, dim2};
				MPI_Cart_rank(comm3d, coord, &AllRanks(dim0, dim1, dim2));
			}
		}
	}

	// if(rank==2) {
	// 	std::cout << " Neigh: " << rank << " "<<Neighbour(0,0,0) << " " << AllRanks(2,0,0) << std::endl;
	// }

	
	// Now make additional mpi groups relating to planes:

	int count(0);
	int num_xy = nproc[0]*nproc[1];
	int num_xz = nproc[0]*nproc[2];
	int num_yz = nproc[1]*nproc[2];
	
	NumMatrix<int,1> x_ranks[nproc[0]];
	NumMatrix<int,1> y_ranks[nproc[1]];
	NumMatrix<int,1> z_ranks[nproc[2]];

	// Walk trough z-axis -- xy plane
	for(int irz=0; irz<nproc[2]; irz++) {
		count = 0;
		z_ranks[irz].resize(Index::set(0), Index::set(num_xy));
		for(int irx=0; irx<nproc[0]; irx++) {
			for(int iry=0; iry<nproc[1]; iry++) {
				z_ranks[irz](count) = AllRanks(irx,iry,irz);
				count++;
			}
		}
	}

	// Walk trough y-axis -- xz plane
	for(int iry=0; iry<nproc[1]; iry++) {
		count = 0;
		y_ranks[iry].resize(Index::set(0), Index::set(num_xz));
		for(int irx=0; irx<nproc[0]; irx++) {
			for(int irz=0; irz<nproc[2]; irz++) {
				y_ranks[iry](count) = AllRanks(irx,iry,irz);
				count++;
			}
		}
	}

	// Walk trough x-axis -- yz plane
	for(int irx=0; irx<nproc[0]; irx++) {
		count = 0;
		x_ranks[irx].resize(Index::set(0), Index::set(num_yz));
		for(int iry=0; iry<nproc[1]; iry++) {
			for(int irz=0; irz<nproc[2]; irz++) {
				x_ranks[irx](count) = AllRanks(irx,iry,irz);
				count++;
			}
		}
	}

	// Build local communicator:
	MPI_Group group_all, group_constz, group_consty, group_constx;
	// Get standard group handle:
	MPI_Comm_group(comm3d, &group_all);


	// Devide tasks into groups based on z-position
	MPI_Group_incl(group_all, num_xy, z_ranks[coords[2]], &group_constz);

	// Devide tasks into groups based on z-position
	MPI_Group_incl(group_all, num_xz, y_ranks[coords[1]], &group_consty);

	// Devide tasks into groups based on x-position
	MPI_Group_incl(group_all, num_yz, x_ranks[coords[0]], &group_constx);

	// // Make corresponding communicators:
	// MPI_Comm_create(comm3d, group_constz, &comm_plane_xy); // const z
	// MPI_Comm_create(comm3d, group_consty, &comm_plane_xz); // const x
	// MPI_Comm_create(comm3d, group_constx, &comm_plane_yz); // const x
	// // Get corresponding rank
	// MPI_Group_rank (group_constz, &rank_plane_xy);
	// MPI_Group_rank (group_consty, &rank_plane_xz);
	// MPI_Group_rank (group_constx, &rank_plane_yz);

	int remain_dims[3];
	// x-y plane:
	remain_dims[0] = 1;
	remain_dims[1] = 1;
	remain_dims[2] = 0;
	MPI_Cart_sub(comm3d, remain_dims, &comm_plane_xy);
	MPI_Comm_rank(comm_plane_xy, &rank_plane_xy);

	// x-z plane
	remain_dims[0] = 1;
	remain_dims[1] = 0;
	remain_dims[2] = 1;
	MPI_Cart_sub(comm3d, remain_dims, &comm_plane_xz);
	MPI_Comm_rank(comm_plane_xz, &rank_plane_xz);

	// y-z plane
	remain_dims[0] = 0;
	remain_dims[1] = 1;
	remain_dims[2] = 1;
	MPI_Cart_sub(comm3d, remain_dims, &comm_plane_yz);
	MPI_Comm_rank(comm_plane_yz, &rank_plane_yz);

}



grid_manager mpi_manager_3D::make_LocalGrid(grid_manager &GlobalGrid) {
	//! Take the global grid and generate a local one
	
	// First get all properties from global grid manager:
	NumArray<int> mx(DIM);
	NumArray<double> xb(DIM), Len(DIM);
	
	for(int dir=0; dir<DIM; ++dir) {
		mx[dir] = GlobalGrid.get_mx(dir);
		xb[dir] = GlobalGrid.get_xb(dir);
		Len[dir] = GlobalGrid.get_Len(dir);
	}

	// Now compute local extent of grid (cell-wise and space-wise)
	if (rank == 0) {
		for (int dir=0; dir<DIM; ++dir) {
			// std::cout << " The grid: " << dir << " " <<mx[dir] << " " << nproc[dir] << std::endl;
			mx[dir] /= nproc[dir];
			Len[dir] /= nproc[dir];
		}
	}

	
	MPI_Barrier(comm3d);
	MPI_Bcast(mx, 3, MPI_INT, 0, comm3d);
	MPI_Bcast(Len, 3, MPI_DOUBLE, 0, comm3d);
	MPI_Bcast(xb, 3, MPI_DOUBLE, 0, comm3d);
	MPI_Barrier(comm3d);

	// Now the local computations that need to be done by each rank:
	NumArray<double> xe(DIM);
	for (int dir=0; dir<DIM; ++dir) {
		xb[dir] += Len[dir]*coords[dir];
		xe[dir] = xb[dir] + Len[dir];
	}

	// if(rank == 2) {
	// 	std::cout << " mx: " << mx[0] << " " << mx[1] << " " << mx[2];
	// 	std::cout << std::endl;
	// 	std::cout << " Len: " << Len[0] << " " << Len[1] << " " << Len[2];
	// 	std::cout << std::endl;
	// 	std::cout << " xb: " << xb[0] << " " << xb[1] << " " << xb[2];
	// 	std::cout << std::endl;
	// 	std::cout << " xe: " << xe[0] << " " << xe[1] << " " << xe[2];
	// 	std::cout << std::endl;
	// }


	int rim = GlobalGrid.get_rim();

	// Now make the local grid manager:
	grid_manager LocalGrid(xb[0], xb[1], xb[2], xe[0], xe[1], xe[2],
	                       mx[0]+1, mx[1]+1, mx[2]+1, rim);
	
	// Now set corresponding boundary types (old type at
	// outer-boundaries / -1 at MPI boundaries)
	for(int bound=0; bound<6; ++bound) {
		if(is_OuterBoundary(bound)) {
			LocalGrid.set_bcType(bound, GlobalGrid.get_bcType(bound));
		} else {
			LocalGrid.set_bcType(bound, -1);
		}
	}

	return LocalGrid;

}


int mpi_manager_3D::get_rank() const {
	return rank;
}

int mpi_manager_3D::get_left() const {
	return left;
}

int mpi_manager_3D::get_right() const {
	return right;
}

int mpi_manager_3D::get_front() const {
	return front;
}

int mpi_manager_3D::get_back() const {
	return back;
}

int mpi_manager_3D::get_top() const {
	return top;
}

int mpi_manager_3D::get_bottom() const {
	return bottom;
}

int mpi_manager_3D::get_coord(int dir) const {
	return coords[dir];
}

int mpi_manager_3D::get_nproc(int dir) const {
	return nproc[dir];
}


int mpi_manager_3D::get_OtherRankAbs(int ix, int iy, int iz) const {
	//! Get rank at some coordinate
	assert(ix>=0 && ix<nproc[0]);
	assert(iy>=0 && iy<nproc[1]);
	assert(iz>=0 && iz<nproc[1]);

	return AllRanks(ix, iy, iz);
	
}


void mpi_manager_3D::Finalise() {
	if(rank==0) {
		std::cerr << " Ending the program " << std::endl;
	}
	MPI_Finalize();
	exit(-5);
}

bool mpi_manager_3D::is_OuterBoundary(int boundary) const {
	//! Check if a certain boundary is at end of domain
	/*!  Check if a local boundary is at the same time the global
	  boundary. Possible boundaries are:
	  0 - lower x-boundary
	  1 - upper x-boundary
	  2 - lower y-boundary
	  3 - upper y-boundary
	  4 - lower z-boundary
	  5 - upper z-boundary
	 */

	assert(boundary>=0 && boundary<6);

	bool at_bound = false;
	switch (boundary) {
	case 0 :
		if(left == MPI_PROC_NULL) {
			at_bound = true;
		}
		break;
	case 1 :
		if(right == MPI_PROC_NULL) {
			at_bound = true;
		}
		break;
	case 2 :
		if(front == MPI_PROC_NULL) {
			at_bound = true;
		}
		break;
	case 3 :
		if(back == MPI_PROC_NULL) {
			at_bound = true;
		}
		break;
	case 4 :
		if(bottom == MPI_PROC_NULL) {
			at_bound = true;
		}
		break;
	case 5 :
		if(top == MPI_PROC_NULL) {
			at_bound = true;
		}
		break;
	}
	return at_bound;
}


mpi_manager_2D::mpi_manager_2D() {
	DIM = 2;
}

void mpi_manager_2D::setup(int nproc_x, int nproc_y,
                           int coords_x, int coords_y,
                           int rank, int ntasks,
                           int left, int right,
                           int front, int back,
                           MPI_Comm comm2d) {
	//! Constructor for 2D mpi-manager
	DIM = 2;
	this->nproc[0] = nproc_x;
	this->nproc[1] = nproc_y;
	this->coords[0] = coords_x;
	this->coords[1] = coords_y;
	this->rank = rank;
	this->ntasks = ntasks;
	this->left = left;
	this->right = right;
	this->front = front;
	this->back = back;
	this->comm2d = comm2d;

	// Determine position of other ranks
	determin_OtherRanks();
	get_SubComms();
}

mpi_manager_2D mpi_manager_3D::boil_down(int plane_normal) {
	//! Copy constructor
	/*! Here we just copy all the parameters
	  \param plane_normal normal to the 2D plane in 3D space. Options
	  are 0: x-y plane, 1: x-z plane, 2: y-z plane
	 */

	mpi_manager_2D MySmallMPI;

	assert(plane_normal>=0 && plane_normal<3);

	if(plane_normal==0) {
		MySmallMPI.setup(nproc[0], nproc[1], coords[0], coords[1],
		                 rank_plane_xy, nproc[0]*nproc[1],
		                 left, right, front, back,
		                 comm_plane_xy);
	} else if (plane_normal==1) {
		MySmallMPI.setup(nproc[0], nproc[2], coords[0], coords[2],
		                 rank_plane_xz, nproc[0]*nproc[2],
		                 left, right, bottom, top,
		                 comm_plane_xz);
	} else {
		MySmallMPI.setup(nproc[1], nproc[2], coords[1], coords[2],
		                 rank_plane_yz, nproc[1]*nproc[2],
		                 front, back, bottom, top,
		                 comm_plane_yz);
	}

	// Determine position of other ranks:
	MySmallMPI.determin_OtherRanks();
	MySmallMPI.get_SubComms();

	return MySmallMPI;

}


mpi_manager_2D::mpi_manager_2D(NumArray<int> &nproc, NumArray<int> &mx) {
	DIM = 2;
	for(int dir=0; dir<DIM; ++dir) {
		this->nproc[dir] = nproc[dir];
	}
	setup(nproc, mx);
}



mpi_manager_2D::mpi_manager_2D(const mpi_manager_2D &mpi) {
	//! Copy constructor
	/*! Here we just copy all the parameters
	 */
	DIM = mpi.DIM;
	for(int dir=0; dir<DIM; ++dir) {
		nproc[dir] = mpi.nproc[dir];
		coords[dir] = mpi.coords[dir];
	}

	left = mpi.left;
	right = mpi.right;
	front = mpi.front;
	back = mpi.back;

	// Communucators:
	comm2d = mpi.comm2d;
	comm_line_x = mpi.comm_line_x;
	comm_line_y = mpi.comm_line_y;
	
	// Different rank definitions:
	rank = mpi.rank;
	rank_line_x = mpi.rank_line_x;
	rank_line_y = mpi.rank_line_y;

	AllRanks = mpi.AllRanks;
	Neighbours = mpi.Neighbours;
	NeighboursCyclic = mpi.NeighboursCyclic;

}


void mpi_manager_2D::setup(NumArray<int> &nproc, NumArray<int> &mx) {
	
	// Determine the rank of the current task
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// Get number of ranks from MPI
	int ntasks;
	MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
	this->ntasks = ntasks;

	// Set the distribution of processes:
	if(ntasks != nproc[0]*nproc[1]){
		std::cerr << " Wrong number of processes " << std::endl;
		std::cout << ntasks << " " << nproc[0]*nproc[1] << std::endl;
		Finalise();
	}

	if(rank==0) {
		std::cout << " Number of tasks: " << ntasks << " " << nproc[0] << " " << nproc[1] << std::endl;
	}

	// Check if grid can be subdevided as desired
	for(int dir = 0; dir < DIM; ++dir) {
		if(mx[dir] < nproc[dir] && nproc[dir] > 1) {
			if(rank == 0) {
				std::cerr << " Wrong grid topology for dimension ";
				std::cerr << dir << std::endl;
				std::cerr << "  mx[" << dir << "]:" << mx[dir] << std::endl;
				std::cerr << " nproc[" << dir << "]:" << nproc[dir] << std::endl;
			}
			Finalise();
		}
	}

	// Check if grid is a power of 2:
	double eps = 1.e-12;
	for(int dir = 0; dir < DIM; ++dir) {
		double exponent = log(mx[dir])/log(2.);
		int i_exponent = static_cast<int>(exponent+eps);

		if(exponent - i_exponent > 2.*eps) {
			if(rank == 0) {
				std::cerr << " Error: grid must be of the form mx = 2^n ";
				std::cerr << std::endl;
				std::cerr << " Exiting " << std::endl;
			}
			Finalise();
		}
	}

	// Grid is not periodic
	int periods[3] = {false, false, false};
	int reorder = false;
	// If all is okay: Create new communicator "comm3d"  
	MPI_Cart_create(MPI_COMM_WORLD, DIM, nproc, periods, reorder, &comm2d);

	// Retrieve the cartesian topology
	if (rank == 0) {
		int TopoType;
		std::cout << " Cart topology:  ";
		MPI_Topo_test(comm2d, &TopoType);
		switch (TopoType) {
		case MPI_UNDEFINED : 
			std::cout << " MPI_UNDEFINED " << std::endl;
			break;
		case MPI_GRAPH     :
			std::cout << "MPI_GRAPH" << std::endl;
			break;
		case MPI_CART      :
			std::cout << "MPI_CART" << std::endl;
			break;
		}
	}
	
	//   Determine rank again for cartesian communicator -> overwrite rank
	MPI_Comm_rank(comm2d, &rank);

	// std::cout << " my rank: " << rank << std::endl;

	// Translate rank to coordinates
	MPI_Cart_coords(comm2d, rank, DIM, coords);

	// // Backwards translation
	// int TranslateRank;
	// MPI_Cart_rank(comm3d, coords, &TranslateRank);

	// Find neighbouring ranks
	// Syntax: comm3d, shift direction, displacement, source, destination
	MPI_Cart_shift(comm2d, 0, 1, &left , &right);
	MPI_Cart_shift(comm2d, 1, 1, &front, &back);

	// std::cout << " My rank " << rank << " " << left << " " << right << " " << front << " " << back << " " << bottom << " " << top << std::endl;
	// Determine position of other ranks
	determin_OtherRanks();
	get_SubComms();

	

}


void mpi_manager_2D::get_SubComms() {
	//! Obtain ranks and communicators for 1D

	int remain_dims[2];
	// x-direction:
	remain_dims[0] = 1;
	remain_dims[1] = 0;
	MPI_Cart_sub(comm2d, remain_dims, &comm_line_x);
	MPI_Comm_rank(comm_line_x, &rank_line_x);

	// y-direction
	remain_dims[0] = 0;
	remain_dims[1] = 1;
	MPI_Cart_sub(comm2d, remain_dims, &comm_line_y);
	MPI_Comm_rank(comm_line_y, &rank_line_y);
}



void mpi_manager_2D::do_MPISendRecv(NumMatrix<double,2> &buff,
                                    int Destination) {
	//! Do a send-receive operation, where the send-buffer is overwritten
	/*! Origin and destination is the same in this case
	 */
	// Get size of buffer:
	int size = ((buff.getHigh(1) - buff.getLow(1) + 1)*
	            (buff.getHigh(0) - buff.getLow(0) + 1));

	// MPI_Request request[1] = {MPI_REQUEST_NULL};
	// MPI_Request request;
	MPI_Status status;

	//	int tag = rank;
	int SendTag = rank;
	int RecvTag = Destination;
	// int SendTag = rank + Destination;
	// int RecvTag = Destination + rank;


	// Now do the communication:
	MPI_Sendrecv_replace((double *) buff, size, MPI_DOUBLE, Destination,
	                     SendTag, Destination, RecvTag,
	                     comm2d, &status);


	// MPI_Waitall(1, request);
	

}

void mpi_manager_2D::do_MPISendRecv(NumMatrix<double,2> &buff,
                                    int Source, int Destination) {
	//! Do a send-receive operation, where the send-buffer is overwritten
	/*! Get data from somewhere and send own data somewhere else. The original
	  data will be overwritten
	 */
	// Get size of buffer:
	int size = ((buff.getHigh(1) - buff.getLow(1) + 1)*
	            (buff.getHigh(0) - buff.getLow(0) + 1));

	// MPI_Request request[1] = {MPI_REQUEST_NULL};
	// MPI_Request request;
	MPI_Status status;

	//	int tag = rank;
	int SendTag = rank;
	int RecvTag = Source;
	// int SendTag = rank + Destination;
	// int RecvTag = Destination + rank;


	// Now do the communication:
	MPI_Sendrecv_replace((double *) buff, size, MPI_DOUBLE, Destination,
	                     SendTag, Source, RecvTag,
	                     comm2d, &status);


	// MPI_Waitall(1, request);
	

}


int mpi_manager_2D::get_rank() const {
	return rank;
}

int mpi_manager_2D::get_left() const {
	return left;
}

int mpi_manager_2D::get_right() const {
	return right;
}

int mpi_manager_2D::get_front() const {
	return front;
}

int mpi_manager_2D::get_back() const {
	return back;
}

int mpi_manager_2D::get_nproc(int dir) const {
	assert(dir>=0 && dir<DIM);

	return nproc[dir];
}

int mpi_manager_2D::get_coord(int dir) const {
	assert(dir>=0 && dir<DIM);

	return coords[dir];
}

int mpi_manager_2D::get_OtherRankAbs(int ix, int iy) const {
	//! Get rank at some coordinate
	assert(ix>=0 && ix<nproc[0]);
	assert(iy>=0 && iy<nproc[1]);

	return AllRanks(ix, iy);
	
}

int mpi_manager_2D::get_OtherRankRel(int delx, int dely) const {
	//! Get rank at some coordinate
	assert(std::abs(delx)<=nproc[0]);
	assert(std::abs(dely)<=nproc[1]);

	return Neighbours(delx, dely);
	
}


int mpi_manager_2D::get_OtherRankRel_cyclic(int delx, int dely) const {
	//! Get rank at some coordinate
	assert(std::abs(delx)<=nproc[0]);
	assert(std::abs(dely)<=nproc[1]);

	return NeighboursCyclic(delx, dely);

}

bool mpi_manager_2D::is_OuterBoundary(int boundary) const {
	//! Check if a certain boundary is at end of domain
	/*!  Check if a local boundary is at the same time the global
	  boundary. Possible boundaries are:
	  0 - lower x-boundary
	  1 - upper x-boundary
	  2 - lower y-boundary
	  3 - upper y-boundary
	 */
	assert(boundary>=0 && boundary<4);
	bool at_bound = false;
	switch (boundary) {
	case 0 :
		if(left == MPI_PROC_NULL) {
			at_bound = true;
		}
		break;
	case 1 :
		if(right == MPI_PROC_NULL) {
			at_bound = true;
		}
		break;
	case 2 :
		if(front == MPI_PROC_NULL) {
			at_bound = true;
		}
		break;
	case 3 :
		if(back == MPI_PROC_NULL) {
			at_bound = true;
		}
		break;
	}
	return at_bound;
}


grid_manager mpi_manager_2D::make_LocalGrid(grid_manager &GlobalGrid) {
	//! Take the global grid and generate a local one
	
	// First get all properties from global grid manager:
	NumArray<int> mx(DIM);
	NumArray<double> xb(DIM), Len(DIM);
	for(int dir=0; dir<DIM; ++dir) {
		mx[dir] = GlobalGrid.get_mx(dir);
		xb[dir] = GlobalGrid.get_xb(dir);
		Len[dir] = GlobalGrid.get_Len(dir);
	}

	// Now compute local extent of grid (cell-wise and space-wise)
	if (rank == 0) {
		for (int dir=0; dir<DIM; ++dir) {
			mx[dir] /= nproc[dir];
			Len[dir] /= nproc[dir];
		}
	}

	
	MPI_Barrier(comm2d);
	MPI_Bcast(mx, DIM, MPI_INT, 0, comm2d);
	MPI_Bcast(Len, DIM, MPI_DOUBLE, 0, comm2d);
	MPI_Bcast(xb, DIM, MPI_DOUBLE, 0, comm2d);
	MPI_Barrier(comm2d);

	// Now the local computations that need to be done by each rank:
	NumArray<double> xe(DIM);
	for (int dir=0; dir<DIM; ++dir) {
		xb[dir] += Len[dir]*coords[dir];
		xe[dir] = xb[dir] + Len[dir];
	}

	// if(rank == 2) {
	// 	std::cout << " mx: " << mx[0] << " " << mx[1] << " " << mx[2];
	// 	std::cout << std::endl;
	// 	std::cout << " Len: " << Len[0] << " " << Len[1] << " " << Len[2];
	// 	std::cout << std::endl;
	// 	std::cout << " xb: " << xb[0] << " " << xb[1] << " " << xb[2];
	// 	std::cout << std::endl;
	// 	std::cout << " xe: " << xe[0] << " " << xe[1] << " " << xe[2];
	// 	std::cout << std::endl;
	// }


	int rim = GlobalGrid.get_rim();


	// Now make the local grid manager:
	grid_manager LocalGrid(xb[0], xb[1], xe[0], xe[1],
	                       mx[0]+1, mx[1]+1, rim);

	// Now set corresponding boundary types (old type at
	// outer-boundaries / -1 at MPI boundaries)
	for(int bound=0; bound<2*DIM; ++bound) {
		if(is_OuterBoundary(bound)) {
			LocalGrid.set_bcType(bound, GlobalGrid.get_bcType(bound));
		} else {
			LocalGrid.set_bcType(bound, -1);
		}
	}

	return LocalGrid;

}


void mpi_manager_2D::determin_OtherRanks() {

	// Find neighbouring ranks:
	MPI_Cart_shift(comm2d, 0, 1, &left , &right);
	MPI_Cart_shift(comm2d, 1, 1, &front, &back);

	// Determine ranks of neighbour processes:
	int shiftcoord[DIM];
	int lbound[DIM],ubound[DIM];
	for(int dim=0;dim<DIM;dim++){
		lbound[dim]=-nproc[dim];
		ubound[dim]= nproc[dim];
	}
	Neighbours.resize(lbound,ubound);
	Neighbours.clear();

	for(int dim0=-nproc[0]; dim0<=nproc[0]; dim0++){
		shiftcoord[0] = (coords[0]+dim0);
		if(shiftcoord[0] < 0) shiftcoord[0]+=nproc[0];
		for(int dim1=-nproc[1]; dim1<=nproc[1]; dim1++){
			shiftcoord[1] = (coords[1]+dim1);
			if(shiftcoord[1] < 0) shiftcoord[1]+=nproc[1];

			if(shiftcoord[0]>=0 && shiftcoord[0]<nproc[0] &&
			   shiftcoord[1]>=0 && shiftcoord[1]<nproc[1]) {
				// Now determine rank at relative shifted position
				// std::cout << " Cart ";
				// std::cout << shiftcoord[0] << " ";
				// std::cout << shiftcoord[1] << " ";
				// std::cout << rank << " ";
				// std::cout << nproc[0] << " ";
				// std::cout << nproc[1] << " ";
				// std::cout << std::endl;
				MPI_Cart_rank(comm2d, shiftcoord, &Neighbours(dim0,dim1));
			} else {
				// If outside domain set to error value
				Neighbours(dim0, dim1) = MPI_PROC_NULL;
			}

		}
	}

	NeighboursCyclic.resize(lbound,ubound);
	NeighboursCyclic.clear();

	for(int dim0=-nproc[0]; dim0<=nproc[0]; dim0++){
		shiftcoord[0] = (coords[0]+dim0)%nproc[0];
		if(shiftcoord[0] < 0) shiftcoord[0]+=nproc[0];
		for(int dim1=-nproc[1]; dim1<=nproc[1]; dim1++){
			shiftcoord[1] = (coords[1]+dim1)%nproc[1];
			if(shiftcoord[1] < 0) shiftcoord[1]+=nproc[1];

			// Now determine rank at relative shifted position
			MPI_Cart_rank(comm2d, shiftcoord, &NeighboursCyclic(dim0,dim1));

		}
	}
	
	// Now determine absolute position of ranks
	AllRanks.resize(Index::set(0,0),
	                Index::set(nproc[0]-1,nproc[1]-1));

	for(int dim1=0; dim1<nproc[1]; ++dim1) {
		for(int dim0=0; dim0<nproc[0]; ++dim0) {
			int coord[2] = {dim0, dim1};
			MPI_Cart_rank(comm2d, coord, &AllRanks(dim0, dim1));
		}
	}

}

void mpi_manager_2D::Finalise() {
	if(rank==0) {
		std::cerr << " Ending the program " << std::endl;
	}
	MPI_Finalize();
	exit(-5);
}




mpi_manager_1D::mpi_manager_1D() {
	DIM = 1;
}

mpi_manager_1D::mpi_manager_1D(int ntasks_user, int mx) {
	DIM = 1;
	this->ntasks_user = ntasks_user;
	setup(ntasks_user, mx);
}


mpi_manager_1D::mpi_manager_1D(const mpi_manager_1D &mpi) {
	//! Copy constructor
	/*! Here we just copy all the parameters
	 */
	DIM = mpi.DIM;

	ntasks = mpi.ntasks;
	ntasks_user = mpi.ntasks_user;
	coords = mpi.coords;

	left = mpi.left;
	right = mpi.right;

	// Communucators:
	comm1d = mpi.comm1d;
	
	// Different rank definitions:
	rank = mpi.rank;

	AllRanks = mpi.AllRanks;
	Neighbours = mpi.Neighbours;
	NeighboursCyclic = mpi.NeighboursCyclic;

}




void mpi_manager_1D::setup(int ntasks_user, int coords,
                           int rank, int ntasks,
                           int left, int right,
                           MPI_Comm comm1d) {
	//! Constructor for 2D mpi-manager
	DIM = 1;
	this->coords = coords;
	this->rank = rank;
	this->ntasks_user = ntasks_user;
	this->ntasks = ntasks;
	this->left = left;
	this->right = right;
	this->comm1d = comm1d;

	// Determine position of other ranks
	determin_OtherRanks();
}


void mpi_manager_1D::setup(int ntasks_user, int mx) {
	
	// Determine the rank of the current task
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// Get number of ranks from MPI
	int ntasks;
	MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
	this->ntasks = ntasks;

	// Set the distribution of processes:
	if(ntasks != ntasks_user){
		std::cerr << " Wrong number of processes " << std::endl;
		std::cout << ntasks << " " << ntasks_user << std::endl;
		Finalise();
	}

	if(rank==0) {
		std::cout << " Number of tasks: " << ntasks << " " << ntasks_user << std::endl;
	}

	// Check if grid can be subdevided as desired
	if(mx < ntasks && ntasks > 1) {
		if(rank == 0) {
			std::cerr << " Wrong grid topology for dimension ";
			std::cerr << "  mx = " << mx << std::endl;
			std::cerr << " ntasks = " << ntasks << std::endl;
			Finalise();
		}
	}

	// Check if grid is a power of 2:
	double eps = 1.e-12;
	
	double exponent = log(mx)/log(2.);
	int i_exponent = static_cast<int>(exponent+eps);

	if(exponent - i_exponent > 2.*eps) {
		if(rank == 0) {
			std::cerr << " Error: grid must be of the form mx = 2^n ";
			std::cerr << std::endl;
			std::cerr << " Exiting " << std::endl;
		}
		Finalise();
	}

	// Grid is not periodic
	int periods = false;
	int reorder = false;
	// If all is okay: Create new communicator "comm3d"  
	MPI_Cart_create(MPI_COMM_WORLD, DIM, &ntasks, &periods, reorder, &comm1d);

	// Retrieve the cartesian topology
	if (rank == 0) {
		int TopoType;
		std::cout << " Cart topology:  ";
		MPI_Topo_test(comm1d, &TopoType);
		switch (TopoType) {
		case MPI_UNDEFINED : 
			std::cout << " MPI_UNDEFINED " << std::endl;
			break;
		case MPI_GRAPH     :
			std::cout << "MPI_GRAPH" << std::endl;
			break;
		case MPI_CART      :
			std::cout << "MPI_CART" << std::endl;
			break;
		}
	}
	
	//   Determine rank again for cartesian communicator -> overwrite rank
	MPI_Comm_rank(comm1d, &rank);

	// Translate rank to coordinates
	MPI_Cart_coords(comm1d, rank, DIM, &coords);

	// // Backwards translation
	// int TranslateRank;
	// MPI_Cart_rank(comm3d, coords, &TranslateRank);

	// Find neighbouring ranks
	// Syntax: comm3d, shift direction, displacement, source, destination
	MPI_Cart_shift(comm1d, 0, 1, &left , &right);

	// std::cout << " My rank " << rank << " " << left << " " << right << " " << front << " " << back << " " << bottom << " " << top << std::endl;

	

}



void mpi_manager_1D::determin_OtherRanks() {

	// Find neighbouring ranks:
	MPI_Cart_shift(comm1d, 0, 1, &left , &right);

	// Determine ranks of neighbour processes:
	int shiftcoord;

	Neighbours.resize(Index::set(-ntasks), Index::set(ntasks));
	Neighbours.clear();

	for(int pos=-ntasks; pos<=ntasks; ++pos){
		shiftcoord = (coords + pos);
		if(shiftcoord < 0) shiftcoord += ntasks;

		if(shiftcoord>=0 && shiftcoord<ntasks) {
			// Now determine rank at relative shifted position
			// std::cout << " Cart ";
			// std::cout << shiftcoord[0] << " ";
			// std::cout << shiftcoord[1] << " ";
			// std::cout << rank << " ";
			// std::cout << nproc[0] << " ";
			// std::cout << nproc[1] << " ";
			// std::cout << std::endl;
			MPI_Cart_rank(comm1d, &shiftcoord, &Neighbours(pos));
		} else {
			// If outside domain set to error value
			Neighbours(pos) = MPI_PROC_NULL;
		}

	
	}


	NeighboursCyclic.resize(Index::set(-ntasks), Index::set(ntasks));
	NeighboursCyclic.clear();

	for(int pos=-ntasks; pos<=ntasks; ++pos){
		shiftcoord = (coords + pos)%ntasks;
		if(shiftcoord < 0) shiftcoord += ntasks;

		// Now determine rank at relative shifted position
		MPI_Cart_rank(comm1d, &shiftcoord, &NeighboursCyclic(pos));
			
	}
	
	// Now determine absolute position of ranks
	AllRanks.resize(ntasks);

	for(int coord=0; coord<ntasks; ++coord) {
		MPI_Cart_rank(comm1d, &coord, &AllRanks(coord));
	}

}


mpi_manager_1D mpi_manager_2D::boil_down(int direction) {
	//! Copy constructor
	/*! Here we just copy all the parameters
	  \param direction -- remaining direction after reduction
	 */

	mpi_manager_1D MySmallMPI;

	assert(direction>=0 && direction<2);

	if(direction==0) {
		MySmallMPI.setup(nproc[0], coords[0],
		                 rank_line_x, nproc[0],
		                 left, right,
		                 comm_line_x);
	} else {
		MySmallMPI.setup(nproc[1], coords[1],
		                 rank_line_y, nproc[1],
		                 front, back,
		                 comm_line_y);
	}

	return MySmallMPI;

}

int mpi_manager_1D::get_rank() const {
	return rank;
}

int mpi_manager_1D::get_left() const {
	return left;
}

int mpi_manager_1D::get_right() const {
	return right;
}

int mpi_manager_1D::get_ntasks() const {
	return ntasks;
}

int mpi_manager_1D::get_coord() const {
	return coords;
}


int mpi_manager_1D::get_OtherRankAbs(int ipos) const {
	//! Get rank at some coordinate
	assert(ipos>=0 && ipos<ntasks);

	return AllRanks(ipos);
	
}

int mpi_manager_1D::get_OtherRankRel(int ishift) const {
	//! Get rank at some coordinate
	assert(std::abs(ishift)<=ntasks);

	return Neighbours(ishift);
	
}


int mpi_manager_1D::get_OtherRankRel_cyclic(int ishift) const {
	//! Get rank at some coordinate
	assert(std::abs(ishift)<=ntasks);

	return NeighboursCyclic(ishift);

}

bool mpi_manager_1D::is_OuterBoundary(int boundary) const {
	//! Check if a certain boundary is at end of domain
	/*!  Check if a local boundary is at the same time the global
	  boundary. Possible boundaries are:
	  0 - lower boundary
	  1 - upper boundary
	 */
	assert(boundary>=0 && boundary<2);
	bool at_bound = false;
	switch (boundary) {
	case 0 :
		if(left == MPI_PROC_NULL) {
			at_bound = true;
		}
		break;
	case 1 :
		if(right == MPI_PROC_NULL) {
			at_bound = true;
		}
		break;
	}
	return at_bound;
}



void mpi_manager_1D::do_MPISendRecv(NumMatrix<double,2> &buff,
                                    int Destination) {
	//! Do a send-receive operation, where the send-buffer is overwritten
	/*! Origin and destination is the same in this case
	 */
	// Get size of buffer:
	int size = ((buff.getHigh(1) - buff.getLow(1) + 1)*
	            (buff.getHigh(0) - buff.getLow(0) + 1));

	// MPI_Request request[1] = {MPI_REQUEST_NULL};
	// MPI_Request request;
	MPI_Status status;

	//	int tag = rank;
	int SendTag = rank;
	int RecvTag = Destination;
	// int SendTag = rank + Destination;
	// int RecvTag = Destination + rank;


	// Now do the communication:
	MPI_Sendrecv_replace((double *) buff, size, MPI_DOUBLE, Destination,
	                     SendTag, Destination, RecvTag,
	                     comm1d, &status);


	// MPI_Waitall(1, request);
	

}

void mpi_manager_1D::do_MPISendRecv(NumMatrix<double,2> &buff,
                                    int Source, int Destination) {
	//! Do a send-receive operation, where the send-buffer is overwritten
	/*! Get data from somewhere and send own data somewhere else. The original
	  data will be overwritten
	 */
	// Get size of buffer:
	int size = ((buff.getHigh(1) - buff.getLow(1) + 1)*
	            (buff.getHigh(0) - buff.getLow(0) + 1));

	// MPI_Request request[1] = {MPI_REQUEST_NULL};
	// MPI_Request request;
	MPI_Status status;

	//	int tag = rank;
	int SendTag = rank;
	int RecvTag = Source;
	// int SendTag = rank + Destination;
	// int RecvTag = Destination + rank;


	// Now do the communication:
	MPI_Sendrecv_replace((double *) buff, size, MPI_DOUBLE, Destination,
	                     SendTag, Source, RecvTag,
	                     comm1d, &status);


	// MPI_Waitall(1, request);
	

}

grid_manager mpi_manager_1D::make_LocalGrid(grid_manager &GlobalGrid) {
	//! Take the global grid and generate a local one
	
	// First get all properties from global grid manager:
	int mx;
	double xb, Len;

	mx = GlobalGrid.get_mx(0);
	xb = GlobalGrid.get_xb(0);
	Len = GlobalGrid.get_Len(0);

	// Now compute local extent of grid (cell-wise and space-wise)
	if (rank == 0) {
		mx  /= ntasks_user;
		Len /= ntasks_user;
	}

	
	MPI_Barrier(comm1d);
	MPI_Bcast(&mx, DIM, MPI_INT, 0, comm1d);
	MPI_Bcast(&Len, DIM, MPI_DOUBLE, 0, comm1d);
	MPI_Bcast(&xb, DIM, MPI_DOUBLE, 0, comm1d);
	MPI_Barrier(comm1d);

	// Now the local computations that need to be done by each rank:
	double xe;
	xb += Len*coords;
	xe = xb + Len;

	// if(rank == 2) {
	// 	std::cout << " mx: " << mx[0] << " " << mx[1] << " " << mx[2];
	// 	std::cout << std::endl;
	// 	std::cout << " Len: " << Len[0] << " " << Len[1] << " " << Len[2];
	// 	std::cout << std::endl;
	// 	std::cout << " xb: " << xb[0] << " " << xb[1] << " " << xb[2];
	// 	std::cout << std::endl;
	// 	std::cout << " xe: " << xe[0] << " " << xe[1] << " " << xe[2];
	// 	std::cout << std::endl;
	// }


	int rim = GlobalGrid.get_rim();


	// Now make the local grid manager:
	grid_manager LocalGrid(xb, xe, mx+1, rim);

	return LocalGrid;

}


void mpi_manager_1D::Finalise() {
	if(rank==0) {
		std::cerr << " Ending the program " << std::endl;
	}
	MPI_Finalize();
	exit(-5);
}


