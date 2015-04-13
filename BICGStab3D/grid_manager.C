#include "grid_manager.H"
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

grid_manager::grid_manager(int mx1D, int rim, int DIM) {

	bc_Type = NULL;
	mx = NULL;
	delx = NULL;
	xb = NULL;
	xe = NULL;
	Len = NULL;

	centres = NULL;

	this->DIM = DIM;
	if(mx == NULL){
		mx = new int[DIM];
	}
	for(int dir=0; dir<DIM; ++dir) {
		mx[dir] = mx1D;
	}

	// Default for boundaries is periodic (type = 2)
	// for(int bound=0; bound<6; ++bound) {
	// 	bc_Type[bound] = 2;
	// }

	// Default: not periodic:
	if(bc_Type == NULL){
		bc_Type = new int[2*DIM];
	}
	for(int bound=0; bound<2*DIM; ++bound) {
		bc_Type[bound] = 1;
	}

	// Values of xb, xe, Lx and so on are just fixed to some values
	// for the time being

	// xb is value of x at index 0
	// xe is value of x at index mx

	if(xb == NULL) {
		xb = new double[DIM];
	}
			
	if(xe == NULL) {
		xe = new double[DIM];
	}
	for(int dir=0; dir<DIM; ++dir) {
		xb[dir] = 0.;
		xe[dir] = 1.;
	}

	this->rim = rim;

	compute_depedentVars();

	build_Grid();

}

// Specific implementation for 2D
grid_manager::grid_manager(double _xb, double _yb,
                           double _xe, double _ye,
                           int Nx, int Ny, int rim) {

	bc_Type = NULL;
	mx = NULL;
	delx = NULL;
	xb = NULL;
	xe = NULL;
	Len = NULL;

	centres = NULL;

	this->DIM = 2;
	if(mx == NULL) {
		mx = new int[DIM];
	}
	mx[0] = Nx-1;
	mx[1] = Ny-1;

	// Default for boundaries is periodic (type = 2)
	// for(int bound=0; bound<6; ++bound) {
	// 	bc_Type[bound] = 2;
	// }

	// Default: non-periodic...
	if(bc_Type == NULL) {
		bc_Type = new int[2*DIM];
	}
	for(int bound=0; bound<2*DIM; ++bound) {
		bc_Type[bound] = 1;
	}


	// Values of xb, xe, Lx and so on are just fixed to some values
	// for the time being

	// xb is value of x at index 0
	// xe is value of x at index mx
	
	if(xb == NULL) {
		xb = new double[DIM];
	}
	xb[0] = _xb;
	xb[1] = _yb;

	if(xe == NULL) {
		xe = new double[DIM];
	}
	xe[0] = _xe;
	xe[1] = _ye;

	this->rim = rim;

	compute_depedentVars();

	build_Grid();

}



// Specific implementation for 2D
grid_manager::grid_manager(double _xb, double _xe, int Nx, int rim) {

	bc_Type = NULL;
	mx = NULL;
	delx = NULL;
	xb = NULL;
	xe = NULL;
	Len = NULL;

	centres = NULL;

	this->DIM = 1;
	if(mx == NULL) {
		mx = new int[DIM];
	}
	mx[0] = Nx-1;

	// Default for boundaries is periodic (type = 2)
	// for(int bound=0; bound<6; ++bound) {
	// 	bc_Type[bound] = 2;
	// }

	// Default: non-periodic...
	if(bc_Type == NULL) {
		bc_Type = new int[2*DIM];
	}
	for(int bound=0; bound<2*DIM; ++bound) {
		bc_Type[bound] = 1;
	}


	// Values of xb, xe, Lx and so on are just fixed to some values
	// for the time being

	// xb is value of x at index 0
	// xe is value of x at index mx
	
	if(xb == NULL) {
		xb = new double[DIM];
	}
	xb[0] = _xb;

	if(xe == NULL) {
		xe = new double[DIM];
	}
	xe[0] = _xe;

	this->rim = rim;

	compute_depedentVars();

	build_Grid();

}



// Specific implementation for 3D
grid_manager::grid_manager(double _xb, double _yb, double _zb,
                           double _xe, double _ye, double _ze,
                           int Nx, int Ny, int Nz, int rim) {

	bc_Type = NULL;
	mx = NULL;
	delx = NULL;
	xb = NULL;
	xe = NULL;
	Len = NULL;

	centres = NULL;

	this->DIM = 3;
	if(mx == NULL) {
		mx = new int[DIM];
	}
	mx[0] = Nx-1;
	mx[1] = Ny-1;
	mx[2] = Nz-1;

	// Default for boundaries is periodic (type = 2)
	// for(int bound=0; bound<6; ++bound) {
	// 	bc_Type[bound] = 2;
	// }

	// Default: non-periodic...
	if(bc_Type == NULL) {
		bc_Type = new int[2*DIM];
	}
	for(int bound=0; bound<6; ++bound) {
		bc_Type[bound] = 1;
	}


	// Values of xb, xe, Lx and so on are just fixed to some values
	// for the time being

	// xb is value of x at index 0
	// xe is value of x at index mx
	
	if(xb == NULL) {
		xb = new double[DIM];
	}
	xb[0] = _xb;
	xb[1] = _yb;
	xb[2] = _zb;

	if(xe == NULL) {
		xe = new double[DIM];
	}
	xe[0] = _xe;
	xe[1] = _ye;
	xe[2] = _ze;

	this->rim = rim;

	compute_depedentVars();

	build_Grid();
}


grid_manager::grid_manager(const grid_manager &gm) {
	// Copy constructor

	DIM = gm.DIM;

	bc_Type = NULL;
	mx = NULL;
	delx = NULL;
	xb = NULL;
	xe = NULL;
	Len = NULL;

	centres = NULL;

	if(mx == NULL) {
		mx = new int[DIM];
	}
	if(xb == NULL) {
		xb = new double[DIM];
	}
	if(xe == NULL) {
		xe = new double[DIM];
	}
	if(delx == NULL) {
		delx = new double[DIM];
	}
	if(Len == NULL) {
		Len = new double[DIM];
	}

	for(int dir=0; dir<DIM; ++dir) {
		mx[dir] = gm.mx[dir];
		xb[dir] = gm.xb[dir];
		xe[dir] = gm.xe[dir];
		delx[dir] = gm.delx[dir];
		Len[dir] = gm.Len[dir];
	}

	build_Grid();

	if(bc_Type == NULL) {
		bc_Type = new int[DIM];
	}
	for(int bound=0; bound<2*DIM; ++bound) {
		bc_Type[bound] = gm.bc_Type[bound];
	}

}


void grid_manager::build_Grid() {
	// int rim=2;
	
	if(centres == NULL) {
		centres = new NumMatrix<double,1> [DIM];
	}

	for(int dir=0; dir<DIM; ++dir) {
		centres[dir].resize(Index::set(-rim), Index::set(mx[dir]+rim));
		for(int ix=-rim; ix<=mx[dir]+rim; ++ix) {
			centres[dir](ix) = xb[dir] +  delx[dir]*ix;
		}
	}

	// Compute position of cell face: Due to the fact that there are N
	// cells and N+1 cell faces we need one additional constraint to
	// compute the position of the cell faces. This constraint is here
	// chose to be dx_2 = (dx_1)^2/dx_0

	if(DIM>2) {
		// So far only for 2-direction (momentum)
		width.resize(Index::set(-rim), Index::set(mx[2]+rim));
		facesL.resize(Index::set(-rim), Index::set(mx[2]+rim+1));

		width(0) = 2.*sqr(centres[2](1) -
		                  centres[2](0))/(centres[2](2) - centres[2](0));
		for(int ix=-1; ix>=-rim; --ix) {
			width(ix) = 2.*(centres[2](ix+1) - centres[2](ix)) - width(ix+1);
		}
		for(int ix=1; ix<=mx[2]+rim; ++ix) {
			width(ix) = 2.*(centres[2](ix) - centres[2](ix-1)) - width(ix-1);
		}
		
		
		// Determine position of left face of cell (only for 2 direction)
		for(int ix=-rim; ix<=mx[2]+rim; ++ix) {
			facesL(ix) = centres[2](ix) - 0.5*width(ix);
		}
		facesL(mx[2]+rim+1) = centres[2](mx[2]+rim) + 0.5*width(mx[2]+rim);
	}
}

double grid_manager::get_Pos(int dir, int ix) const {
	assert(dir>=0 && dir<DIM);
	assert(ix>=centres[dir].getLow(0) &&
	       ix<=centres[dir].getHigh(0));
	return centres[dir](ix);
}

NumMatrix<double,1> grid_manager::get_Grid(int dir) {
	assert(dir>=0 && dir<DIM);
	return centres[dir];
}

NumMatrix<double,1> grid_manager::get_CellFaces(int dir) {
	assert(dir == 2);
	return facesL;
}

NumMatrix<double,1> grid_manager::get_CellWidths(int dir) {
	assert(dir == 2);
	return width;
}


void grid_manager::compute_depedentVars() {
	if(Len == NULL) {
		Len = new double[DIM];
	}
	if(delx == NULL) {
		delx = new double[DIM];
	}
	for(int dir=0; dir<DIM; ++dir) {
		Len[dir] = xe[dir] - xb[dir];
		delx[dir] = Len[dir]/(1.*mx[dir]);
	}
}

void grid_manager::set_bcType(int boundary, int type) {
	assert(boundary>=0 && boundary<2*DIM);
	if(bc_Type == NULL) {
		bc_Type = new int[2*DIM];
	}
	bc_Type[boundary] = type;
}

void grid_manager::set_xb(int dir, int value) {
	assert(dir>=0 && dir<DIM);
	if(xb == NULL) {
		xb = new double[DIM];
	}
	xb[dir] = value;
}

void grid_manager::set_xe(int dir, int value) {
	assert(dir>=0 && dir<DIM);
	if(xe == NULL) {
		xe = new double[DIM];
	}
	xe[dir] = value;
}


int grid_manager::get_bcType(int boundary) {
	assert(boundary>=0 && boundary<2*DIM && bc_Type!=NULL);

	return bc_Type[boundary];
}

int grid_manager::get_mx(int dir) {
	assert(dir>=0 && dir<DIM && mx!=NULL);
	return mx[dir];
}

int grid_manager::get_rim() {
	return rim;
}


double grid_manager::get_delx(int dir) const {
	assert(dir>=0 && dir<DIM && delx!=NULL);
	return delx[dir];
}

double grid_manager::get_xb(int dir) {
	assert(dir>=0 && dir<DIM && xb!=NULL);
	return xb[dir];
}

double grid_manager::get_xe(int dir) {
	assert(dir>=0 && dir<DIM && xe!=NULL);
	return xe[dir];
}

double grid_manager::get_Len(int dir) {
	assert(dir>=0 && dir<DIM && Len!=NULL);
	return Len[dir];
}

grid_manager::~grid_manager() {
	if(bc_Type != NULL) delete [] bc_Type;
	if(mx != NULL) delete [] mx;
	if(delx != NULL) delete [] delx;
	if(xb != NULL) delete [] xb;
	if(xe != NULL) delete [] xe;
	if(Len != NULL) delete [] Len;
	if(centres != NULL) delete [] centres;
}


// grid_2D::grid_2D(int mx1D, int rim) {
// 	mx[0] = mx1D;
// 	mx[1] = mx1D;

// 	// Default for boundaries is periodic (type = 2)
// 	// for(int bound=0; bound<6; ++bound) {
// 	// 	bc_Type[bound] = 2;
// 	// }

// 	// Default: not periodic:
// 	for(int bound=0; bound<4; ++bound) {
// 		bc_Type[bound] = 1;
// 	}

// 	// Values of xb, xe, Lx and so on are just fixed to some values
// 	// for the time being

// 	// xb is value of x at index 0
// 	// xe is value of x at index mx

// 	for(int dir=0; dir<2; ++dir) {
// 		xb[dir] = 1.;
// 		xe[dir] = 2.;
// 	}

// 	this->rim = rim;

// 	compute_depedentVars();

// 	build_Grid();

// }


// grid_2D::grid_2D(double _xb, double _yb,
//                  double _xe, double _ye,
//                  int Nx, int Ny, int rim) {
// 	mx[0] = Nx-1;
// 	mx[1] = Ny-1;

// 	// Default for boundaries is periodic (type = 2)
// 	// for(int bound=0; bound<6; ++bound) {
// 	// 	bc_Type[bound] = 2;
// 	// }

// 	// Default: non-periodic...
// 	for(int bound=0; bound<4; ++bound) {
// 		bc_Type[bound] = 1;
// 	}


// 	// Values of xb, xe, Lx and so on are just fixed to some values
// 	// for the time being

// 	// xb is value of x at index 0
// 	// xe is value of x at index mx
	
// 	xb[0] = _xb;
// 	xb[1] = _yb;

// 	xe[0] = _xe;
// 	xe[1] = _ye;

// 	this->rim = rim;

// 	compute_depedentVars();

// 	build_Grid();
// }


// grid_2D::grid_2D(const grid_manager &gm) {
// 	// Copy constructor

// 	for(int dir=0; dir<2; ++dir) {
// 		mx[dir] = gm.mx[dir];
// 		xb[dir] = gm.xb[dir];
// 		xe[dir] = gm.xe[dir];
// 		delx[dir] = gm.delx[dir];
// 		Len[dir] = gm.Len[dir];
// 	}

// 	build_Grid();

// 	for(int bound=0; bound<4; ++bound) {
// 		bc_Type[bound] = gm.bc_Type[bound];
// 	}

// }

// void grid_2D::build_Grid() {
// 	// int rim=2;
// 	for(int dir=0; dir<2; ++dir) {
// 		centres[dir].resize(Index::set(-rim), Index::set(mx[dir]+rim));

// 		for(int ix=-rim; ix<=mx[dir]+rim; ++ix) {
// 			centres[dir](ix) = xb[dir] +  delx[dir]*ix;
// 		}
// 	}

// 	// Compute position of cell face: Due to the fact that there are N
// 	// cells and N+1 cell faces we need one additional constraint to
// 	// compute the position of the cell faces. This constraint is here
// 	// chose to be dx_2 = (dx_1)^2/dx_0

// }

// NumMatrix<double,1> grid_2D::get_Grid(int dir) {
// 	return centres[dir];
// }


// void grid_2D::compute_depedentVars() {
// 	for(int dir=0; dir<2; ++dir) {
// 		Len[dir] = xe[dir] - xb[dir];
// 		delx[dir] = Len[dir]/(1.*mx[dir]);
// 	}
// }

// void grid_2D::set_bcType(int boundary, int type) {
// 	assert(boundary>=0 && boundary<4);

// 	bc_Type[boundary] = type;
// }

// void grid_2D::set_xb(int dir, int value) {
// 	assert(dir>=0 && dir<2);
// 	xb[dir] = value;
// }

// void grid_2D::set_xe(int dir, int value) {
// 	assert(dir>=0 && dir<2);
// 	xe[dir] = value;
// }


// int grid_2D::get_bcType(int boundary) {
// 	assert(boundary>=0 && boundary<4);

// 	return bc_Type[boundary];
// }

// int grid_manager::get_mx(int dir) {
// 	assert(dir>=0 && dir<3);
// 	return mx[dir];
// }


// double grid_manager::get_delx(int dir) const {
// 	assert(dir>=0 && dir<3);
// 	return delx[dir];
// }

// double grid_manager::get_xb(int dir) {
// 	assert(dir>=0 && dir<3);
// 	return xb[dir];
// }

// double grid_manager::get_xe(int dir) {
// 	assert(dir>=0 && dir<3);
// 	return xe[dir];
// }

// double grid_manager::get_Len(int dir) {
// 	assert(dir>=0 && dir<3);
// 	return Len[dir];
// }




grid_1D::grid_1D(double xb, double xe, int Nx, int rim, int type, bool centred) {
	is_set = false;
	this->Nx = Nx;
	this->centred = centred;
	mx = Nx-1;
	this->rim = rim; // Number of boundary cells
	this->xb = xb;
	this->xe = xe;
	this->type = type; // Type of grid

	compute_depedentVars();
	build_grid();
}


grid_1D::grid_1D(const grid_1D& grid) {
	is_set = false;
	Nx = grid.Nx;
	mx = grid.mx;
	rim = grid.rim;
	xb = grid.xb;
	xe = grid.xe;
	type = grid.type;
	centred = grid.centred;
	is_linear = grid.is_linear;
	compute_depedentVars();
	build_grid();
}

grid_1D::grid_1D() {
	is_set = false;
	centred = false;
	is_linear = false;
	Nx = 0;
	mx = 0;
	rim = 0;
	xb = 0.;
	xe = 0.;
	type = -99;
}

void grid_1D::compute_depedentVars() {
	ipos_get = 0;
	Len = xe - xb;

	if(centred) {
		del = Len/(1.*(Nx-1));
//		xb -= 0.5*del;
//		xe += 0.5*del;
	} else {
		del = Len/(1.*Nx);
	}

	if(type != 0) {
		build_grid();
	}

	idel = 1./del;
}


void grid_1D::set_grid(NumMatrix<double,1> Edges, int rim, int Nx, bool centred) {
	this->Nx = Nx;
	this->rim = rim;
	this->centred = centred;
	mx = Nx-1;
	cellCentres.resize(Index::set(-rim), Index::set(mx + rim));
	cellWidhts.resize(Index::set(-rim), Index::set(mx + rim));
	cellEdges.resize(Index::set(-rim), Index::set(mx + rim +1));

	assert(Edges.getHigh(0) >= cellCentres.getHigh(0) &&
	       Edges.getLow(0)  <= cellCentres.getLow(0));

	cellEdges = Edges;

	// Compute cell centres & widhts:
	for(int ipos=-rim; ipos<=mx+rim; ++ipos) {
		cellCentres(ipos) = 0.5*(cellEdges(ipos) + cellEdges(ipos+1));
		cellWidhts(ipos) = cellEdges(ipos+1) - cellEdges(ipos);
	}

	is_set = true;
	is_linear = false;

}


void grid_1D::build_grid() {
	cellCentres.resize(Index::set(-rim), Index::set(mx + rim));
	cellWidhts.resize(Index::set(-rim), Index::set(mx + rim));
	cellEdges.resize(Index::set(-rim), Index::set(mx + rim +1));

	if(type == 0) { // Linear grid
		
		// Loop over all cell edges:
		if(centred) {
			for(int ipos=-rim; ipos<=mx+rim+1; ++ipos) {
				cellEdges(ipos) = xb + del*(ipos-0.5);
			}
		} else {
			for(int ipos=-rim; ipos<=mx+rim+1; ++ipos) {
				cellEdges(ipos) = xb + del*ipos;
			}
		}

	} else if(type == 1) { // Sinusoidal grid

		// Loop over all cell edges:
		for(int ipos=-rim; ipos<=mx+rim+1; ++ipos) {
			cellEdges(ipos) = xb + del*(ipos + 2.*sin(2.*M_PI*ipos/(1.*Nx)));
		}

	}

	// Compute cell centres & widhts:
	for(int ipos=-rim; ipos<=mx+rim; ++ipos) {
		cellCentres(ipos) = 0.5*(cellEdges(ipos) + cellEdges(ipos+1));
		cellWidhts(ipos) = cellEdges(ipos+1) - cellEdges(ipos);
	}

	is_set = true;


	// indicator for linear grid
	if(type==0) {
		is_linear = true;
	} else {
		is_linear = false;
	}

}

double grid_1D::get_del(int ipos) const {
	// return del;
	assert(is_set);
	return cellWidhts(ipos);
}

double grid_1D::get_del() const {
	assert(type == 0);
	return cellWidhts(0);
}

double grid_1D::get_xb() const {
	assert(is_set);
	return xb;
}

double grid_1D::get_xe() const {
	assert(is_set);
	return xe;
}

double grid_1D::get_Len() const {
	assert(is_set);
	return Len;
}

double grid_1D::get_xCen(int ipos) const {
	// return (ipos+0.5)*del + xb;
	assert(is_set);
	return cellCentres(ipos);
}

double grid_1D::get_xL(int ipos) const {
	// return ipos*del + xb;
	assert(is_set);
	return cellEdges(ipos);
}

int grid_1D::get_ipos(double pos) {
	assert(is_set);
	bool goLeft(false);
	if(pos < get_xL(ipos_get)) {
		goLeft = true;
	}

	if(goLeft) {
		while(pos < get_xL(ipos_get)) {
			ipos_get--;
		}
	} else {
		while(pos > get_xL(ipos_get)) {
			ipos_get++;
		}
		ipos_get--;
	}
	return ipos_get;
}

int grid_1D::get_mx() const {
	assert(is_set);
	return mx;
}

int grid_1D::get_Nx() const {
	assert(is_set);
	return Nx;
}

int grid_1D::get_rim() const {
	return rim;
}

int grid_1D::get_bcType(int bound) const {
	//! Return type of boundary condition at boundary 'bound'
	/*! Currently not set - use Dirichlet as default */
	return 0;
}

int grid_1D::get_gridLin() const {
	return is_linear;
}
