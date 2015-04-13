#ifndef DUMMY_GRID_MANAGER_H
#define DUMMY_GRID_MANAGER_H

#include <matrix.H>

class grid_manager {
public:
	grid_manager(int mx1D, int rim, int DIM);
	grid_manager(double xb, double yb, double zb,
	             double xe, double ye, double ze,
	             int mx, int my, int mz,
	             int rim);
	grid_manager(double xb, double yb,
	             double xe, double ye,
	             int mx, int my,
	             int rim);
	grid_manager(double xb, double xe,
	             int mx, int rim);
	grid_manager(const grid_manager& gm);
	~grid_manager();
	void set_bcType(int boundary, int type);
	void set_xb(int dir, int value);
	void set_xe(int dir, int value);
	int get_bcType(int boundary);
	int get_mx(int dir);
	int get_rim();
	double get_delx(int dir) const;
	double get_xb(int dir);
	double get_xe(int dir);
	double get_Len(int dir);
	double get_Pos(int dir, int ix) const;
	NumMatrix<double,1> get_Grid(int dir);
	NumMatrix<double,1> get_CellFaces(int dir);
	NumMatrix<double,1> get_CellWidths(int dir);
private:
	void build_Grid();
	void compute_depedentVars();
	int *bc_Type;
	int *mx;
	double *delx, *xb, *xe, *Len;
	// NumMatrix<double,1> centres[3], facesL[3], width[3];
	NumMatrix<double,1> *centres;
	NumMatrix<double,1> facesL, width;
	int rim, DIM;
};

// class grid_manager {
// public:
// 	grid_2D(int mx1D, int rim);
// 	grid_2D(double xb, double yb,
// 	        double xe, double ye,
// 	        int mx, int my,
// 	        int rim);
// 	grid_2D(const grid_manager& gm);
// 	// grid_2D();
// 	void set_bcType(int boundary, int type);
// 	void set_xb(int dir, int value);
// 	void set_xe(int dir, int value);
// 	int get_bcType(int boundary);
// 	int get_mx(int dir);
// 	double get_delx(int dir) const;
// 	double get_xb(int dir) const;
// 	double get_xe(int dir) const;
// 	double get_Len(int dir) const;
// 	NumMatrix<double,1> get_Grid(int dir);
// private:
// 	void build_Grid();
// 	void compute_depedentVars();
// 	int bc_Type[4];
// 	int mx[2];
// 	double delx[2], xb[2], xe[2], Len[2];
// 	NumMatrix<double,1> centres[2], facesL[2], width[2];
// 	int rim;
// };

class grid_1D {
public:
	grid_1D(double xb, double xe, int Nx, int rim, int type, bool centred=false);
	grid_1D(const grid_1D& gm);
	grid_1D();
	void set_grid(NumMatrix<double,1> Edges, int rim, int Nx, bool centred=false);
	double get_del(int ipos) const;
	double get_del() const;
	double get_xb() const;
	double get_xe() const;
	double get_Len() const;
	double get_xCen(int ix) const;
	double get_xL(int ix) const;
	int get_ipos(double pos);
	int get_mx() const;
	int get_Nx() const;
	int get_rim() const;
	int get_bcType(int bound) const;
	int get_gridLin() const;
private:
	void compute_depedentVars();
	void build_grid();
	bool is_set, centred, is_linear;
	double xb, xe, Len, del, idel;
	int mx, Nx, rim, type, ipos_get;
	NumMatrix<double,1> cellCentres, cellWidhts;
	NumMatrix<double,1> cellEdges;
};


#endif
