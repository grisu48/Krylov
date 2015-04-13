#ifndef CRONOS_HDF5FILE_H
#define CRONOS_HDF5FILE_H = 1

#include "hdf5.h"
#ifdef parallel
#include "mpi.h"
#endif
#include "matrix.H"


#define SWITCH_ON 1
#define SWITCH_OFF 0
// Here the user can choose if a parallel IO to a single hdf5 file is
// desired:
#define HDF_PARALLEL_IO SWITCH_ON


class Hdf5Stream{
public:
	Hdf5Stream(std::string, int, int rank=0, bool use_MPI_IO=false
#ifdef parallel
#if (HDF_PARALLEL_IO==SWITCH_ON)
	           ,MPI_Comm = MPI_COMM_WORLD
#endif
#endif
	           );
	~Hdf5Stream();
	bool close();
	void Reopen();
	bool AddGlobalAttr(std::string AttrName, double);
	bool ChangeGlobalAttr(std::string AttrName, double);
	bool AddGlobalAttr(std::string AttrName, float);
	bool ChangeGlobalAttr(std::string AttrName, float);
	bool AddGlobalAttr(std::string AttrName, int);
	bool ChangeGlobalAttr(std::string AttrName, int);
	bool AddGlobalAttr(std::string AttrName, std::string AttrData);
	// bool AddGlobalAttr(std::string AttrName, std::string *AttrData);
	bool AddGlobalAttr(std::string AttrName, double *AttrData, int);
	bool AddGlobalAttr(std::string AttrName, float *AttrData, int);
	bool AddGlobalAttr(std::string AttrName, unsigned long *AttrData, int);
	bool AddGlobalAttr(std::string AttrName, int *AttrData, int);
	bool AddGlobalAttr(std::string AttrName, std::string *AttrData, int num);
	template < class T >
	bool AddGlobalAttr(std::string AttrName, T AttrData, hid_t datatype);
	template < class T >
	bool AddGlobalAttr(std::string AttrName, T *, hid_t, int entries);
	bool AddAttributeToArray(std::string, const std::string &, int);
	bool AddAttributeToArray(std::string, const std::string &, float);
	bool AddAttributeToArray(std::string, const std::string &, double);

	bool Write1DMatrix(std::string ArrayName, NumMatrix<double,1> &data,
	                   double xb, double dx, int);
	bool Write1DMatrix(std::string ArrayName, NumMatrix<float,1> &data);
	bool Write3DMatrix(std::string ArrayName, NumMatrix<double,3> &data,
	                   double *xb, double *dx);
	bool Write3DVecMatrix(std::string ArrayName, NumMatrix<float,3> &data_x,
	                      NumMatrix<float,3> &data_y, NumMatrix<float,3> &data_z);
	std::string Write2DMatrix(std::string ArrayName, NumMatrix<double,2> &data,
	                          double *xb, double *dx, int);
	bool Write3DMatrix(std::string ArrayName, NumMatrix<float,3> &data,
	                   double *xb, double *dx, bool with_opendxinfo=true);
#ifdef parallel
#if (HDF_PARALLEL_IO==SWITCH_ON)
	bool Write3DMatrix_withMPI_IO(std::string ArrayName,
	                              NumMatrix<float,3> &data,
	                              NumArray<int> &mx_global,
	                              NumArray<int> &mx_local,
	                              NumArray<int> &rank_shift,
	                              // NumArray<int> &rank_pos,
	                              NumArray<float> &xb,
	                              NumArray<float> &dx,
	                              bool with_opendxinfo=true);
#endif
#endif
	bool Write3DMatrix(std::string ArrayName, NumMatrix<float,3> &data);
	bool Write3DMatrixSwap(std::string ArrayName, NumMatrix<double,3> &data,
	                       double *xb, double *dx);
	bool Write2DMatrix(std::string ArrayName, NumMatrix<float,2> &data,
	                   double *xb, double *dx, bool with_opendxinfo=true);
	bool WriteArray(int *data, int max);
	bool WriteArray(std::string ,int *data, int max);
	bool WriteArray(float *data, int max);
	bool WriteArray(std::string ,float *data, int max);
	bool WriteArray(double *data, int max);
	bool WriteArray(std::string ,double *data, int max);
	bool WriteNDArray(std::string ,float *data, int mx[], int dim);
private:
	template <typename T>
	bool AddAttrToArrSingle(std::string, hid_t &, const std::string &, T);
	bool AddDatasetName(std::string &);
	hid_t hdf5file;
	hid_t group;
	hid_t plist_file_id, plist_dset_id;
	herr_t return_val;
	std::string filename;
	// char* fname;
	int NumEntries;
	int num;
	bool open, use_MPI_IO;
	unsigned MajorNum, MinorNum, ReleaseNum;
};



class Hdf5iStream{
public:
	Hdf5iStream(std::string, int rank=0);
	~Hdf5iStream();
	bool close();
	bool Read2DMatrix(std::string, NumMatrix<float,2> &);
	bool Read3DMatrix(std::string, NumMatrix<float,3> &);
	bool Read3DMatrix(std::string ArrayName, NumMatrix<float,3> &data,
	                  double *xb, double *dx);
	bool Read3DMatrix(std::string, NumMatrix<double,3> &);
	bool Read2DFrom3DMatrix(std::string, NumMatrix<float,2> &, int dir,
	                        int offset);
	float ReadPointFromMatrix(std::string, NumArray<int> &position);
	float ReadPointFrom2DMatrix(std::string, int ix, int iy);
	float ReadPointFrom3DMatrix(std::string, int ix, int iy, int iz);
	void getSize(std::string, int [], int);
	bool ReadGlobalAttr(std::string , double &);
	bool ReadGlobalAttr(std::string , float &);
	bool ReadGlobalAttr(std::string , unsigned long &);
	bool ReadGlobalAttr(std::string , int &);

	// int ReadDatasetAttr(std::string DataSetName, std::string AttrName);
	float ReadDatasetAttr(std::string DataSetName, std::string AttrName);
	bool ReadDatasetGrid(std::string DataSetName, NumArray<float> &xPos,
	                     NumArray<float> &yPos, NumArray<float> &zPos);
	bool ReadDatasetGrid(std::string DataSetName, NumArray<float> &xPos,
	                     NumArray<float> &yPos);
  
	int GetDatasetDimension(std::string DataSetName);
	NumArray<int> GetDatasetExtent(std::string DataSetName);
	std::string GetDatasetName(int);
	// bool doesAttrExist(const char* name ) const;
	bool doesAttrExist(std::string) const;
private:
	hid_t hdf5file;
	hid_t group;
	herr_t return_val;
	int NumEntries;
	unsigned MajorNum, MinorNum, ReleaseNum;
};

#endif
