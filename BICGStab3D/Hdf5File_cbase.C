#include <iostream>
#include <stdlib.h>
#include "Hdf5File_cbase.H"
#ifdef parallel
#endif

// #ifndef H5_NO_NAMESPACE
// using namespace H5;
// #endif
using namespace std;

Hdf5Stream::Hdf5Stream(string filename, int NumEntries, int rank,
                       bool use_MPI_IO
#ifdef parallel
#if (HDF_PARALLEL_IO==SWITCH_ON)
                       , MPI_Comm comm
#endif
#endif
                       )
{
	this->filename = filename;

#ifdef parallel
#if (HDF_PARALLEL_IO==SWITCH_ON)
	this->use_MPI_IO = use_MPI_IO;
	// Set specific property lists:

	if(this->use_MPI_IO) {
		// Create file access property list for parallel I/O
		plist_file_id = H5Pcreate(H5P_FILE_ACCESS);
		MPI_Info info_mpi  = MPI_INFO_NULL;
		H5Pset_fapl_mpio(plist_file_id, comm, info_mpi);

		// property list for dataset access
		plist_dset_id = H5Pcreate(H5P_DATASET_XFER);
		H5Pset_dxpl_mpio(plist_dset_id, H5FD_MPIO_COLLECTIVE);
	}
#endif
#else 
	this->use_MPI_IO = false;
#endif

	if(!this->use_MPI_IO) {
		plist_file_id = H5P_DEFAULT;
		plist_dset_id = H5P_DEFAULT;
	}

	hdf5file =  H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
	                      plist_file_id);

	group    = H5Gcreate2(hdf5file, "/Data", H5P_DEFAULT, H5P_DEFAULT,
	                      H5P_DEFAULT);

	// Create dataspace
	hid_t info_id = H5Screate(H5S_SCALAR);

	// Create Attribute
	hid_t info = H5Acreate2(group, "Entries", H5T_NATIVE_INT, info_id,
	                        H5P_DEFAULT, H5P_DEFAULT);
	// hid_t info = H5Acreate2(group, "Entries", H5T_NATIVE_INT, H5S_SCALAR,
	//                         H5P_DEFAULT, H5P_DEFAULT);
	// Write Attribute
	return_val = H5Awrite(info, H5T_NATIVE_INT, &NumEntries);
	// Close Attribute
	return_val = H5Aclose(info);
	

	// DataSpace infospace( 1, &DimsInfo);
	// datatype.setOrder( H5T_ORDER_LE ); // Little endian
	// Attribute* Info = new Attribute(group->createAttribute("Entries", datatype, infospace));
	// Info->write( PredType::NATIVE_INT, &NumEntries);
  

	// Indicate new version of hdf5 output:
	// Create Attribute
	info = H5Acreate2(group, "using_cbase", H5T_NATIVE_INT, info_id,
	                  H5P_DEFAULT, H5P_DEFAULT);
	int cbase_val = 1;
	// Write Attribute
	return_val = H5Awrite(info, H5T_NATIVE_INT, &cbase_val);
	// Close Attribute
	return_val = H5Aclose(info);


	// Close dataspace
	return_val = H5Sclose(info_id);


	this->NumEntries = NumEntries;
	this->num = 0;
	this->open = true;

}




void Hdf5Stream::Reopen()
{
	if(!open) {
	  hdf5file =  H5Fopen(filename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
	  
	  group = H5Gopen2(hdf5file, "Data", H5P_DEFAULT);
	}
	this->open = true;
}



Hdf5iStream::Hdf5iStream(string filename, int rank)
{
	if(rank == 0){
		cout << " Opening file: " << filename;
	}

	// int len = filename.size();
	// char* fname = new char[len+1];
	// filename.copy(fname,len);
	// fname[len] = '\0';

	// Open File
	// hdf5file = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
	hdf5file = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

	// Open Group in File
	group = H5Gopen2( hdf5file, "Data", H5P_DEFAULT);

	if(rank == 0){
		cout << " ...done " << endl;
	}

	int numh5;
	// Open Attribute "Entries"
	hid_t Info = H5Aopen(group, "Entries", H5P_DEFAULT);
	// Read Attribute (number of entries)
	return_val = H5Aread(Info, H5T_NATIVE_INT, &numh5);
	H5Aclose(Info);

	this->NumEntries = numh5;
	// Getting the version 
	// H5::H5Library::getLibVersion(MajorNum, MinorNum, ReleaseNum);

}




bool Hdf5Stream::AddGlobalAttr(string AttrName, double AttrData)
{

	// Choose double, little endian of size 1
	hid_t datatype = H5Tcopy(H5T_NATIVE_DOUBLE);
	return_val = H5Tset_order(datatype, H5T_ORDER_LE);

	// Write Attribute
	AddGlobalAttr(AttrName, AttrData, datatype);
	return true;
}


bool Hdf5Stream::ChangeGlobalAttr(string AttrName, double AttrData)
{
	// Open Attribute
	hid_t info = H5Aopen(group, AttrName.c_str(), H5P_DEFAULT);
	// Rewrite Attribute
	return_val = H5Awrite(info, H5T_NATIVE_DOUBLE, &AttrData);
	H5Aclose(info);
	return true;
}


bool Hdf5iStream::ReadGlobalAttr(string AttrName, double &AttrData)
{
	// Open Attribute
	hid_t info = H5Aopen(group, AttrName.c_str(), H5P_DEFAULT);
	// Read Attribute Data
	return_val = H5Aread(info, H5T_NATIVE_DOUBLE, &AttrData);
	H5Aclose(info);
	return true;
}

template < class T >
bool Hdf5Stream::AddGlobalAttr(string AttrName, T AttrData, hid_t datatype) {

	// Create dataspace
	hid_t AttrSpace = H5Screate(H5S_SCALAR);
	
	// Create Attribute
	hid_t info = H5Acreate2(group, AttrName.c_str(), datatype, AttrSpace,
	                        H5P_DEFAULT, H5P_DEFAULT);
	// Write Attribute
	return_val = H5Awrite(info, datatype, &AttrData);

	// Close Attribute
	return_val = H5Aclose(info);
	// Close Dataspace
	return_val = H5Sclose(AttrSpace);
	return true;
	
}


template < class T >
bool Hdf5Stream::AddGlobalAttr(string AttrName, T *AttrData,
                               hid_t datatype, int entries) {

	// Create dataspace
	hid_t AttrSpace = H5Screate(H5S_SIMPLE);
	hsize_t dimAttr = entries;
	return_val  = H5Sset_extent_simple(AttrSpace, 1, &dimAttr, NULL);
	
	// Create Attribute
	hid_t info = H5Acreate2(group, AttrName.c_str(), datatype, AttrSpace,
	                        H5P_DEFAULT, H5P_DEFAULT);
	// Write Attribute
	return_val = H5Awrite(info, datatype, AttrData);

	// Close Attribute
	return_val = H5Aclose(info);
	// Close Dataspace
	return_val = H5Sclose(AttrSpace);
	return true;
	
}



bool Hdf5Stream::AddGlobalAttr(string AttrName, float AttrData)
{
	// Choose float, little endian of size 1
	hid_t datatype = H5Tcopy(H5T_NATIVE_FLOAT);
	return_val = H5Tset_order(datatype, H5T_ORDER_LE);
	
	// Write Attribute
	AddGlobalAttr(AttrName, AttrData, datatype);
	return true;
}

bool Hdf5Stream::ChangeGlobalAttr(string AttrName, float AttrData)
{
	// Open Attribute
	hid_t info = H5Aopen(group, AttrName.c_str(), H5P_DEFAULT);
	// Rewrite Attribute
	return_val = H5Awrite(info, H5T_NATIVE_FLOAT, &AttrData);
	H5Aclose(info);
	return true;
}

bool Hdf5iStream::ReadGlobalAttr(string AttrName, float &AttrData)
{
	// Open Attribute
	hid_t info = H5Aopen(group, AttrName.c_str(), H5P_DEFAULT);
	// Read Attribute
	return_val = H5Aread(info, H5T_NATIVE_FLOAT, &AttrData);
	H5Aclose(info);
	return true;
}


bool Hdf5Stream::AddGlobalAttr(string AttrName, int AttrData)
{
	// Choose integer, little endian of size 1
	hid_t datatype = H5Tcopy(H5T_NATIVE_INT);
	return_val = H5Tset_order(datatype, H5T_ORDER_LE);

	// Write Attribute
	AddGlobalAttr(AttrName, AttrData, datatype);
	return true;
}


bool Hdf5Stream::ChangeGlobalAttr(string AttrName, int AttrData)
{
	// Open Attribute
	hid_t info = H5Aopen(group, AttrName.c_str(), H5P_DEFAULT);
	// Rewrite Attribute
	return_val = H5Awrite(info, H5T_NATIVE_INT, &AttrData);
	// Close Attribute
	return_val = H5Aclose(info);
	return true;
}

bool Hdf5iStream::ReadGlobalAttr(string AttrName, int &AttrData)
{
	// Open Attribute
	hid_t info = H5Aopen(group, AttrName.c_str(), H5P_DEFAULT);
	// Read Attribute Data
	return_val = H5Aread(info, H5T_NATIVE_INT, &AttrData);
	// Close Attribute
	return_val = H5Aclose(info);
	return true;
}

bool Hdf5iStream::ReadGlobalAttr(string AttrName, unsigned long &AttrData)
{
	// Open Attribute
	hid_t info = H5Aopen(group, AttrName.c_str(), H5P_DEFAULT);
	// Read Attribute Data
	return_val = H5Aread(info, H5T_NATIVE_ULONG, &AttrData);
	// Close Attribute
	return_val = H5Aclose(info);
	return true;
}


bool Hdf5Stream::AddGlobalAttr(string AttrName, string AttrData)
{
	// Set datatype to string
	hid_t datatype = H5Tcopy(H5T_C_S1);

	// Set size to be variable
	return_val = H5Tset_size(datatype, H5T_VARIABLE );

	// Use little endian
	return_val = H5Tset_order(datatype, H5T_ORDER_LE);

	// Write Attribute
	AddGlobalAttr(AttrName, AttrData, datatype);
	return true;
}

bool Hdf5Stream::AddGlobalAttr(string AttrName, double *AttrData, int num)
{

	// Choose double, little endian of size num
	hid_t datatype = H5Tcopy(H5T_NATIVE_DOUBLE);
	return_val = H5Tset_order(datatype, H5T_ORDER_LE);
	
	// Write Attribute
	AddGlobalAttr(AttrName, AttrData, datatype, num);
	return true;
}



bool Hdf5Stream::AddGlobalAttr(string AttrName, float *AttrData, int num)
{
	// Choose double, little endian of size num
	hid_t datatype = H5Tcopy(H5T_NATIVE_FLOAT);
	return_val = H5Tset_order(datatype, H5T_ORDER_LE);

	// Write Attribute
	AddGlobalAttr(AttrName, AttrData, datatype, num);
	return true;
}


bool Hdf5Stream::AddGlobalAttr(string AttrName,unsigned long *AttrData, int num)
{
	// Choose double, little endian of size num
	hid_t datatype = H5Tcopy(H5T_NATIVE_ULONG);
	return_val = H5Tset_order(datatype, H5T_ORDER_LE);

	// Write Attribute
	AddGlobalAttr(AttrName, AttrData, datatype, num);
	return true;
}


bool Hdf5Stream::AddGlobalAttr(string AttrName, int *AttrData, int num)
{
	// Choose double, little endian of size num
	hid_t datatype = H5Tcopy(H5T_NATIVE_INT);
	return_val = H5Tset_order(datatype, H5T_ORDER_LE);

	// Write Attribute
	AddGlobalAttr(AttrName, AttrData, datatype, num);
	return true;
}


bool Hdf5Stream::AddGlobalAttr(string AttrName, string *AttrData, int num)
{
	// Choose double, little endian of size num
	hid_t datatype = H5Tcopy(H5T_C_S1);
	// Set size to be variable
	return_val = H5Tset_size(datatype, H5T_VARIABLE );
	return_val = H5Tset_order(datatype, H5T_ORDER_LE);

	// Write Attribute
	AddGlobalAttr(AttrName, AttrData, datatype, num);
	return true;
}


bool Hdf5Stream::AddDatasetName(string &DatasetName) {

	char cnum[255];
	sprintf(cnum,"%2.2d",this->num);
	string AttrName = "Name_om";
	AttrName += cnum;
 
	// get length of string
	hsize_t StrLen = DatasetName.size();

	// Set datatype to string
	hid_t datatype = H5Tcopy(H5T_C_S1);
	// Set size to length of string
	return_val = H5Tset_size(datatype, StrLen+1);
	// return_val = H5Tset_size(datatype, H5T_VARIABLE);

	// Set order to little endian
	return_val = H5Tset_order(datatype, H5T_ORDER_LE);

	// Create dataspace for attribute
	hid_t AttrSpace = H5Screate(H5S_SCALAR);
	// hsize_t dims[1] = {DatasetName.size()};
	// hid_t AttrSpace = H5Screate_simple (1, dims, NULL);
	
	// Create Attribute
	hid_t info = H5Acreate2(group, AttrName.c_str(), datatype, AttrSpace,
	                        H5P_DEFAULT, H5P_DEFAULT);
	// hid_t info = H5Acreate(group, AttrName.c_str(), datatype, AttrSpace,
	//                        H5P_DEFAULT);
	// Write Attribute
	return_val = H5Awrite(info, datatype, DatasetName.c_str());

	// Close Dataspace
	return_val = H5Sclose(AttrSpace);
	// Close Attribute
	return_val = H5Aclose(info);
	return true;
}

string Hdf5iStream::GetDatasetName(int num) {

	char cnum[255];
	sprintf(cnum,"%2.2d",num);
	string AttrName = "Name_om";
	AttrName += cnum;
	
	
	// Open Attribute
	hid_t Info = H5Aopen(group, AttrName.c_str(), H5P_DEFAULT);
	hid_t ftype = H5Aget_type(Info);
	hid_t type = H5Tget_native_type(ftype, H5T_DIR_ASCEND);

	htri_t size_var;
	string DatasetName;
	
	if((size_var = H5Tis_variable_str(ftype)) == 1) {
		char *string_attr;
		return_val = H5Aread(Info, type, &string_attr);
		DatasetName = string_attr;
	} else {
		char string_out[255];
		return_val = H5Aread(Info, type, string_out);
		DatasetName = string_out;
	}
	H5Aclose(Info);

	return DatasetName;
}





bool Hdf5Stream::Write1DMatrix(string ArrayName, NumMatrix<double,1> &data,
                               double Origin, double Delta, int numin)
{
	/* Routine to write NumMatrix data in wrong ordering to hdf5 file.

	   Remarks:
     
	   On reading the hdf data the swapped dimensions have to be taken
	   into account
    
	*/
	int mx = data.getHigh(0) - data.getLow(0) + 1;
  
	num+=1;

	int DIM = 1;
	
	// Choose double, little endian of size 1
	hid_t datatype = H5Tcopy(H5T_NATIVE_DOUBLE);
	return_val = H5Tset_order(datatype, H5T_ORDER_LE);

	// Create dataspace
	hsize_t DimsData = mx;
	hid_t dataspace = H5Screate_simple(DIM, &DimsData, NULL);


	// Supplying additional attributes for opendx input

	// Datatype: double, little endian of size 1
	hid_t datatypefloat = H5Tcopy(H5T_NATIVE_DOUBLE);
	return_val = H5Tset_order(datatypefloat, H5T_ORDER_LE);
	
	// Create dataspace for attribute
	hid_t attrspace = H5Screate(H5S_SCALAR);


	// Create dataset
	hid_t dataset = H5Dcreate2(group, ArrayName.c_str(), datatype, dataspace,
	                           H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	// Create attributes
	hid_t origin = H5Acreate2(dataset, "origin", datatypefloat, attrspace,
	                          H5P_DEFAULT, H5P_DEFAULT);
	hid_t delta = H5Acreate2(dataset, "delta", datatypefloat, attrspace,
	                         H5P_DEFAULT, H5P_DEFAULT);

	// Write attributes
	return_val = H5Awrite(origin, datatype, &Origin);
	return_val = H5Awrite(delta,  datatype, &Delta);

	// Write data
	return_val = H5Dwrite(dataset, datatype, H5S_ALL,
	                      H5S_ALL, H5P_DEFAULT, data);
	return true;
}


bool Hdf5Stream::Write1DMatrix(string ArrayName, NumMatrix<float,1> &data)
{
	/* Routine to write NumMatrix data in wrong ordering to hdf5 file.

	*/
	int mx = data.getHigh(0) - data.getLow(0) + 1;
  
	num+=1;

	int DIM = 1;
	
	// Choose float, little endian of size 1
	hid_t datatype = H5Tcopy(H5T_NATIVE_FLOAT);
	return_val = H5Tset_order(datatype, H5T_ORDER_LE);

	// Create dataspace
	hsize_t DimsData = mx;
	hid_t dataspace = H5Screate_simple(DIM, &DimsData, NULL);


	// Create dataset
	hid_t dataset = H5Dcreate2(group, ArrayName.c_str(), datatype, dataspace,
	                           H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

	// Write data
	return_val = H5Dwrite(dataset, datatype, H5S_ALL,
	                      H5S_ALL, H5P_DEFAULT, data);
	return true;
}


bool Hdf5Stream::AddAttributeToArray(string ArrayName,
                                     const string &AttributeName,
                                     double AttributeData)
{

	/*******************************************************
	 *  Routine to write double Attribute to Array Data   *
	 ******************************************************/

	// Preparing Attribute:
	hid_t datatype = H5Tcopy(H5T_NATIVE_DOUBLE);

	return AddAttrToArrSingle(ArrayName, datatype, AttributeName, AttributeData);
  
}


bool Hdf5Stream::AddAttributeToArray(string ArrayName,
                                     const string &AttributeName,
                                     float AttributeData)
{

	/*******************************************************
	 *  Routine to write float Attribute to Array Data   *
	 ******************************************************/

	// Preparing Attribute:
	hid_t datatype = H5Tcopy(H5T_NATIVE_FLOAT);

	return AddAttrToArrSingle(ArrayName, datatype, AttributeName, AttributeData);
  
}



bool Hdf5Stream::AddAttributeToArray(string ArrayName,
                                     const string &AttributeName,
                                     int AttributeData)
{

	/*******************************************************
	 *  Routine to write integer Attribute to Array Data   *
	 ******************************************************/
	// Preparing Attribute:
	hid_t datatype = H5Tcopy(H5T_NATIVE_INT);

	return AddAttrToArrSingle(ArrayName, datatype, AttributeName, AttributeData);
  
}

template <typename T>
bool Hdf5Stream::AddAttrToArrSingle(string ArrayName,
                                    hid_t &datatype,
                                    const string &AttributeName,
                                    T AttributeData)
{
	/*******************************************************
	 *  Routine to add Attribute to Any written Array Data *
	 *  the corresponding array is identified by its name  *
	 ******************************************************/

	// Reopen written dataset:
	hid_t dataset = H5Dopen2(group, ArrayName.c_str(), H5P_DEFAULT);

	hid_t AttrSpace = H5Screate(H5S_SCALAR);

	return_val = H5Tset_order(datatype, H5T_ORDER_LE);

	// Create Attribute
	hid_t info = H5Acreate2(dataset, AttributeName.c_str(), datatype, AttrSpace,
	                        H5P_DEFAULT, H5P_DEFAULT);
	// Write Attribute
	return_val = H5Awrite(info, datatype, &AttributeData);
	return_val = H5Aclose(info);
	return_val = H5Sclose(AttrSpace);
	return_val = H5Dclose(dataset);


	return true;
}



bool Hdf5Stream::Write3DMatrix(string ArrayName, NumMatrix<double,3> &data,
                               double *xb, double *dx)
{
	/* Routine to write NumMatrix data in wrong ordering to hdf5 file.

	   Remarks:
     
	   On reading the hdf data the swapped dimensions have to be taken
	   into account
    
	*/
	int mx[3];
	mx[0]=data.getHigh(2) - data.getLow(2) + 1;
	mx[1]=data.getHigh(1) - data.getLow(1) + 1;
	mx[2]=data.getHigh(0) - data.getLow(0) + 1;
  
	AddDatasetName(ArrayName);

	num+=1;

	int DIM = 3;
	// Choose double, little endian of size 1
	hid_t datatype = H5Tcopy(H5T_NATIVE_DOUBLE);
	return_val = H5Tset_order(datatype, H5T_ORDER_LE);
  
	hsize_t DimsData[DIM];
	for(int q=0; q<DIM; ++q){
		DimsData[q]  = mx[q];
	}
	hid_t dataspace = H5Screate_simple(DIM, DimsData, NULL);
 

	// Supplying additional attributes for opendx input
	// Datatype: double, little endian of size 1
	hid_t datatypefloat = H5Tcopy(H5T_NATIVE_DOUBLE);
	return_val = H5Tset_order(datatypefloat, H5T_ORDER_LE);
	
	// Create dataspace for attribute
	hsize_t DimsAttr = 3;
	hid_t attrspace = H5Screate_simple(1, &DimsAttr, NULL);

	double Origin[3];
	double Delta[3];
	for(int q=0; q<3; ++q){
		Origin[q] = xb[q];
		Delta[q]  = dx[q];
	}

	// Create dataset
	hid_t dataset = H5Dcreate2(group, ArrayName.c_str(), datatype, dataspace,
	                           H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

	// Create attributes
	hid_t origin = H5Acreate2(dataset, "origin", datatypefloat, attrspace,
	                          H5P_DEFAULT, H5P_DEFAULT);
	hid_t delta = H5Acreate2(dataset, "delta", datatypefloat, attrspace,
	                         H5P_DEFAULT, H5P_DEFAULT);
	// Write attributes
	return_val = H5Awrite(origin, datatypefloat, &Origin);
	return_val = H5Awrite(delta,  datatypefloat, &Delta);

	// Close attributes
	return_val = H5Aclose(origin);
	return_val = H5Aclose(delta);

	// Write data
	return_val = H5Dwrite(dataset, datatype, H5S_ALL,
	                      H5S_ALL, H5P_DEFAULT, data);
	return_val = H5Dclose(dataset);
	return true;
}




string Hdf5Stream::Write2DMatrix(string ArrayName, NumMatrix<double,2> &data,
                                 double *xb, double *dx, int numin)
{
	/* Routine to write NumMatrix data in wrong ordering to hdf5 file.

	   Remarks:
     
	   On reading the hdf data the swapped dimensions have to be taken
	   into account
    
	*/
	int mx[2];
	mx[0]=data.getHigh(1) - data.getLow(1) + 1;
	mx[1]=data.getHigh(0) - data.getLow(0) + 1;

	string SaveName = ArrayName;

	char numchar[255];
	sprintf(numchar,"%5.5i",numin);
	ArrayName = ArrayName+numchar;
	SaveName = SaveName+numchar;
	num+=1;
	if(num > NumEntries){
		return false;
	}

	int DIM = 2;
	// Choose double, little endian of size 1
	hid_t datatype = H5Tcopy(H5T_NATIVE_DOUBLE);
	return_val = H5Tset_order(datatype, H5T_ORDER_LE);
  
	hsize_t DimsData[DIM];
	for(int q=0; q<DIM; ++q){
		DimsData[q]  = mx[q];
	}
	hid_t dataspace = H5Screate_simple(DIM, DimsData, NULL);
 

	// Supplying additional attributes for opendx input
	// Datatype: double, little endian of size 1
	hid_t datatypefloat = H5Tcopy(H5T_NATIVE_DOUBLE);
	return_val = H5Tset_order(datatypefloat, H5T_ORDER_LE);

	hsize_t DimsAttr = 2;
	hid_t attrspace = H5Screate_simple(1, &DimsAttr, NULL);

	double Origin[2];
	double Delta[2];
	for(int q=0; q<DIM; ++q){
		Origin[q] = xb[q];
		Delta[q]  = dx[q];
	}


	// Supplying frame number for movie in opendx:
	hid_t datatypenum = H5Tcopy(H5T_NATIVE_INT);
	return_val = H5Tset_order(datatypenum, H5T_ORDER_LE);

	hsize_t DimsInfo = 1;
	hid_t numspace = H5Screate_simple(1, &DimsInfo, NULL);


	// Create dataset
	hid_t dataset = H5Dcreate2(group, ArrayName.c_str(), datatype, dataspace,
	                           H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

	// Create attributes
	hid_t origin = H5Acreate2(dataset, "origin", datatypefloat, attrspace,
	                          H5P_DEFAULT, H5P_DEFAULT);
	hid_t delta = H5Acreate2(dataset, "delta", datatypefloat, attrspace,
	                         H5P_DEFAULT, H5P_DEFAULT);
	hid_t number = H5Acreate2(dataset, "num", datatypenum, numspace,
	                          H5P_DEFAULT, H5P_DEFAULT);

	// Write attributes
	return_val = H5Awrite(origin, datatypefloat, &Origin);
	return_val = H5Awrite(delta,  datatypefloat, &Delta);
	return_val = H5Awrite(number,  datatypenum, &num);

	// Close attributes
	return_val = H5Aclose(origin);
	return_val = H5Aclose(delta);
	return_val = H5Aclose(number);

	// Write data
	return_val = H5Dwrite(dataset, datatype, H5S_ALL,
	                      H5S_ALL, H5P_DEFAULT, data);
	return_val = H5Dclose(dataset);

	return SaveName;
}



bool Hdf5Stream::Write3DVecMatrix(string ArrayName,
                                  NumMatrix<float,3> &data_x,
                                  NumMatrix<float,3> &data_y,
                                  NumMatrix<float,3> &data_z)
{
	/* Routine to write a Matrix of vectorial values to the file.

	   Remarks:
     
	   On reading the hdf data the swapped dimensions have to be taken
	   into account
    
	*/
	int ib[3];
	ib[0] = data_x.getLow(2);
	ib[1] = data_x.getLow(1);
	ib[2] = data_x.getLow(0);
	int mx[3];
	mx[0]=data_x.getHigh(2) - data_x.getLow(2) + 1;
	mx[1]=data_x.getHigh(1) - data_x.getLow(1) + 1;
	mx[2]=data_x.getHigh(0) - data_x.getLow(0) + 1;
  
	AddDatasetName(ArrayName);

	num+=1;

	int DIM = 3;

	/* 
     * Define array datatype for the data in the file.
     - 1 component 
     - 3 entries
     */ 
	hsize_t ArrayExtent[1];
	ArrayExtent[0] = 3;
	hid_t datatype = H5Tcopy(H5T_NATIVE_FLOAT);
	H5Tset_size(datatype, 3);
	return_val = H5Tset_order(datatype, H5T_ORDER_LE);

	cerr << " Has to be tested " << endl;
	exit(2);

    // ArrayType datatype( PredType::NATIVE_FLOAT, 1, ArrayExtent);
    // datatype.setOrder( H5T_ORDER_LE );


	// FloatType datatype( PredType::NATIVE_DOUBLE );
	// datatype.setOrder( H5T_ORDER_LE );
  
	hsize_t DimsData[DIM];
	for(int q=0; q<DIM; ++q){
		DimsData[q]  = mx[q];
	}
	hid_t dataspace = H5Screate_simple(DIM, DimsData, NULL);
 
	float data[mx[0]][mx[1]][mx[2]][3];
	for (int i = 0; i < mx[0]; i++) {
		for (int j = 0; j < mx[1]; j++) {
			for (int k = 0; k < mx[2]; k++) {
				data[i][j][k][0] = data_x(k+ib[2],j+ib[1],i+ib[0]);
				data[i][j][k][1] = data_y(k+ib[2],j+ib[1],i+ib[0]);
				data[i][j][k][2] = data_z(k+ib[2],j+ib[1],i+ib[0]);
			}
		}
	}

	// Create dataset
	hid_t dataset = H5Dcreate2(group, ArrayName.c_str(), datatype, dataspace,
	                           H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

	// Write data
	return_val = H5Dwrite(dataset, datatype, H5S_ALL,
	                      H5S_ALL, H5P_DEFAULT, data);
	return_val = H5Dclose(dataset);
	return true;
}


int Hdf5iStream::GetDatasetDimension(std::string DataSetName) {
	hid_t dataset = H5Dopen2(group, DataSetName.c_str(), H5P_DEFAULT);
	
	// Get dataspace handler
	hid_t dataspace = H5Dget_space(dataset);
	int  dimhdf = H5Sget_simple_extent_ndims(dataspace);

	return dimhdf;
}


NumArray<int> Hdf5iStream::GetDatasetExtent(std::string DataSetName) {
	//! Get extent of each dimension
	hid_t dataset = H5Dopen2(group, DataSetName.c_str(), H5P_DEFAULT);
	
	// Get dataspace handler
	hid_t dataspace = H5Dget_space(dataset);

	int  ndims = H5Sget_simple_extent_ndims(dataspace);
	
	hsize_t dims_out[ndims];

	ndims = H5Sget_simple_extent_dims(dataspace, dims_out, NULL);

	NumArray<int> Nx(ndims);

	cout << endl << " my dims: " << ndims << " " << endl << endl;

	for(int idim=0; idim<ndims; ++idim) {
		Nx[idim] = dims_out[ndims-(idim+1)];
		// Nx[idim] = dims_out[idim];
	}

	H5Dclose(dataset);

	return Nx;

}


// int Hdf5iStream::ReadDatasetAttr(std::string DataSetName, std::string AttrName)
// {
// 	// Open the dataset
// 	hid_t dataset = H5Dopen2(group, DataSetName.c_str(), H5P_DEFAULT);

// 	hid_t attr = H5Aopen(dataset, AttrName.c_str(), H5P_DEFAULT);
// 	int value;
// 	H5Aread(attr, H5T_NATIVE_INT, &value);
// 	return value;
	
// }


float Hdf5iStream::ReadDatasetAttr(std::string DataSetName, std::string AttrName)
{
	// Open the dataset
	hid_t dataset = H5Dopen2(group, DataSetName.c_str(), H5P_DEFAULT);

	hid_t attr = H5Aopen(dataset, AttrName.c_str(), H5P_DEFAULT);
	float value;
	H5Aread(attr, H5T_NATIVE_FLOAT, &value);

	H5Aclose(attr);
	H5Dclose(dataset);

	return value;
	
}



bool Hdf5iStream::ReadDatasetGrid(string DataSetName, NumArray<float> &xPos,
                                  NumArray<float> &yPos) {
	// Read grid for a 2D matrix

	// Open the dataset
	hid_t dataset = H5Dopen2(group, DataSetName.c_str(), H5P_DEFAULT);

	float xb[2];
	hid_t Origin = H5Aopen(dataset, "origin", H5P_DEFAULT);
	H5Aread(Origin, H5T_NATIVE_FLOAT, xb);
	H5Aclose(Origin);

	float dx[2];
	hid_t Delta = H5Aopen(dataset, "delta", H5P_DEFAULT);
	H5Aread(Delta, H5T_NATIVE_FLOAT, dx);
	H5Aclose(Delta);

	hid_t dataspace = H5Dget_space(dataset);
	int  ndims = H5Sget_simple_extent_ndims(dataspace);
	if(ndims != 2) {
		cerr << " Only for 2D data " << endl;
		exit(3);
	}
	hsize_t dims_out[ndims];
	ndims = H5Sget_simple_extent_dims(dataspace, dims_out, NULL);

	// Now build the data-arrays:
	xPos.resize(dims_out[1]);
	yPos.resize(dims_out[0]);

	for(unsigned int ix=0; ix<dims_out[1]; ++ix) {
		xPos[ix] = xb[0] + dx[0]*ix;
	}
	for(unsigned int iy=0; iy<dims_out[0]; ++iy) {
		yPos[iy] = xb[1] + dx[1]*iy;
	}

	// Close the dataset:
	H5Dclose(dataset);

	return true;

}

bool Hdf5iStream::ReadDatasetGrid(string DataSetName, NumArray<float> &xPos,
                                  NumArray<float> &yPos, NumArray<float> &zPos) {

	// Open the dataset
	hid_t dataset = H5Dopen2(group, DataSetName.c_str(), H5P_DEFAULT);

	float xb[3];
	hid_t Origin = H5Aopen(dataset, "origin", H5P_DEFAULT);
	H5Aread(Origin, H5T_NATIVE_FLOAT, xb);
	H5Aclose(Origin);

	float dx[3];
	hid_t Delta = H5Aopen(dataset, "delta", H5P_DEFAULT);
	H5Aread(Delta, H5T_NATIVE_FLOAT, dx);
	H5Aclose(Delta);

	hid_t dataspace = H5Dget_space(dataset);
	int  ndims = H5Sget_simple_extent_ndims(dataspace);
	hsize_t dims_out[ndims];
	ndims = H5Sget_simple_extent_dims(dataspace, dims_out, NULL);

	// Now build the data-arrays:
	xPos.resize(dims_out[2]);
	yPos.resize(dims_out[1]);
	zPos.resize(dims_out[0]);

	for(unsigned int ix=0; ix<dims_out[2]; ++ix) {
		xPos[ix] = xb[0] + dx[0]*ix;
	}
	for(unsigned int iy=0; iy<dims_out[1]; ++iy) {
		yPos[iy] = xb[1] + dx[1]*iy;
	}
	for(unsigned int iz=0; iz<dims_out[0]; ++iz) {
		zPos[iz] = xb[2] + dx[2]*iz;
	}

	// Close the dataset:
	H5Dclose(dataset);

	return true;

}



bool Hdf5iStream::Read2DMatrix(string ArrayName, NumMatrix<float,2> &data)
{
	/* Routine to write NumMatrix data in wrong ordering to hdf5 file.

	   Remarks:
     
	   On reading the hdf data the swapped dimensions have to be taken
	   into account
    
	*/

	int DIM = 2;
	int lbound[2], ubound[2];

	hid_t dataset = H5Dopen2(group, ArrayName.c_str(), H5P_DEFAULT);

	// Get dataspace handler
	hid_t dataspace = H5Dget_space(dataset); 

	int  dimhdf = H5Sget_simple_extent_ndims(dataspace);
	if(dimhdf != DIM){
		cerr << " Wrong dimensionality of input data: " << endl;
		cerr << dimhdf << " " << DIM << endl;
		exit(-2);
	}
	hsize_t dims_out[DIM];
	int ndims = H5Sget_simple_extent_dims(dataspace, dims_out, NULL);

	if(ndims != DIM) {
		cerr << " Wrong number of dimensions " << ndims << " - " << DIM << endl;
		exit(-22);
	}

	for(int i=0;i<DIM;i++){
		lbound[i]=0;
		ubound[i]=int(dims_out[DIM-(i+1)])-1;
	}
	data.resize(lbound,ubound);

	return_val = H5Dread(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
	                     H5P_DEFAULT, data);
	// Close the dataset:
	return_val = H5Dclose(dataset);

	return true;
}



bool Hdf5iStream::Read3DMatrix(string ArrayName, NumMatrix<double,3> &data)
{
	/* Routine to write NumMatrix data in wrong ordering to hdf5 file.

	   Remarks:
     
	   On reading the hdf data the swapped dimensions have to be taken
	   into account
    
	*/

	int DIM = 3;
	int mx[3];
	mx[0] = data.getHigh(2)-data.getLow(2)+1;
	mx[1] = data.getHigh(1)-data.getLow(1)+1;
	mx[2] = data.getHigh(0)-data.getLow(0)+1;

	hid_t dataset = H5Dopen2(group, ArrayName.c_str(), H5P_DEFAULT);
	
	// Get dataspace handler
	hid_t dataspace = H5Dget_space(dataset);    /* dataspace handle */

	int  dimhdf = H5Sget_simple_extent_ndims(dataspace);
	if(dimhdf != DIM){
		cerr << " Wrong dimensionality of input data: " << endl;
		cerr << dimhdf << " " << DIM << endl;
		exit(-2);
	}
	hsize_t dims_out[DIM];
	// Get dims
	int ndims = H5Sget_simple_extent_dims(dataspace, dims_out, NULL);
	if(ndims != DIM) {
		cerr << " Wrong number of dimensions " << ndims << " - " << DIM << endl;
		exit(-22);
	}
	for(int i=0; i<DIM; ++i){
		if((int(dims_out[i])) != mx[i]){
			cerr << " Wrong size of dimension " << i << ":" << endl;
			cerr << int(dims_out[i]) << " " << mx[i] << endl;
		}
	}

	return_val = H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
	                     H5P_DEFAULT, data);
	return true;
}


void Hdf5iStream::getSize(string ArrayName, int mx[], int DIM) 
{
  
	hid_t dataset = H5Dopen2(group, ArrayName.c_str(), H5P_DEFAULT);
	// Get dataspace handler
	hid_t dataspace = H5Dget_space(dataset);

	int dimhdf = H5Sget_simple_extent_ndims(dataspace);
	if(dimhdf != DIM){
		cerr << " Wrong dimensionality of input data: " << endl;
		cerr << dimhdf << " " << DIM << endl;
		exit(-2);
	}

	hsize_t dims_out[DIM];
	H5Sget_simple_extent_dims(dataspace, dims_out, NULL);
	for(int i=0; i<DIM; ++i){
		mx[DIM-i-1] = dims_out[i];
	}
	H5Dclose(dataset);

}


bool Hdf5Stream::Write3DMatrix(string ArrayName, NumMatrix<float,3> &data) {
	double dummy[3];
	return Write3DMatrix(ArrayName, data, dummy, dummy, false);
}


bool Hdf5Stream::Write3DMatrix(string ArrayName, NumMatrix<float,3> &data,
                               double *xb, double *dx, bool with_opendxinfo)
{
	/* Routine to write NumMatrix data in wrong ordering to hdf5 file.

	   Remarks:
     
	   On reading the hdf data the swapped dimensions have to be taken
	   into account
    
	*/
	int mx[3];
	mx[0]=data.getHigh(2) - data.getLow(2) + 1;
	mx[1]=data.getHigh(1) - data.getLow(2) + 1;
	mx[2]=data.getHigh(0) - data.getLow(2) + 1;
  
	AddDatasetName(ArrayName);
	num+=1;

	int DIM = 3;
	hid_t datatype = H5Tcopy(H5T_NATIVE_FLOAT);
	return_val = H5Tset_order(datatype, H5T_ORDER_LE);
  
	hsize_t DimsData[DIM];
	for(int q=0; q<DIM; ++q){
		DimsData[q]  = mx[q];
	}
	hid_t dataspace = H5Screate_simple(DIM, DimsData, NULL);
 
	// Supplying additional attributes for opendx input

	hid_t datatypefloat = H5Tcopy(H5T_NATIVE_FLOAT);
	return_val = H5Tset_order(datatypefloat, H5T_ORDER_LE);

	// Create dataspace for attribute
	hsize_t DimsAttr = 3;
	hid_t attrspace = H5Screate_simple(1, &DimsAttr, NULL);
	float Origin[3];
	float Delta[3];
	if( with_opendxinfo ) {
		for(int q=0; q<3; ++q){
			Origin[q] = float(xb[q]);
			Delta[q]  = float(dx[q]);
		}
	}

	// Create dataset
	hid_t dataset = H5Dcreate2(group, ArrayName.c_str(), datatype, dataspace,
	                           H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	if( with_opendxinfo ) {
		// Create attributes
		hid_t origin = H5Acreate2(dataset, "origin", datatypefloat, attrspace,
		                          H5P_DEFAULT, H5P_DEFAULT);
		hid_t delta = H5Acreate2(dataset, "delta", datatypefloat, attrspace,
		                         H5P_DEFAULT, H5P_DEFAULT);

		// Write attributes
		return_val = H5Awrite(origin, datatypefloat, &Origin);
		return_val = H5Awrite(delta,  datatypefloat, &Delta);

		// Close attributes
		return_val = H5Aclose(origin);
		return_val = H5Aclose(delta);
	}

	// Write data
	return_val = H5Dwrite(dataset, datatype, H5S_ALL,
	                      H5S_ALL, H5P_DEFAULT, data);
	
	// Close the dataset:
	return_val = H5Dclose(dataset);

	return true;
}


#ifdef parallel
#if (HDF_PARALLEL_IO==SWITCH_ON)
bool Hdf5Stream::Write3DMatrix_withMPI_IO(string ArrayName,
                                          NumMatrix<float,3> &data,
                                          NumArray<int> &mx_global,
                                          NumArray<int> &mx_local,
                                          NumArray<int> &rank_shift,
                                          // NumArray<int> &rank_pos,
                                          NumArray<float> &xb,
                                          NumArray<float> &dx,
                                          bool with_opendxinfo)
{
	/* Routine to write NumMatrix data in wrong ordering to hdf5 file. MPI
	   parallel form of the routine that writes all data to a single file.

	   Remarks:
     
	   On reading the hdf data the swapped dimensions have to be taken
	   into account

	   rim can be computed from mx_local.
    
	*/

	// Determine rim of data
	int rim = 0;
	if(data.getLow(0) < 0) {
		rim = -data.getLow(0);
	}

	

	int mx[3];
	mx[0]=data.getHigh(2) - data.getLow(2) + 1;
	mx[1]=data.getHigh(1) - data.getLow(2) + 1;
	mx[2]=data.getHigh(0) - data.getLow(2) + 1;
  
	AddDatasetName(ArrayName);
	num+=1;

	int DIM = 3;
	hid_t datatype = H5Tcopy(H5T_NATIVE_FLOAT);
	return_val = H5Tset_order(datatype, H5T_ORDER_LE);
  
	hsize_t DimsData[DIM];
	DimsData[0] = mx_global[2] + 1 + 2*rim;
	DimsData[1] = mx_global[1] + 1 + 2*rim;
	DimsData[2] = mx_global[0] + 1 + 2*rim;

	hid_t dataspace = H5Screate_simple(DIM, DimsData, NULL);
 
	// Supplying additional attributes for opendx input

	hid_t datatypefloat = H5Tcopy(H5T_NATIVE_FLOAT);
	return_val = H5Tset_order(datatypefloat, H5T_ORDER_LE);

	// Create dataspace for attribute
	hsize_t DimsAttr = 3;
	hid_t attrspace = H5Screate_simple(1, &DimsAttr, NULL);
	float Origin[3];
	float Delta[3];
	if( with_opendxinfo ) {
		for(int q=0; q<3; ++q){
			Origin[q] = xb[q];
			Delta[q]  = dx[q];
		}
	}

	// Create dataset
	hid_t dataset_id = H5Dcreate2(group, ArrayName.c_str(), datatype, dataspace,
	                              H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	if( with_opendxinfo ) {
		// Create attributes
		hid_t origin = H5Acreate2(dataset_id, "origin", datatypefloat,
		                          attrspace, H5P_DEFAULT, H5P_DEFAULT);
		hid_t delta = H5Acreate2(dataset_id, "delta", datatypefloat, attrspace,
		                         H5P_DEFAULT, H5P_DEFAULT);

		// Write attributes
		return_val = H5Awrite(origin, datatypefloat, &Origin);
		return_val = H5Awrite(delta,  datatypefloat, &Delta);

		// Close attributes
		return_val = H5Aclose(origin);
		return_val = H5Aclose(delta);
	}

	// Now make distinguis local from global data via hyperslabs:
	hsize_t sizeLocal[DIM];
	sizeLocal[0] = mx_local[2] + 1 + 2*rim;
	sizeLocal[1] = mx_local[1] + 1 + 2*rim;
	sizeLocal[2] = mx_local[0] + 1 + 2*rim;

	hsize_t offset[DIM];
	// offset[0] = rank_pos[2]*rank_shift[2];
	// offset[1] = rank_pos[1]*rank_shift[1];
	// offset[2] = rank_pos[0]*rank_shift[0];
	offset[0] = rank_shift[2];
	offset[1] = rank_shift[1];
	offset[2] = rank_shift[0];

	hid_t dataspaceLocal = H5Screate_simple(DIM, sizeLocal, NULL);

	// Select local data as hyperslab in dataset
	H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, offset, NULL,
	                    sizeLocal, NULL);


	// Write data
	hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
	H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
	return_val = H5Dwrite(dataset_id, datatype, dataspaceLocal, dataspace,
	                      plist_id, data);
	H5Pclose(plist_id);

	H5Sclose(attrspace);
	H5Sclose(dataspaceLocal);
	H5Sclose(dataspace);


	
	// Close the dataset:
	return_val = H5Dclose(dataset_id);

	return true;
}
#endif
#endif



bool Hdf5iStream::Read3DMatrix(string ArrayName, NumMatrix<float,3> &data)
{
	/* Routine to write NumMatrix data in wrong ordering to hdf5 file.

	   Remarks:
     
	   On reading the hdf data the swapped dimensions have to be taken
	   into account
    
	*/

	int DIM = 3;
	int lbound[3], ubound[3];

	hid_t dataset = H5Dopen2(group, ArrayName.c_str(), H5P_DEFAULT);

	// Get dataspace handler
	hid_t dataspace = H5Dget_space(dataset); 

	int  dimhdf = H5Sget_simple_extent_ndims(dataspace);
	if(dimhdf != DIM){
		cerr << " Wrong dimensionality of input data: " << endl;
		cerr << dimhdf << " " << DIM << endl;
		exit(-2);
	}
	hsize_t dims_out[DIM];
	int ndims = H5Sget_simple_extent_dims(dataspace, dims_out, NULL);

	if(ndims != DIM) {
		cerr << " Wrong number of dimensions " << ndims << " - " << DIM << endl;
		exit(-22);
	}

	for(int i=0;i<3;i++){
		lbound[i]=0;
		ubound[i]=int(dims_out[DIM-(i+1)])-1;
	}
	data.resize(lbound,ubound);

	return_val = H5Dread(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
	                     H5P_DEFAULT, data);
	// Close the dataset:
	return_val = H5Dclose(dataset);

	return true;
}


bool Hdf5iStream::Read2DFrom3DMatrix(std::string ArrayName,
                                     NumMatrix<float,2> &data,
                                     int dir, int posPerp) {

	// Open the dataset:
	hid_t dataset = H5Dopen2(group, ArrayName.c_str(), H5P_DEFAULT);
	
	// Get dimensions of dataset:
	int DIM = 3;

	hid_t dataspace = H5Dget_space(dataset); 
	int  dimhdf = H5Sget_simple_extent_ndims(dataspace);
	if(dimhdf != DIM){
		cerr << " Wrong dimensionality of input data: " << endl;
		cerr << dimhdf << " " << DIM << endl;
		exit(-2);
	}
	hsize_t dims_out[DIM];
	int ndims = H5Sget_simple_extent_dims(dataspace, dims_out, NULL);

	// Beware - dimensions are swapped
	hsize_t mx[2];
	if(dir==0) { // x,y
		mx[0] = dims_out[1];
		mx[1] = dims_out[2];
	} else if (dir==1) { // x,z
		mx[0] = dims_out[0];
		mx[1] = dims_out[2];
	} else if (dir==2) { // y,z
		mx[0] = dims_out[0];
		mx[1] = dims_out[1];
	} else {
		cerr << " Error no such direction " << endl;
		exit(-234);
	}

	cerr << " Making memspace " << endl;
	hid_t memspace =  H5Screate_simple(2, mx, NULL);
	cerr << " done " << endl;
	
	hsize_t count[3], offset[3];

	int lbound[2], ubound[2];
	lbound[0] = 0;
	lbound[1] = 0;

	if(dir==0) { // x,y plane
		// count[0] = mx[0];
		// count[1] = mx[1];
		// count[2] = 1;
		// offset[0] = 0;
		// offset[1] = 0;
		// offset[2] = posPerp;
		count[0] = 1;
		count[1] = dims_out[1];
		count[2] = dims_out[2];
		offset[0] = posPerp;
		offset[1] = 0;
		offset[2] = 0;
		ubound[0] = dims_out[2]-1;
		ubound[1] = dims_out[1]-1;
	} else if (dir==1) { // x,z plane
		// count[0] = mx[0];
		// count[1] = 1;
		// count[2] = mx[1];
		// offset[0] = 0;
		// offset[1] = posPerp;
		// offset[2] = 0;
		count[0] = dims_out[0];
		count[1] = 1;
		count[2] = dims_out[2];
		offset[0] = 0;
		offset[1] = posPerp;
		offset[2] = 0;
		ubound[0] = dims_out[2]-1;
		ubound[1] = dims_out[0]-1;
	} else { // y,z plane
		// count[0] = 1;
		// count[1] = mx[0];
		// count[2] = mx[1];
		// offset[0] = posPerp;
		// offset[1] = 0;
		// offset[2] = 0;
		count[0] = dims_out[0];
		count[1] = dims_out[1];
		count[2] = 1;
		offset[0] = 0;
		offset[1] = 0;
		offset[2] = posPerp;
		ubound[0] = dims_out[1]-1;
		ubound[1] = dims_out[0]-1;
	}


	cerr << " resize " << endl;
	data.resize(lbound,ubound);
	cerr << " resize done " << endl;

	H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, offset, NULL,
	                    count, NULL);
	cerr << " selc hyper done " << endl;
	H5Dread(dataset, H5T_NATIVE_FLOAT, memspace, dataspace,
	        H5P_DEFAULT, data);
	cerr << " read done " << endl;

	// Close the dataset:
	return_val = H5Dclose(dataset);
		

}


float Hdf5iStream::ReadPointFromMatrix(std::string ArrayName, NumArray<int> &position) {

	// Open the dataset:
	hid_t dataset = H5Dopen2(group, ArrayName.c_str(), H5P_DEFAULT);
	
	// Get dimensions of dataset:
	int DIM = position.getLength();

	hid_t dataspace = H5Dget_space(dataset); 
	int  dimhdf = H5Sget_simple_extent_ndims(dataspace);
	if(dimhdf != DIM){
		cerr << " Wrong dimensionality of input data: " << endl;
		cerr << dimhdf << " " << DIM << endl;
		exit(-2);
	}

	// Select just a single point
	hsize_t dim_point[] = {1};

	hid_t memspace =  H5Screate_simple(1, dim_point, NULL);

	int numPoints = 1;
	// Set coordinates of a point
	// hssize_t coord[numPoints][DIM];
	// coord[0][0] = iz;
	// coord[0][1] = iy;
	// coord[0][2] = ix;
	hsize_t coord[DIM];// = {iz, iy, ix};
	for(int dir=0; dir<DIM; ++dir) {
		coord[dir] = position(DIM-dir-1);
	}
	

	// H5Sselect_elements(dataspace, H5S_SELECT_SET, numPoints, 
	//                    (const hssize_t **)coord);
	H5Sselect_elements(dataspace, H5S_SELECT_SET, 1, 
	                   coord);

	float data[numPoints];
	H5Dread(dataset, H5T_NATIVE_FLOAT, memspace, dataspace,
	        H5P_DEFAULT, data);

	// Close the dataset:
	return_val = H5Dclose(dataset);

	return data[0];

}


float Hdf5iStream::ReadPointFrom2DMatrix(std::string ArrayName,
                                         int ix, int iy) {

	// Open the dataset:
	hid_t dataset = H5Dopen2(group, ArrayName.c_str(), H5P_DEFAULT);
	
	// Get dimensions of dataset:
	int DIM = 2;

	hid_t dataspace = H5Dget_space(dataset); 
	int  dimhdf = H5Sget_simple_extent_ndims(dataspace);
	if(dimhdf != DIM){
		cerr << " Wrong dimensionality of input data: " << endl;
		cerr << dimhdf << " " << DIM << endl;
		exit(-2);
	}

	// Select just a single point
	hsize_t dim_point[] = {1};

	hid_t memspace =  H5Screate_simple(1, dim_point, NULL);

	int numPoints = 1;
	// Set coordinates of a point
	// hssize_t coord[numPoints][DIM];
	// coord[0][0] = iz;
	// coord[0][1] = iy;
	// coord[0][2] = ix;
	hsize_t coord[DIM];// = {iz, iy, ix};
	coord[0] = iy;
	coord[1] = ix;
	

	// H5Sselect_elements(dataspace, H5S_SELECT_SET, numPoints, 
	//                    (const hssize_t **)coord);
	H5Sselect_elements(dataspace, H5S_SELECT_SET, 1, 
	                   coord);

	float data[numPoints];
	H5Dread(dataset, H5T_NATIVE_FLOAT, memspace, dataspace,
	        H5P_DEFAULT, data);

	cout << " Data: " << data[0] << endl;

	// Close the dataset:
	return_val = H5Dclose(dataset);

	return data[0];

}



float Hdf5iStream::ReadPointFrom3DMatrix(std::string ArrayName,
                                         int ix, int iy, int iz) {

	// Open the dataset:
	hid_t dataset = H5Dopen2(group, ArrayName.c_str(), H5P_DEFAULT);
	
	// Get dimensions of dataset:
	int DIM = 3;

	hid_t dataspace = H5Dget_space(dataset); 
	int  dimhdf = H5Sget_simple_extent_ndims(dataspace);
	if(dimhdf != DIM){
		cerr << " Wrong dimensionality of input data: " << endl;
		cerr << dimhdf << " " << DIM << endl;
		exit(-2);
	}

	// Select just a single point
	hsize_t dim_point[] = {1};

	hid_t memspace =  H5Screate_simple(1, dim_point, NULL);

	int numPoints = 1;
	// Set coordinates of a point
	// hssize_t coord[numPoints][DIM];
	// coord[0][0] = iz;
	// coord[0][1] = iy;
	// coord[0][2] = ix;
	hsize_t coord[DIM];// = {iz, iy, ix};
	coord[0] = iz;
	coord[1] = iy;
	coord[2] = ix;
	

	// H5Sselect_elements(dataspace, H5S_SELECT_SET, numPoints, 
	//                    (const hssize_t **)coord);
	H5Sselect_elements(dataspace, H5S_SELECT_SET, 1, 
	                   coord);

	float data[numPoints];
	H5Dread(dataset, H5T_NATIVE_FLOAT, memspace, dataspace,
	        H5P_DEFAULT, data);

	cout << " Data: " << data[0] << endl;

	// Close the dataset:
	return_val = H5Dclose(dataset);

	return data[0];

}


bool Hdf5iStream::Read3DMatrix(string ArrayName, NumMatrix<float,3> &data,
                               double *xb, double *dx)
{
	/* 
	   Routine to read 3D matrix from h5-file. Parameters are:
	   ArrayName -> Name of dataset w/o group name
	   data      -> NumMatrix-Array to hold data
	   xb        -> Array to hold values of lower bound positions
	   dx        -> Array to hold grid size
	*/

	int DIM = 3;
	int lbound[3], ubound[3];
	float dummy[3];

	hid_t dataset = H5Dopen2(group, ArrayName.c_str(), H5P_DEFAULT);
	hid_t Origin = H5Aopen(dataset, "origin", H5P_DEFAULT);
	return_val = H5Aread(Origin, H5T_NATIVE_FLOAT, dummy);
	xb[0] = double(dummy[0]);
	xb[1] = double(dummy[1]);
	xb[2] = double(dummy[2]);

	hid_t Delta = H5Aopen(dataset, "delta", H5P_DEFAULT);
	return_val = H5Aread(Delta, H5T_NATIVE_FLOAT, dummy);
	dx[0] = double(dummy[0]);
	dx[1] = double(dummy[1]);
	dx[2] = double(dummy[2]);

	// Get dataspace handler
	hid_t dataspace = H5Dget_space(dataset);

	int dimhdf = H5Sget_simple_extent_ndims(dataspace);
	if(dimhdf != DIM){
		cerr << " Wrong dimensionality of input data: " << endl;
		cerr << dimhdf << " " << DIM << endl;
		exit(-2);
	}
	hsize_t dims_out[DIM];
	H5Sget_simple_extent_dims(dataspace, dims_out, NULL);


	for(int i=0;i<3;i++){
		lbound[i]=0;
		ubound[i]=int(dims_out[DIM-(i+1)])-1;
	}
	data.resize(lbound,ubound);
  
	return_val = H5Dread(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
	                     H5P_DEFAULT, data);

	// Close attribute ids
	H5Aclose(Origin);
	H5Aclose(Delta);

	// Close the dataset:
	H5Dclose(dataset);

	return true;
}



bool Hdf5Stream::Write3DMatrixSwap(string ArrayName, NumMatrix<double,3> &data,
                                   double *xb, double *dx)
{
	/* Routine to write NumMatix data in correct ordering to hdf5 file

	   Remarks: Only use if sufficient memory available - dummy
	   NumMatrix has to be created
	*/
	int hi[3];
	hi[0]=data.getHigh(2);
	hi[1]=data.getHigh(1);
	hi[2]=data.getHigh(0);
	int lo[3];
	lo[0] = data.getLow(2);
	lo[1] = data.getLow(1);
	lo[2] = data.getLow(0);

	NumMatrix<double,3> outdata(Index::set(lo[0],lo[1],lo[2]),
	                            Index::set(hi[0],hi[1],hi[2]));

	// Swapping data dimensions
	for(int i = lo[0]; i <= hi[0]; ++i){
		for(int j = lo[1]; j <= hi[1]; ++j){
			for(int k = lo[2]; k <= hi[2]; ++k){
				outdata(i,j,k) = data(k,j,i);
			}
		}
	}
  
	int mx[3];
	mx[2]= hi[0] - lo[0] + 1;
	mx[1]= hi[1] - lo[1] + 1;
	mx[0]= hi[2] - lo[2] + 1; 


	num+=1;
	if(num > NumEntries){
		return false;
	}
	hid_t datatype = H5Tcopy(H5T_NATIVE_DOUBLE);
	return_val = H5Tset_order(datatype, H5T_ORDER_LE);
	int DIM = 3;
	hsize_t DimsData[DIM];
	for(int q=0; q<DIM; ++q){
		DimsData[q]  = mx[q];
	}
	hid_t dataspace = H5Screate_simple(DIM, DimsData, NULL);


	// Supplying additional attributes for opendx input
	// Datatype: double, little endian of size 1
	hid_t datatypefloat = H5Tcopy(H5T_NATIVE_DOUBLE);
	return_val = H5Tset_order(datatypefloat, H5T_ORDER_LE);

	// Create dataspace for attribute
	hsize_t DimsAttr = 3;
	hid_t attrspace = H5Screate_simple(1, &DimsAttr, NULL);

	double Origin[3];
	double Delta[3];
	for(int q=0; q<3; ++q){
		Origin[q] = xb[q];
		Delta[q]  = dx[q];
	}

	// Create dataset
	hid_t dataset = H5Dcreate2(group, ArrayName.c_str(), datatype, dataspace,
	                           H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	// Create attributes
	hid_t origin = H5Acreate2(dataset, "origin", datatypefloat, attrspace,
	                          H5P_DEFAULT, H5P_DEFAULT);
	hid_t delta = H5Acreate2(dataset, "delta", datatypefloat, attrspace,
	                         H5P_DEFAULT, H5P_DEFAULT);
	// Write attributes
	return_val = H5Awrite(origin, datatypefloat, &Origin);
	return_val = H5Awrite(delta,  datatypefloat, &Delta);

	// Close attributes
	return_val = H5Aclose(origin);
	return_val = H5Aclose(delta);

	// Write data
	return_val = H5Dwrite(dataset, datatype, H5S_ALL,
	                      H5S_ALL, H5P_DEFAULT, outdata);
	return_val = H5Dclose(dataset);

	return true;
}


bool Hdf5Stream::Write2DMatrix(string ArrayName, NumMatrix<float,2> &data,
                               double *xb, double *dx, bool with_opendxinfo)
{
	/* Routine to write NumMatrix data in wrong ordering to hdf5 file.

	   Remarks:
     
	   On reading the hdf data the swapped dimensions have to be taken
	   into account
    
	*/
	int mx[2];
	mx[0]=data.getHigh(1) - data.getLow(1) + 1;
	mx[1]=data.getHigh(0) - data.getLow(0) + 1;
  
	AddDatasetName(ArrayName);

	num+=1;

	int DIM = 2;
	// Set datatype to float
	hid_t datatype = H5Tcopy(H5T_NATIVE_FLOAT);
	// Use little endian
	return_val = H5Tset_order(datatype, H5T_ORDER_LE);
  
	hsize_t DimsData[DIM];
	for(int q=0; q<DIM; ++q){
		DimsData[q]  = mx[q];
	}
	// Make dataspace:
	hid_t dataspace = H5Screate_simple(DIM, DimsData, NULL);
 
	// Supplying additional attributes for opendx input
	hsize_t DimsAttr = 2;
	hid_t attrspace = H5Screate_simple(1, &DimsAttr, NULL);
	float Origin[DIM];
	float Delta[DIM];
	if( with_opendxinfo ) {
		for(int q=0; q<DIM; ++q){
			Origin[q] = static_cast<float>(xb[q]);
			Delta[q]  = static_cast<float>(dx[q]);
		}
	}
	// Create dataset
	hid_t dataset = H5Dcreate2(group, ArrayName.c_str(), datatype, dataspace,
	                           H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

	// Writing attributes (if necessary)
	if( with_opendxinfo ) {
		hid_t origin = H5Acreate2(dataset, "origin", datatype, attrspace,
		                          H5P_DEFAULT, H5P_DEFAULT);
		return_val = H5Awrite(origin, datatype, &Origin);
		return_val = H5Aclose(origin);

		hid_t delta = H5Acreate2(dataset, "delta", datatype, attrspace,
		                         H5P_DEFAULT, H5P_DEFAULT);
		return_val = H5Awrite(delta,  datatype, &Delta);
		return_val = H5Aclose(delta);
	}

	// Write data
	return_val = H5Dwrite(dataset, datatype, H5S_ALL,
	                      H5S_ALL, H5P_DEFAULT, data);
	return_val = H5Dclose(dataset);

	return true;
}



bool Hdf5Stream::WriteArray(string ArrayName, int *data, int max)
{
	num+=1;
	if(num > NumEntries){
		return false;
	}

	// Choose int, little endian
	hid_t datatype = H5Tcopy(H5T_NATIVE_INT);
	return_val = H5Tset_order(datatype, H5T_ORDER_LE);

	int DIM = 1;
	hsize_t DimsData[DIM];
	DimsData[0] = max;//sizeof(data)/sizeof(data[0]);

	// Set up dataspace
	hid_t dataspace = H5Screate_simple(DIM, DimsData, NULL);

	// Create dataset
	hid_t dataset = H5Dcreate2(group, ArrayName.c_str(), datatype, dataspace,
	                           H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	// Write data
	return_val = H5Dwrite(dataset, datatype, H5S_ALL,
	                      H5S_ALL, H5P_DEFAULT, data);
	return_val = H5Dclose(dataset);
	return true;
}



bool Hdf5Stream::WriteArray(int *data, int max)
{
	char ArrayName[256];
	sprintf(ArrayName,"data%3.3d",num+1);
	return WriteArray(ArrayName, data, max);
}





bool Hdf5Stream::WriteArray(string ArrayName, float *data, int max)
{
	num+=1;
	if(num > NumEntries){
		return false;
	}

	// Choose float, little endian
	hid_t datatype = H5Tcopy(H5T_NATIVE_FLOAT);
	return_val = H5Tset_order(datatype, H5T_ORDER_LE);

	int DIM = 1;
	hsize_t DimsData[DIM];
	DimsData[0] = max;//sizeof(data)/sizeof(data[0]);

	// Set up dataspace
	hid_t dataspace = H5Screate_simple(DIM, DimsData, NULL);

	// Create dataset
	hid_t dataset = H5Dcreate2(group, ArrayName.c_str(), datatype, dataspace,
	                           H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	// Write data
	return_val = H5Dwrite(dataset, datatype, H5S_ALL,
	                      H5S_ALL, H5P_DEFAULT, data);
	return_val = H5Dclose(dataset);
	return true;
}



bool Hdf5Stream::WriteArray(float *data, int max)
{
	char ArrayName[256];
	sprintf(ArrayName,"data%3.3d",num+1);
	string AName = ArrayName;
	return WriteArray(AName, data, max);
}





bool Hdf5Stream::WriteArray(string ArrayName, double *data, int max)
{
	num+=1;
	if(num > NumEntries){
		return false;
	}

	// Choose double, little endian
	hid_t datatype = H5Tcopy(H5T_NATIVE_DOUBLE);
	return_val = H5Tset_order(datatype, H5T_ORDER_LE);

	int DIM = 1;
	hsize_t DimsData[DIM];
	DimsData[0] = max;//sizeof(data)/sizeof(data[0]);

	// Set up dataspace
	hid_t dataspace = H5Screate_simple(DIM, DimsData, NULL);

	// Create dataset
	hid_t dataset = H5Dcreate2(group, ArrayName.c_str(), datatype, dataspace,
	                           H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	// Write data
	return_val = H5Dwrite(dataset, datatype, H5S_ALL,
	                      H5S_ALL, H5P_DEFAULT, data);
	return_val = H5Dclose(dataset);
	return true;
}




bool Hdf5Stream::WriteArray(double *data, int max)
{
	char ArrayName[256];
	sprintf(ArrayName,"data%3.3d",num+1);
	string AName = ArrayName;
	return WriteArray(AName, data, max);
}


bool Hdf5Stream::WriteNDArray(string ArrayName, float *data, int mx[], int dim)
{
	num+=1;
	if(num > NumEntries){
		return false;
	}

	// Set datatype to float
	hid_t datatype = H5Tcopy(H5T_NATIVE_FLOAT);
	// Use little endian:
	return_val = H5Tset_order(datatype, H5T_ORDER_LE);

	hsize_t DimsData[dim];
	for(int dir=0; dir<dim; ++dir) {
		DimsData[dir] = mx[dir];
	}

	// Create a dataspace of arbitrary dimension
	hid_t dataspace = H5Screate_simple(dim, DimsData, NULL);

	// Create dataset
	hid_t dataset = H5Dcreate2(group, ArrayName.c_str(), datatype, dataspace,
	                           H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	// Write data
	return_val = H5Dwrite(dataset, datatype, H5S_ALL,
	                      H5S_ALL, H5P_DEFAULT, data);
	return_val = H5Dclose(dataset);

	return true;
}


bool Hdf5Stream::close()
{
	// Reopen attribute
	hid_t Info = H5Aopen(group, "Entries", H5P_DEFAULT);
	// Write attribute
	return_val = H5Awrite(Info, H5T_NATIVE_INT, &num);
	// Close attribute
	H5Aclose(Info);

	// Close group and file
	H5Gclose(group);
	H5Fclose(hdf5file);
	this->open = false;
	return true;
}


bool Hdf5iStream::close()
{
	//  delete group;
	// H5Gclose(group);
	// H5Fclose(hdf5file);
	return true;
}


Hdf5Stream::~Hdf5Stream()
{
	if(this->use_MPI_IO) {
		H5Pclose( plist_file_id );
		H5Pclose( plist_dset_id );
	}
	if(open) {
		close();
	}
}


Hdf5iStream::~Hdf5iStream()
{
	//  delete group and file;
	H5Gclose(group);
	H5Fclose(hdf5file);
}

bool Hdf5iStream::doesAttrExist(string name) const {

	return( H5Aexists(group, name.c_str()) > 0 ? true : false );

}
