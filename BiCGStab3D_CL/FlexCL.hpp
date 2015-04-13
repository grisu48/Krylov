#ifndef _FLEXCL_OPENCL_LIBRARY_HPP_
#define _FLEXCL_OPENCL_LIBRARY_HPP_

/* =============================================================================
 * 
 * Title:         FlexCL
 * Author:        Felix Niederwanger
 * Description:   OpenCL wrapper library
 *                This library is designed to provide an easy-to-use
 *                object-oriented approach for OpenCL. It uses the default
 *                OpenCL API and creates various wrapper objects around it.
 *                It's designed in a way to reduce the thinking overhead when
 *                dealing with memory objects on device Context and provide a 
 *                easy cleanup, so that memory leaks are less probable to occur.
 * =============================================================================
 */

/* ==== CONFIGURATION SWITCHES ===== */

// UNCOMMENT this line if you need serious debug output
#ifndef _FLEXCL_DEBUG_SWITCH_
#define _FLEXCL_DEBUG_SWITCH_ 0
#endif

#ifndef _FLEXCL_OPENGL_SUPPORT_
#define _FLEXCL_OPENGL_SUPPORT_ 1
#endif







#include <iostream>
#include <vector>
#include <string>
#include <exception>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#if _FLEXCL_OPENGL_SUPPORT_ == 1
// Compile the library with the following additional libs, if OpenGL support is included:
// "-lm -lGLU -lglut -lGLU -lGL -lGLEW"
#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>
#include <GL/glx.h>
#include <CL/cl_gl.h>
#endif

/** Build and version number for this library */
#define _FLEX_CL_BUILD_ 800
#define _FLEX_CL_VERSION_ "0.80"

/* All FlexCL classes are in this namespace */
namespace flexCL {


class OpenCLException;
class IOException;
class DeviceException;
class CompileException;
class Context;
class Program;
class Kernel;
class PlatformInfo;
class DeviceInfo;
template<class T> class CLMatrix;
template<class T> class CLMatrix2d;


/* General OpenCL Exception that is thrown from FlexCL */
class OpenCLException : public std::exception {
protected:
    /** Error message. */
    std::string _msg;
    
    cl_int err_code;
public:
    explicit OpenCLException(const char* msg, cl_int err_code = 0):_msg(std::string(msg)) { this->err_code = err_code; }
    explicit OpenCLException(std::string msg, cl_int err_code = 0):_msg(msg) { this->err_code = err_code; }
    explicit OpenCLException(cl_int err_code = 0):_msg("") { this->err_code = err_code; }
    virtual ~OpenCLException() throw () {}
    
    /** @return error message */
    std::string getMessage() { return _msg; }
    /** @return error message */
    virtual const char* what() const throw () {
    	return _msg.c_str();
    }
    /** @return returned opencl error code  */
    virtual const cl_int error_code(void) { return this->err_code; }
    /** @return OpenCL error code message as a string */
    virtual std::string opencl_error_string(void) {
		switch(err_code) {
			case CL_SUCCESS:                            return "Success";
			case CL_DEVICE_NOT_FOUND:                   return "Device not found.";
			case CL_DEVICE_NOT_AVAILABLE:               return "Device not available";
			case CL_COMPILER_NOT_AVAILABLE:             return "Compiler not available";
			case CL_MEM_OBJECT_ALLOCATION_FAILURE:      return "Memory object allocation failure";
			case CL_OUT_OF_RESOURCES:                   return "Out of resources";
			case CL_OUT_OF_HOST_MEMORY:                 return "Out of host memory";
			case CL_PROFILING_INFO_NOT_AVAILABLE:       return "Profiling information not available";
			case CL_MEM_COPY_OVERLAP:                   return "Memory copy overlap";
			case CL_IMAGE_FORMAT_MISMATCH:              return "Image format mismatch";
			case CL_IMAGE_FORMAT_NOT_SUPPORTED:         return "Image format not supported";
			case CL_BUILD_PROGRAM_FAILURE:              return "Program build failure";
			case CL_MAP_FAILURE:                        return "Map failure";
			case CL_INVALID_VALUE:                      return "Invalid value";
			case CL_INVALID_DEVICE_TYPE:                return "Invalid device type";
			case CL_INVALID_PLATFORM:                   return "Invalid platform";
			case CL_INVALID_DEVICE:                     return "Invalid device";
			case CL_INVALID_CONTEXT:                    return "Invalid context";
			case CL_INVALID_QUEUE_PROPERTIES:           return "Invalid queue properties";
			case CL_INVALID_COMMAND_QUEUE:              return "Invalid command queue";
			case CL_INVALID_HOST_PTR:                   return "Invalid host pointer";
			case CL_INVALID_MEM_OBJECT:                 return "Invalid memory object";
			case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    return "Invalid image format descriptor";
			case CL_INVALID_IMAGE_SIZE:                 return "Invalid image size";
			case CL_INVALID_SAMPLER:                    return "Invalid sampler";
			case CL_INVALID_BINARY:                     return "Invalid binary";
			case CL_INVALID_BUILD_OPTIONS:              return "Invalid build options";
			case CL_INVALID_PROGRAM:                    return "Invalid program";
			case CL_INVALID_PROGRAM_EXECUTABLE:         return "Invalid program executable";
			case CL_INVALID_KERNEL_NAME:                return "Invalid kernel name";
			case CL_INVALID_KERNEL_DEFINITION:          return "Invalid kernel definition";
			case CL_INVALID_KERNEL:                     return "Invalid kernel";
			case CL_INVALID_ARG_INDEX:                  return "Invalid argument index";
			case CL_INVALID_ARG_VALUE:                  return "Invalid argument value";
			case CL_INVALID_ARG_SIZE:                   return "Invalid argument size";
			case CL_INVALID_KERNEL_ARGS:                return "Invalid kernel arguments";
			case CL_INVALID_WORK_DIMENSION:             return "Invalid work dimension";
			case CL_INVALID_WORK_GROUP_SIZE:            return "Invalid work group size";
			case CL_INVALID_WORK_ITEM_SIZE:             return "Invalid work item size";
			case CL_INVALID_GLOBAL_OFFSET:              return "Invalid global offset";
			case CL_INVALID_EVENT_WAIT_LIST:            return "Invalid event wait list";
			case CL_INVALID_EVENT:                      return "Invalid event";
			case CL_INVALID_OPERATION:                  return "Invalid operation";
			case CL_INVALID_GL_OBJECT:                  return "Invalid OpenGL object";
			case CL_INVALID_BUFFER_SIZE:                return "Invalid buffer size";
			case CL_INVALID_MIP_LEVEL:                  return "Invalid mip-map level";
			default:									return "";
		}
	}
};

/** Exception that is associated to a device access  */
class DeviceException : public OpenCLException {
public:
    explicit DeviceException(std::string msg, cl_int err_code = 0) : OpenCLException(msg, err_code) {}
    explicit DeviceException(const char* msg, cl_int err_code = 0) : OpenCLException(msg, err_code) {}
    virtual ~DeviceException() {}
};

/** Exception when processing a IO operations  */
class IOException : public OpenCLException {
public:
    explicit IOException(std::string msg, cl_int err_code = 0) : OpenCLException(msg, err_code) {}
    explicit IOException(const char* msg, cl_int err_code = 0) : OpenCLException(msg, err_code) {}
    virtual ~IOException() {}
};

/** OpenCL compile exception  */
class CompileException : public OpenCLException {
protected:
	const char* source;
	size_t length;
	
	cl_device_id *_device_id;
	std::string _compile_output;
	
public:
    explicit CompileException(std::string msg, cl_device_id *device_id, std::string compile_output, cl_int err_code = 0) : OpenCLException(msg, err_code) {
		this->_device_id = device_id;
		this->_compile_output = compile_output;
		this->length = 0;
		this->source = "";
	}
    virtual ~CompileException() throw () {}
    
    cl_device_id* device_id() { return this->_device_id; }
    std::string compile_output() { return this->_compile_output; }
};




/** Main OpenCL class. From there it all starts */
class OpenCL {
private:
	cl_int ret;
	
	/* OpenCL identifiers */
	cl_device_id device_id = NULL;
	cl_platform_id *platform_ids = NULL;
	
	cl_uint ret_num_devices;
	cl_uint ret_num_platforms;
	
	
	std::vector<Context*> contexts;
	
protected:

	void removeContext(Context* context);

public:
	OpenCL();
	virtual ~OpenCL();
	
	void close();
	
	Context* createContext(void);
	Context* createContext(cl_platform_id, cl_device_id);
	Context* createContext(cl_platform_id);
	Context* createContext(cl_device_type device_type);
	Context* createGPUContext(void);
	Context* createCPUContext(void);
	
#if _FLEXCL_OPENGL_SUPPORT_ == 1
	/** Create context on the current OpenGL device.
	 * If glut has not yet initialized, you can set initOpenGl = true 
	 * to do a basic initialisation.
	 * A initialized OpenGL context is needed, otherwise the library can
	 * not find the used OpenGL platform and device context and the call
	 * will fail.
	 * 
	 * Important: This call will also create a command queue for the
	 * context
	 * */
	Context* createOpenGLContext(bool initOpenGl = true);
#endif
	
	unsigned int plattform_count(void);
	unsigned int device_count(void);
	
	
	static long BUILD(void);
	static std::string VERSION(void);
	
	std::vector<PlatformInfo> get_platforms(void);
	
	
	friend class Context;
	friend class Program;
	friend class Kernel;
	friend class PlatformInfo;
	friend class DeviceInfo;
};

#if _FLEXCL_OPENGL_SUPPORT_ == 1
class OpenGLBuffer {
protected:
	OpenGLBuffer(Context* context, GLuint buffer);
	OpenGLBuffer(Context* context, GLuint buffer, cl_mem mem);
	
	
	/** Associated context */
	Context* _context;

	/** OpenGL object identifier */
	GLuint _buffer;
	/** OpenCL memory object */
	cl_mem _mem;
	
	bool _aquired;
	bool _created;
	bool _closed;
	
	/** Closes this buffer object, releasing all associated memory */
	void close();
public:
	virtual ~OpenGLBuffer();

	/** Aquire OpenCL context */
	void aquire(void);
	/** Release OpenCL context, so that the object is available for OpenGL */
	void release(void);
	
	/** Checks if the buffer is aquired */
	bool isAquired(void);
	
	
	void readBuffer(size_t, void*, bool blocking = true);
	void writeBuffer(size_t, void*, bool blocking = true);
	
	
	friend class Context;
	
};
#endif

/** OpenCL self-cleaning context */
class Context {
private:
	OpenCL *owner;
	
	
	/* OpenCL Context identifiers */
	cl_program program = NULL;
	cl_context context = NULL;
	cl_command_queue command_queue = NULL;
	cl_device_id _device_id = NULL;
	cl_platform_id _platform_id = NULL;
	
	std::vector<cl_mem> buffers;
	std::vector<Program*> programs;
#if _FLEXCL_OPENGL_SUPPORT_ == 1
	std::vector<OpenGLBuffer*> openglBuffers;
#endif
	
	bool command_queue_outOfOrder = false;
	bool command_queue_profiling = false;
	
protected:
	Context(OpenCL *owner, cl_context context, cl_device_id device_id, cl_platform_id platform_id);
	
	void deleteCommandQueue(void);
	
	/** Removes a program from the program list */
	void removeProgram(Program*);
	
#if _FLEXCL_OPENGL_SUPPORT_ == 1
	/** Create cl_mem buffer from GLuint buffer */
	cl_mem createSharedOpenGLBuffer(GLuint buffer, bool readAccess = true, bool writeAccess = true);
	
	void releaseBuffer(OpenGLBuffer*);
#endif
public:
	virtual ~Context();
	
	void close();
	
	cl_device_id device_id(void);
	cl_platform_id platform_id(void);
	
	PlatformInfo platform_info(void);
	DeviceInfo device_info(void);
	
	cl_command_queue createCommandQueue(void);
	cl_command_queue createProfilingCommandQueue(void);
	cl_command_queue createCommandQueue(bool outOfOrder, bool profiling);
	
	cl_mem createBuffer(size_t size, void* host_ptr = NULL);
	cl_mem createReadBuffer(size_t size, void* host_ptr = NULL);
	cl_mem createWriteBuffer(size_t size, void* host_ptr = NULL);
	
#if _FLEXCL_OPENGL_SUPPORT_ == 1
	/** Create shared OpenCL buffer from OpenGL buffer.
	 * This call must take place after glBindBuffer
	 * */
	OpenGLBuffer* createGLBuffer(GLuint buffer);
	/** Create shared OpenGL buffer and associates a OpenCL buffer*/
	OpenGLBuffer* createGLBuffer(size_t, const GLvoid* data = NULL, bool isStatic = false);
#endif

	void releaseBuffer(cl_mem buffer);
	void deleteBuffer(cl_mem buffer);
	void releaseProgram(Program* program);
	void deleteProgram(Program* program);
	
	void writeBuffer(cl_mem buffer, size_t size, void* ptr, bool blockingWrite = true, size_t offset = 0);
	unsigned long writeBufferProfiling(cl_mem buffer, size_t size, void* ptr, size_t offset = 0);
	
	std::string get_compile_output(cl_program program);
	
	Program* createProgramFromSource(std::string source);
	Program* createProgramFromSource(const char *source, size_t length);
	Program* createProgramFromSourceFile(const char *filename);
	Program* createProgramFromSourceFile(std::string filename);
	
	Program* createProgramFromBinary(std::string source);
	Program* createProgramFromBinary(const unsigned char *source, size_t length);
	Program* createProgramFromBinaryFile(const char *filename);
	Program* createProgramFromBinaryFile(std::string filename);
	
	void readBuffer(cl_mem buffer, size_t size, void *dst_ptr, bool blockingRead = true, size_t offset = 0);
	void readBufferBlocking(cl_mem buffer, size_t size, void *dst_ptr, size_t offset = 0);
	unsigned long readBufferProfiling(cl_mem buffer, size_t size, void *dst_ptr, size_t offset = 0);
	
	void copyBuffer(cl_mem dst, cl_mem src, size_t size, size_t src_offset = 0, size_t dst_offset = 0);

	void flush(void);
	void join(void);
	
	/** Enqueue a barrier that ensures that all precedent commands in the queue are
	 * completed, before any following commands are processed. */
	void barrier(void);
	
	bool isOutOfOrder(void);
	bool isProfiling(void);


	friend class OpenCL;
	friend class Program;
	friend class Kernel;
	friend class PlatformInfo;
	friend class DeviceInfo;
	friend class OpenGLBuffer;
	template <class T>friend class CLMatrix;
};

/** OpenCL Platform information class */
class PlatformInfo {
protected:
	PlatformInfo(cl_platform_id platform_id);

	cl_platform_id _platform_id;
	
	std::string _profile;
	std::string _version;
	std::string _name;
	std::string _vendor;
	std::string _extensions;
	
public:
	virtual ~PlatformInfo();
	
	std::string profile();
	std::string version();
	std::string name();
	std::string vendor();
	std::string extensions();
	
	cl_platform_id platform_id();
	
	std::vector<DeviceInfo> devices();
	
	friend class OpenCL;
	friend class Context;
};

/** OpenCL device information class */
class DeviceInfo {
protected:
	DeviceInfo(cl_device_id device_id);
	
	cl_device_id _device_id;
	
	int _device_type = 0;
	
	
	/** Device info as unsigned int*/
	cl_uint getDeviceInfo_ui(cl_device_info param_name);
	/** Device info as int*/
	cl_int getDeviceInfo_i(cl_device_info param_name);
	/** Device info as unsigned long*/
	cl_ulong getDeviceInfo_ul(cl_device_info param_name);
	/** Device info as long*/
	cl_long getDeviceInfo_l(cl_device_info param_name);
	/** Device-info as size_t */
	size_t getDeviceInfo_size_t(cl_device_info param_name);
public:
	virtual ~DeviceInfo();
	cl_device_id device_id();
	
	bool isCPU(void);
	bool isGPU(void);
	bool isAccelerator(void);
	
	std::string name();
	std::string vendor();
	std::string extensions();
	unsigned long timerResolution();
	/** @return max size of memory object allocation in bytes */
	unsigned long maxMemAllocSize();
	/** @return max number of computation units */
	unsigned int maxComputeUnits();
	std::string deviceVersion();
	std::string driverVersion();
	std::string deviceOpenCLVersion();
	unsigned int addressBits();
	unsigned long globalMemSize();
	unsigned long globalMemCacheSize();
	unsigned long localMemSize();
	//std::string local_mem_type();
	size_t maxWorkGroupSize(void);

	
	bool hasImageSupport(void);

	/** Maximum configured clock frequency in MHz */
	unsigned int maxClockFrequency(void);
	unsigned int maxConstantArguments(void);
	unsigned long constantBufferSize(void);
	size_t maxParameterSize(void);
	
	
	/** Query a given cl_device info. A complete list of the available options is found at
	 * https://www.khronos.org/registry/cl/sdk/1.0/docs/man/xhtml/clGetDeviceInfo.html
	 * */
	std::string getDeviceInfo(cl_device_info param_name);
	
	friend class OpenCL;
	friend class Context;
	friend class PlatformInfo;
};

/** A OpenCL program that is associated to a certain context */
class Program {
private:
	Context *context;
	cl_program program = NULL;
	
	
	std::vector<Kernel*> kernels;
	std::vector<cl_mem> program_buffers;
	
protected:
	Program(Context *context, cl_program program);
	
	/** Remove a kernel from the program list */
	void removeKernel(Kernel*);
public:
	virtual ~Program();
	
	Kernel* createKernel(std::string func_name);
	Kernel* createKernel(const char* func_name);
	
	/** Create buffer on the context for this program */
	cl_mem createBuffer(size_t size, void* host_ptr = NULL);
	/** Create read-only buffer on the context for this program */
	cl_mem createReadBuffer(size_t size, void* host_ptr = NULL);
	/** Create write-only buffer on the context for this program */
	cl_mem createWriteBuffer(size_t size, void* host_ptr = NULL);
	
	/** Clean the program removing all kernels and program buffers */
	void cleanup();
	
	/** Get the parent context instance */
	Context* getContext();
	
	friend class OpenCL;
	friend class Context;
	friend class Kernel;
};

/** OpenCL Kernel, assosciated to a ceratin OpenCL program */
class Kernel {
private:
	Program *program;
	cl_kernel kernel;
	cl_event perf_event;
	
	cl_uint arg_index = 0;
protected:
	Kernel(Program *program, cl_kernel kernel);	
	
	/* Profiling times */
	unsigned long profiling_times[4];
	/* Profile infos */
	void collect_profile_infos(void);
	/* Indicating if the profile infos are already collected */
	bool profile_infos_collected = false;
	
	cl_command_queue command_queue();
	bool isProfiling(void);
public:
	virtual ~Kernel();
	
	void setArgument(unsigned int index, cl_mem *arg_ptr);
	void setArgument(unsigned int index, cl_mem &arg_ptr);
	void setArgument(unsigned int index, size_t size, const void* arg_ptr);
	void setArgumentLocalMem(unsigned int index, size_t size);
	void setArgument(unsigned int index, float arg);
	void setArgument(unsigned int index, double arg);
	void setArgument(unsigned int index, int arg);
	void setArgument(unsigned int index, long arg);
	void setArgument(unsigned int index, unsigned char arg);
	void setArgument(unsigned int index, unsigned long arg);
	void addArgument(size_t size, const void* arg_ptr);
	void addArgument(float arg);
	void addArgument(double arg);
	void addArgument(int arg);
	void addArgument(long arg);
	void addArgument(cl_mem *arg_ptr);
	void addArgument(cl_mem &arg_ptr);
	void addArgumentLocalMem(size_t size);
	
	unsigned int getArgumentCount();
	
	void enqueue();
	void enqueueNDRange(unsigned int work_dim, const size_t *global_work_size);
	void enqueueNDRange(unsigned int work_dim, const size_t *global_work_size, const size_t *local_work_size);
	void enqueueNDRange(size_t dim1);
	void enqueueNDRange(size_t dim1, size_t dim2);
	void enqueueNDRange(size_t dim1, size_t dim2, size_t dim3);
	
	/** Runtime in nanoseconds (1e-9)*/
	unsigned long runtime(void);
	/** Total runtime since queued in nanoseconds (1e-9)*/
	unsigned long total_runtime(void);
	
	/** Get the parent program instance */
	Program* getProgram();
	/** Get the parent context instance */
	Context* getContext();
	

	size_t getKernelWorkGroupSize(void);
	size_t getLocalMemSize(void);
	size_t getPreferredWorkGroupSizeMultiple(void);

	friend class OpenCL;
	friend class Context;
	friend class Program;
};


}


#endif
