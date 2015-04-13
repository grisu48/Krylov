
/* =============================================================================
 * 
 * Title:         FlexCL
 * Author:        Felix Niederwanger
 * Description:   OpenCL wrapper library providing object-oriented access to
 *                OpenCL
 * 
 * =============================================================================
 */

#include "FlexCL.hpp"
#include <string>
#include <sstream>
#include <fstream>
#include <streambuf>
#include <algorithm> 
#include <functional> 
#include <cctype>
#include <locale>

using namespace flexCL;
using namespace std;


/* ==== General useful stuff ======================================= */


#if __cplusplus < 201103L

#define SSTR( x ) dynamic_cast< std::ostringstream & >( \
        ( std::ostringstream() << std::dec << x ) ).str()

string to_string(int value) { return SSTR(value); }
string to_string(unsigned int value) { return SSTR(value); }
string to_string(float value) { return SSTR(value); }
string to_string(long value) { return SSTR(value); }
string to_string(unsigned long value) { return SSTR(value); }
string to_string(double value) { return SSTR(value); }

#endif

// Trim from left
static string _flexCL_ltrim(string s) {
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
        return s;
}

// trim from end
static string _flexCL_rtrim(string s) {
        s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
        return s;
}

// trim from both ends
static string _flexCL_trim(string s) {
        return _flexCL_ltrim(_flexCL_rtrim(s));
}


// Static method for removing an item from a vector by it's value
template<typename T>
static void removeFromVector(vector<T>& vec, T &value) {
	// Iterate while the values is still in the vector
	bool found = false;
	do {
		found = false;
		for(typename vector<T>::iterator it = vec.begin(); it != vec.end(); ++it) {
			if(*it == value) {
				found = true;
				vec.erase(it);
				break;
			}
		}
	} while(found);
}


/* ==== Static OpenCL routines ====================================== */

/* Return code check routine */
static inline void checkReturn(cl_int ret, const char* msg) {
	if(ret != CL_SUCCESS) throw OpenCLException(msg, ret);
}

/* Return code check routine */
static inline void checkReturn(cl_int ret, string msg) {
	if(ret != CL_SUCCESS) throw OpenCLException(msg, ret);
}


static inline unsigned long _flexcl_profile_info(cl_event &perf_event, cl_profiling_info info) {
	cl_ulong result;
	clGetEventProfilingInfo(perf_event, info, sizeof(cl_ulong), &result, NULL);
	return (unsigned long) result;
}


/* ==== Here the OpenCL part starts ================================= */

OpenCL::OpenCL() {
	
	try {
		ret_num_devices = 0;
		ret = clGetPlatformIDs(0, NULL, &ret_num_platforms);
		// ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
		checkReturn(ret, "Error getting platform count");
		
		// Get all platform ID's
		platform_ids = new cl_platform_id[(int)ret_num_platforms];
		ret = clGetPlatformIDs(ret_num_platforms, platform_ids, NULL);
		
		//ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);
		checkReturn(ret, "Getting devices failed");
		
	} catch (...) {
		// Cleanup 
		throw;
	}
}

OpenCL::~OpenCL() {
	this->close();
}


long OpenCL::BUILD(void) {
	return _FLEX_CL_BUILD_;
}
string OpenCL::VERSION(void) {
	return _FLEX_CL_VERSION_;
}

void OpenCL::removeContext(Context* context) {
	removeFromVector(contexts, context);
}

void OpenCL::close() {
#if _FLEXCL_DEBUG_SWITCH_ == 1
	cout << "OpenCL::Close" << endl;
#endif
	if(platform_ids != NULL) delete[] platform_ids;
	platform_ids = NULL;


	while(contexts.size() > 0) {
		Context* context = contexts.at(0);
		delete context;
		removeFromVector(contexts, context);
	}
}

vector<PlatformInfo> OpenCL::get_platforms(void) {
	vector<PlatformInfo> result;
	
	const int len = (int)ret_num_platforms;
	for(int i=0;i<len;i++) {
		result.push_back(PlatformInfo(platform_ids[i]));
	}
	
	return result;
}

Context* OpenCL::createGPUContext(void) {
	return createContext(CL_DEVICE_TYPE_GPU);
}

Context* OpenCL::createContext(void) {
	return createContext(CL_DEVICE_TYPE_DEFAULT);
}

Context* OpenCL::createContext(cl_platform_id platform_id, cl_device_id device_id) {
#if _FLEXCL_DEBUG_SWITCH_ == 1
	cout << "OpenCL::createContext(platform_id = " << platform_id << ", device_id = " << device_id << ")" << endl;
#endif
	cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
	switch(ret) {
		case CL_SUCCESS: break;
		case CL_INVALID_PLATFORM: throw DeviceException("Invalid platform");
		case CL_INVALID_VALUE: throw DeviceException("Invalid context properties");
		case CL_DEVICE_NOT_AVAILABLE: throw DeviceException("No GPU device available");
		case CL_DEVICE_NOT_FOUND: throw DeviceException("No GPU device found");
		case CL_OUT_OF_HOST_MEMORY: throw DeviceException("Out of memory");
		case CL_INVALID_DEVICE_TYPE: throw DeviceException("Device type is invalid");
		default: throw DeviceException("Unknwon error while creating context");
	}
	// Ok - The OpenCL context is created sucessfully.
	
	Context *context_Obj = new Context(this, context, device_id, platform_id);
	contexts.push_back(context_Obj);
	return context_Obj;
}

Context* OpenCL::createContext(cl_platform_id p_id) {
#if _FLEXCL_DEBUG_SWITCH_ == 1
	cout << "OpenCL::createContext(platform = " << p_id << ")" << endl;
#endif
	cl_device_id device_id = NULL;
	cl_platform_id platform_id = NULL;
	cl_uint num_devices;
	
	
	ret = clGetDeviceIDs( p_id, CL_DEVICE_TYPE_ALL, 1, &device_id, &num_devices);
	checkReturn(ret, "Querying devices failed");
	if(num_devices > 0) {
		platform_id = p_id;
	} else
		throw DeviceException("Device type not found");
	
	cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
	switch(ret) {
		case CL_SUCCESS: break;
		case CL_INVALID_PLATFORM: throw DeviceException("Invalid platform");
		case CL_INVALID_VALUE: throw DeviceException("Invalid context properties");
		case CL_DEVICE_NOT_AVAILABLE: throw DeviceException("No GPU device available");
		case CL_DEVICE_NOT_FOUND: throw DeviceException("No GPU device found");
		case CL_OUT_OF_HOST_MEMORY: throw DeviceException("Out of memory");
		case CL_INVALID_DEVICE_TYPE: throw DeviceException("Device type is invalid");
		default: throw DeviceException("Unknwon error while creating context");
	}
	// Ok - The OpenCL context is created sucessfully.
	
	Context *context_Obj = new Context(this, context, device_id, platform_id);
	contexts.push_back(context_Obj);
	return context_Obj;
}

Context* OpenCL::createContext(cl_device_type device_type) {
#if _FLEXCL_DEBUG_SWITCH_ == 1
	cout << "OpenCL::createContext(device_type = " << device_type << ")" << endl;
#endif
	cl_device_id device_id = NULL;
	cl_platform_id platform_id = NULL;
	cl_uint num_devices;
	
	for(cl_uint id = 0; id<ret_num_platforms;id++) {
		cl_device_id dev_id;
		ret = clGetDeviceIDs( platform_ids[id], device_type, 1, &dev_id, &num_devices);
		// XXX: Ugly temporary hack!!
		//checkReturn(ret, "Querying devices failed");
		if(ret == 0 && num_devices > 0) {
			platform_id = platform_ids[id];
			device_id = dev_id;
			break;
		}
	} 
	
	if(device_id == NULL || platform_id == NULL) throw DeviceException("Device type not found");
	
	/*cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (int)platform_id, 0};
	cl_context context = clCreateContextFromType(properties, CL_DEVICE_TYPE_GPU, NULL, NULL, &ret);
	*/
	cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
	switch(ret) {
		case CL_SUCCESS: break;
		case CL_INVALID_PLATFORM: throw DeviceException("Invalid platform");
		case CL_INVALID_VALUE: throw DeviceException("Invalid context properties");
		case CL_DEVICE_NOT_AVAILABLE: throw DeviceException("No GPU device available");
		case CL_DEVICE_NOT_FOUND: throw DeviceException("No GPU device found");
		case CL_OUT_OF_HOST_MEMORY: throw DeviceException("Out of memory");
		case CL_INVALID_DEVICE_TYPE: throw DeviceException("Device type is invalid");
		default: throw DeviceException("Unknwon error while creating context");
	}
	// Ok - The OpenCL context is created sucessfully.
	
	Context *context_Obj = new Context(this, context, device_id, platform_id);
	contexts.push_back(context_Obj);
	return context_Obj;
	
}

Context* OpenCL::createCPUContext(void) {
	return this->createContext(CL_DEVICE_TYPE_CPU);
}

#if _FLEXCL_OPENGL_SUPPORT_ == 1
Context* OpenCL::createOpenGLContext(bool initOpenGl) {
	// Initialize GLUT, if desired
	if(initOpenGl) {
		int argc = 0;
		glutInit(&argc, NULL);
		glutCreateWindow("");
		
		GLenum res = glewInit();
		if (res != GLEW_OK)
		{
			string error = "Glew error";
			char* glewError = (char*)glewGetErrorString(res);
			error = error + string( glewError );
			throw OpenCLException(error);
		}
    }
    
    /* ==== Setting up OpenCL and query OpenGL shared devices==== */
	const int max_devices = 1024;
	cl_int state;
	cl_context context;
	cl_command_queue queue;
	cl_platform_id platform[max_devices];
	cl_device_id devices[max_devices];

	//Get platform
	cl_uint numberOfPlatforms = 0;
	state = clGetPlatformIDs(max_devices, platform, &numberOfPlatforms);
	if(state != CL_SUCCESS)
		throw OpenCLException("Device resolution failed", state);
	if(numberOfPlatforms <= 0)
		throw OpenCLException("No shared OpenCL/OpenGL devices found (is GLUT initialized?)", state);

	//Parameters needed to bind OpenGL's context to OpenCL's.
	cl_context_properties properties[] = {	CL_GL_CONTEXT_KHR, (cl_context_properties) glXGetCurrentContext(),
						CL_GLX_DISPLAY_KHR, (cl_context_properties) glXGetCurrentDisplay(),
						CL_CONTEXT_PLATFORM, (cl_context_properties) platform[0],
						0};

	//Find openGL devices.
	typedef CL_API_ENTRY cl_int (CL_API_CALL *CLpointer)(const cl_context_properties *properties, cl_gl_context_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret);
	CL_API_ENTRY cl_int (CL_API_CALL *myCLGetGLContextInfoKHR)(const cl_context_properties *properties, cl_gl_context_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret) = (CLpointer)clGetExtensionFunctionAddressForPlatform(platform[0], "clGetGLContextInfoKHR");

	size_t size;
	state = myCLGetGLContextInfoKHR(properties, CL_DEVICES_FOR_GL_CONTEXT_KHR, max_devices*sizeof(cl_device_id), devices, &size);

	if(state != CL_SUCCESS) {
		throw OpenCLException("Device resolution failed (clGetGLContextInfoKHR)", state);
	}
	
	const int numberSharedDevices = (int)(size/sizeof(cl_device_id));
	if(numberSharedDevices <= 0) {
		throw OpenCLException("No shared OpenCL/OpenGL devices found (number of shared devices = 0)", state);
	}
	
	context = clCreateContext(properties, 1, &devices[0], NULL, NULL, &state);
	if(state != CL_SUCCESS) {
		throw OpenCLException("Context creation failed", state);
	}
		
	
	// Create profiling command queue
	cl_command_queue_properties queue_props = CL_QUEUE_PROFILING_ENABLE;
	queue = clCreateCommandQueue(context, devices[0], queue_props, &state);
	if(state != CL_SUCCESS) {
		throw OpenCLException("Command queue creation failed", state);
	}
	
	Context *context_Obj = new Context(this, context, devices[0], platform[0]);
	context_Obj->context = context;
	context_Obj->command_queue = queue;
	return context_Obj;
}
#endif

unsigned int OpenCL::plattform_count(void) {
	return (unsigned int)ret_num_platforms;
}

unsigned int OpenCL::device_count(void) {
	return (unsigned int)ret_num_devices;
}
















Context::Context(OpenCL *owner, cl_context context, cl_device_id device_id, cl_platform_id platform_id) {
	this->owner = owner;
	this->context = context;
	this->_device_id = device_id;
	this->_platform_id = platform_id;
	
	createCommandQueue();
}

Context::~Context() {
#if _FLEXCL_DEBUG_SWITCH_ == 1
	cout << "OpenCL::Context closing" << endl;
#endif
	this->close();
}

void Context::deleteCommandQueue(void) {
#if _FLEXCL_DEBUG_SWITCH_ == 1
	cout << "OpenCL::Context deleteCommandQueue()" << endl;
#endif
	if (command_queue != NULL) {
		clFlush(command_queue);
		clFinish(command_queue);
		clReleaseCommandQueue(command_queue);
	}
	command_queue = NULL;
}


void Context::flush() {
#if _FLEXCL_DEBUG_SWITCH_ == 1
	cout << "OpenCL::Context::flush()" << endl;
#endif
	if (command_queue != NULL)
		clFlush(command_queue);
}

void Context::join() {
	if (command_queue != NULL) {
#if _FLEXCL_DEBUG_SWITCH_ == 1
		cout << "OpenCL::Context::join()" << endl;
#endif
		/* Commented out, since the flush is implicit given by
		 * clFinish */
		clFlush(command_queue);
		clFinish(command_queue);
	}
#if _FLEXCL_DEBUG_SWITCH_ == 1
	else
		cout << "OpenCL::Context::join() -- WARN: no command queue" << endl;
#endif
}


void Context::barrier(void) {
#if _FLEXCL_DEBUG_SWITCH_ == 1
	cout << "OpenCL::Context::barrier()" << endl;
#endif
	cl_int ret;
	ret =  clEnqueueBarrier(this->command_queue);
	checkReturn(ret, "Error enqueuing barrier in command queue");
}

void Context::releaseBuffer(cl_mem buffer) {
#if _FLEXCL_DEBUG_SWITCH_ == 1
	cout << "OpenCL::Context::releaseBuffer(" << buffer << ")" << endl;
#endif
	clReleaseMemObject(buffer);
	removeFromVector(this->buffers, buffer);
}

void Context::deleteBuffer(cl_mem buffer) {
	this->releaseBuffer(buffer);
}

void Context::copyBuffer(cl_mem dst, cl_mem src, size_t size, size_t src_offset, size_t dst_offset) {
#if _FLEXCL_DEBUG_SWITCH_ == 1
	cout << "OpenCL::Context::copyBuffer(" << dst << "," << src << "," << size << "," << src_offset << "," << dst_offset << ")" << endl;
#endif
	cl_int ret;

	ret = clEnqueueCopyBuffer(this->command_queue, src, dst, src_offset, dst_offset, size, 0, NULL, NULL);
	checkReturn(ret, "Error copying buffer");
}


void Context::releaseProgram(Program* program) {
#if _FLEXCL_DEBUG_SWITCH_ == 1
	cout << "OpenCL::Context::releaseProgramm(" << program->program << ")" << endl;
#endif
	clReleaseProgram(program->program);
	removeFromVector(this->programs, program);
}

void Context::deleteProgram(Program* program) {
	this->releaseProgram(program);
}

void Context::close() {
#if _FLEXCL_DEBUG_SWITCH_ == 1
	cout << "OpenCL::Context::close()" << endl;
#endif
	// The cleaning sequence matters. Remember that the programs will do a cleanup as well
	// removing their associated memory buffers. This MUST take place before removing the remaining
	// memory objects
	while(programs.size() > 0)
		this->releaseProgram(programs.at(0));

	/*
	for(vector<Program*>::iterator it = programs.begin(); it != programs.end(); ++it)
		delete *it;
	programs.clear();
	*/
	
	// Clear shared OpenGL/OpenCL buffers
#if _FLEXCL_OPENGL_SUPPORT_ == 1
	/*for(vector<OpenGLBuffer*>::iterator it = openglBuffers.begin(); it != openglBuffers.end(); ++it) {
		OpenGLBuffer *buffer = *it;
		buffer->close();
		delete buffer;
	}
	openglBuffers.clear(); */
	while(openglBuffers.size() > 0)
		this->releaseBuffer(openglBuffers.at(0));
#endif
	
	/*
	for(vector<cl_mem>::iterator it = buffers.begin(); it != buffers.end(); ++it)
		clReleaseMemObject(*it);
	buffers.clear();
	*/
	while(this->buffers.size() > 0)
		this->releaseBuffer(this->buffers.at(0));
	
	deleteCommandQueue();
	if (context != NULL) clReleaseContext(context);
	context = NULL;

	this->owner->removeContext(this);
}

cl_command_queue Context::createCommandQueue(void) {
	return this->createCommandQueue(false, false);
}
cl_command_queue Context::createProfilingCommandQueue(void) {
	return this->createCommandQueue(false, true);
}

cl_command_queue Context::createCommandQueue(bool outOfOrder, bool profiling) {
#if _FLEXCL_DEBUG_SWITCH_ == 1
	cout << "OpenCL::Context::createCommandQueue(";
	if(outOfOrder) cout << "OutOfOrder";
	else cout << "InOrder";
	cout << ",";
	if(profiling) cout << "Profiling";
	else cout << "NonProfile";
	cout << ")" << endl;
#endif
	cl_int ret;
	
	cl_command_queue_properties properties(0);
	if(outOfOrder) properties |= CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
	if(profiling)  properties |= CL_QUEUE_PROFILING_ENABLE;
	deleteCommandQueue();
	this->command_queue = clCreateCommandQueue(this->context, this->_device_id, properties, &ret);
	this->command_queue_outOfOrder = outOfOrder;
	this->command_queue_profiling = profiling;
	checkReturn(ret, "Creating queue failed");
	return this->command_queue;
}

cl_mem Context::createBuffer(size_t size, void* host_ptr) {
#if _FLEXCL_DEBUG_SWITCH_ == 1
	if(host_ptr == NULL)
		cout << "OpenCL::Context::createBuffer(" << size << " Bytes)" << endl;
	else
		cout << "OpenCL::Context::createBuffer(" << size << " Bytes, Copy from host)" << endl;
#endif
	cl_mem buffer;
	cl_int ret;
	
	if(host_ptr == NULL)
		buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, &ret);
	else
		buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size, host_ptr, &ret);
	checkReturn(ret, "Creating buffer failed");
	this->buffers.push_back(buffer);
	return buffer;
}

cl_mem Context::createReadBuffer(size_t size, void* host_ptr) {
#if _FLEXCL_DEBUG_SWITCH_ == 1
	if(host_ptr == NULL)
		cout << "OpenCL::Context::createReadBuffer(" << size << " Bytes)" << endl;
	else
		cout << "OpenCL::Context::createReadBuffer(" << size << " Bytes, Copy from host)" << endl;
#endif
	cl_mem buffer;
	cl_int ret;
	
	if(host_ptr == NULL)
		buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, size, NULL, &ret);
	else
		buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, host_ptr, &ret);
	checkReturn(ret, "Creating queue failed");
	return buffer;
}

cl_mem Context::createWriteBuffer(size_t size, void* host_ptr) {
#if _FLEXCL_DEBUG_SWITCH_ == 1
	if(host_ptr == NULL)
		cout << "OpenCL::Context::createWriteBuffer(" << size << " Bytes)" << endl;
	else
		cout << "OpenCL::Context::createWriteBuffer(" << size << " Bytes, Copy from host)" << endl;
#endif
	cl_mem buffer;
	cl_int ret;
	
	buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, size, host_ptr, &ret);
	checkReturn(ret, "Creating queue failed");
	return buffer;
	
}

void Context::writeBuffer(cl_mem buffer, size_t size, void* ptr, bool blockingWrite, size_t offset) {
#if _FLEXCL_DEBUG_SWITCH_ == 1
	cout << "OpenCL::Context::writeBuffer(" << size << " Bytes";
	if(blockingWrite) cout << ", BLOCKING";
	cout << ")" << endl;
#endif
	cl_int ret;
	
	ret = clEnqueueWriteBuffer(this->command_queue, buffer, (blockingWrite?CL_TRUE:CL_FALSE), offset, size, ptr, 0, NULL, NULL);
	checkReturn(ret, "Enqueue write buffer failed");
}

unsigned long Context::writeBufferProfiling(cl_mem buffer, size_t size, void* ptr, size_t offset) {
#if _FLEXCL_DEBUG_SWITCH_ == 1
	cout << "OpenCL::Context::writeBuffer(" << size << " Bytes, Profiling)" << endl;
#endif
	cl_int ret;
	cl_event perf_event;
	
	ret = clEnqueueWriteBuffer(this->command_queue, buffer, CL_TRUE, offset, size, ptr, 0, NULL, &perf_event);
	checkReturn(ret, "Enqueue write buffer failed");
	clWaitForEvents(1, &perf_event);
	
	cl_ulong start = 0, end = 0;
	
	start = _flexcl_profile_info(perf_event, CL_PROFILING_COMMAND_START);
	end = _flexcl_profile_info(perf_event, CL_PROFILING_COMMAND_END);
	// clGetEventProfilingInfo(perf_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	// clGetEventProfilingInfo(perf_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
	
	return (unsigned long)(end-start);
}

std::string Context::get_compile_output(cl_program program) {
	cl_build_status status;
	size_t logSize;
	// Get build status
	clGetProgramBuildInfo(program, _device_id, CL_PROGRAM_BUILD_STATUS,sizeof(cl_build_status), &status, NULL);
	
	// Build log
	clGetProgramBuildInfo(program, _device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
	char* log = new char[logSize+1];
	std::string result_log;
	try {
		clGetProgramBuildInfo(program, _device_id, CL_PROGRAM_BUILD_LOG, logSize+1, log, NULL);

		result_log = std::string(log);
	
		delete[] log;
	} catch (...) {
		delete[] log;
		throw;
	}
	
	return result_log;
}

Program* Context::createProgramFromBinary(string source) {
	return this->createProgramFromBinary((const unsigned char *)source.c_str(), source.length());
}

Program* Context::createProgramFromBinary(const unsigned char *source, size_t length) {
#if _FLEXCL_DEBUG_SWITCH_ == 1
	cout << "OpenCL::Context::createProgramFromBinary( ... ," << length << ")" << endl;
#endif
	cl_int ret;
	cl_int binaryStatus;

	program = clCreateProgramWithBinary(context,
		1,
		&_device_id,
		(const size_t *)&length,
		(const unsigned char **)&source,
		&binaryStatus,
		&ret);
	checkReturn(ret, "Error loading program binary");
	checkReturn(ret, "Invalid binary for device");
	ret = clBuildProgram(program, 1, &_device_id, NULL, NULL, NULL);
	if(ret != CL_SUCCESS) {
		string compile_output = this->get_compile_output(program);
		clReleaseProgram(program);
		throw CompileException("Building OpenCL program failed", &_device_id, compile_output, ret);
	}
	
	Program* program_Obj = new Program(this, program);
	this->programs.push_back(program_Obj);
	return program_Obj;
}

Program* Context::createProgramFromBinaryFile(const char *filename) {
	ifstream in_file;
	in_file.open(filename);
	string contents;
	try {
		contents = string((istreambuf_iterator<char>(in_file)), istreambuf_iterator<char>());
		in_file.close();
		
	} catch (...) {
		in_file.close();
		throw;
	}
	
	return this->createProgramFromBinaryFile(contents);
	
}

Program* Context::createProgramFromBinaryFile(std::string filename) {
	return this->createProgramFromBinaryFile(filename.c_str());
}


Program* Context::createProgramFromSource(std::string source) {
	return this->createProgramFromSource(source.c_str(), source.length());
}

Program* Context::createProgramFromSource(const char *kernel_source, size_t length) {
#if _FLEXCL_DEBUG_SWITCH_ == 1
	cout << "OpenCL::Context::createProgramFromSource( ... ," << length << ")" << endl;
#endif
	cl_int ret;
	program = clCreateProgramWithSource(context, 1, (const char **)&kernel_source, (const size_t *)&length, &ret);
	checkReturn(ret, "Creating program from source failed");
	ret = clBuildProgram(program, 1, &_device_id, NULL, NULL, NULL);
	if(ret != CL_SUCCESS) {
		string compile_output = this->get_compile_output(program);
		clReleaseProgram(program);
		throw CompileException("Building OpenCL program failed", &_device_id, compile_output, ret);
	}
	
	Program* program_Obj = new Program(this, program);
	this->programs.push_back(program_Obj);
	return program_Obj;
}

/** Reads a source file and returns it's value
 * This method also supports the #include directive
 * */
static string _flexCL_readSourceFile(const char* filename) {
	stringstream buffer;
	
	ifstream in_file;
	in_file.open(filename);
	if(!in_file.is_open()) {
		string errmsg = "Cannot open file " + string(filename);
		throw IOException(errmsg);
	}
	string line;
	string include_header = "#include ";
	try {
		unsigned long line_no = 0;
		bool firstLine = true;
		while(! in_file.eof()) {
			// Read line by line
			getline(in_file,line);
			line_no++;
			if(firstLine)
				firstLine = false;
			else
				buffer << '\n';
			// Special hanlding (#include)
			if(line.substr(0,include_header.length()) == include_header) {
				string include_filename = line.substr(include_header.length());
				include_filename = _flexCL_trim(include_filename);
				
				// check filename syntax
				bool localInclude;
				if(include_filename.at(0) == '<' && include_filename.at(include_filename.length()-1) == '>') {
					localInclude = false;
				} else if(include_filename.at(0) == '\"' && include_filename.at(include_filename.length()-1) == '\"') {
					localInclude = true;
				} else {
					string errmsg = string(filename) + " - line " + ::to_string(line_no) + ": #include statement has wrong syntax";
					throw IOException(errmsg);
				}
				
				include_filename = include_filename.substr(1, include_filename.length()-2);
				include_filename = _flexCL_trim(include_filename);
				
				// Search for file, if not a local include
				if(!localInclude) {
					// TODO: This function will be implemented in a future release
				}
				
				// #include the given file
				buffer << _flexCL_readSourceFile(include_filename.c_str());
			} else {
				// Simply add buffer line
				buffer << line;
			}
		}
		in_file.close();
		
	} catch (...) {
		in_file.close();
		throw;
	}
	
	return buffer.str();
}

Program* Context::createProgramFromSourceFile(const char *filename)  {
#if _FLEXCL_DEBUG_SWITCH_ == 1
	cout << "OpenCL::Context::createProgramFromSourceFile(" << filename << ")" << endl;
#endif
	// Deprecated. We now use the _flexCL_readSourceFile function
	/*
	ifstream in_file;
	in_file.open(filename);
	if(!in_file.is_open()) throw IOException("Cannot open file");
	string contents;
	try {
		contents = string((istreambuf_iterator<char>(in_file)), istreambuf_iterator<char>());
		in_file.close();
		
	} catch (...) {
		in_file.close();
		throw;
	} */
	
	string contents = _flexCL_readSourceFile(filename);
	return this->createProgramFromSource(contents);
}

Program* Context::createProgramFromSourceFile(std::string filename)  {
	return this->createProgramFromSourceFile(filename.c_str());
}

void Context::readBuffer(cl_mem buffer, size_t size, void *dst_ptr, bool blockingRead, size_t offset) {
#if _FLEXCL_DEBUG_SWITCH_ == 1
	cout << "OpenCL::Context::readBuffer(" << size << " Bytes";
	if(blockingRead) cout << ",BLOCKING";
	cout << ") ... ";
	cout.flush();
#endif
	cl_int ret;
	cl_bool blocking_read = (blockingRead?CL_TRUE:CL_FALSE);
	
	ret = clEnqueueReadBuffer(command_queue, buffer, blocking_read, offset, size, dst_ptr, 0, NULL, NULL);
#if _FLEXCL_DEBUG_SWITCH_ == 1
	if(ret == CL_SUCCESS)
		cout << " OK " << endl;
	else
		cout << " ERR = " << ret << endl;
#endif
	checkReturn(ret, "Reading buffer failed");
}

unsigned long Context::readBufferProfiling(cl_mem buffer, size_t size, void *dst_ptr, size_t offset) {
#if _FLEXCL_DEBUG_SWITCH_ == 1
	cout << "OpenCL::Program::readBuffer(" << size << " Bytes,PROFILING) ... ";
	cout.flush();
#endif
	cl_int ret;
	cl_event perf_event;
	
	ret = clEnqueueReadBuffer(command_queue, buffer, CL_TRUE, offset, size, dst_ptr, 0, NULL, &perf_event);
#if _FLEXCL_DEBUG_SWITCH_ == 1
	if(ret == CL_SUCCESS)
		cout << " OK " << endl;
	else
		cout << " ERR = " << ret << endl;
#endif
	checkReturn(ret, "Reading buffer failed");
	clWaitForEvents(1, &perf_event);
	
	cl_ulong start = 0, end = 0;
	start = _flexcl_profile_info(perf_event, CL_PROFILING_COMMAND_START);
	end = _flexcl_profile_info(perf_event, CL_PROFILING_COMMAND_END);
	
	return (unsigned long)(end-start);
}

void Context::readBufferBlocking(cl_mem buffer, size_t size, void *dst_ptr, size_t offset) {
	this->readBuffer(buffer, size, dst_ptr, true, offset);
}


cl_device_id Context::device_id() {
	return this->_device_id;
}

cl_platform_id Context::platform_id() {
	return this->_platform_id;
}

bool Context::isOutOfOrder(void) {
	return command_queue_outOfOrder;
}

bool Context::isProfiling(void) {
	return command_queue_profiling;
}

PlatformInfo Context::platform_info() {
	return PlatformInfo(this->_platform_id);
}
DeviceInfo Context::device_info() {
	return DeviceInfo(this->_device_id);
}

void Context::removeProgram(Program *program) {
	removeFromVector(programs, program);
}







Program::Program(Context *context, cl_program program) {
	this->context = context;
	this->program = program;
}

Program::~Program() {
	cleanup();
}

void Program::cleanup() {
#if _FLEXCL_DEBUG_SWITCH_ == 1
	cout << "OpenCL::Program::cleanup()" << endl;
#endif
	for(vector<Kernel*>::iterator it = kernels.begin(); it != kernels.end(); ++it) {
		delete *it;
	}
	kernels.clear();
	for(vector<cl_mem>::iterator it = program_buffers.begin(); it != program_buffers.end(); ++it) {
		cl_mem mem = *it;
		context->releaseBuffer(mem);
	}
	program_buffers.clear();
	
	if (program != NULL) clReleaseProgram(program);
	program = NULL;
}

void Program::removeKernel(Kernel *kernel) {
	removeFromVector(kernels, kernel);
}

Kernel* Program::createKernel(string func_name) {
	return this->createKernel(func_name.c_str());
}
Kernel* Program::createKernel(const char* func_name) {
#if _FLEXCL_DEBUG_SWITCH_ == 1
	cout << "OpenCL::Program::createKernel(\"" << func_name << "\")" << endl;
#endif
	cl_int ret;
	
	cl_kernel kernel = clCreateKernel(program, func_name, &ret);
	checkReturn(ret, "Creating kernel failed");
	
	Kernel *kernel_obj = new Kernel(this, kernel);
	kernels.push_back(kernel_obj);
	return kernel_obj;
}

Context* Program::getContext() {
	return this->context;
}


cl_mem Program::createBuffer(size_t size, void* host_ptr) {
	cl_mem mem = this->context->createBuffer(size, host_ptr);
	this->program_buffers.push_back(mem);
	return mem;
}

cl_mem Program::createReadBuffer(size_t size, void* host_ptr) {
	cl_mem mem = this->context->createReadBuffer(size, host_ptr);
	this->program_buffers.push_back(mem);
	return mem;
}

cl_mem Program::createWriteBuffer(size_t size, void* host_ptr) {
	cl_mem mem = this->context->createWriteBuffer(size, host_ptr);
	this->program_buffers.push_back(mem);
	return mem;
}




Kernel::Kernel(Program *program, cl_kernel kernel) {
#if _FLEXCL_DEBUG_SWITCH_ == 1
	cout << "OpenCL::Kernel::created()" << endl;
#endif
	this->program = program;
	this->kernel = kernel;
	this->perf_event = NULL;
}

Kernel::~Kernel() {
#if _FLEXCL_DEBUG_SWITCH_ == 1
	cout << "OpenCL::Kernel::releasing()" << endl;
#endif
	if (kernel != NULL) clReleaseKernel(kernel);
	kernel = NULL;
	program->removeKernel(this);
}


unsigned int Kernel::getArgumentCount() {
	return (unsigned int)(arg_index);
}

void Kernel::addArgument(size_t size, const void* arg_ptr) {
#if _FLEXCL_DEBUG_SWITCH_ == 1
	cout << "OpenCL::Kernel::addArgument(" << (arg_index+1) << ")" << endl;
#endif
	cl_int ret;
	
	ret = clSetKernelArg(kernel, arg_index++, size, arg_ptr);
	checkReturn(ret, "Failed setting kernel argument " + ::to_string(arg_index));
}

void Kernel::setArgument(unsigned int index, size_t size, const void* arg_ptr) {
#if _FLEXCL_DEBUG_SWITCH_ == 1
	cout << "OpenCL::Kernel::setArgument(" << (index) << "," << size << " Bytes)" << endl;
#endif
	cl_int ret;
	
	ret = clSetKernelArg(kernel, index, size, arg_ptr);
	if( (unsigned int)(index) >= arg_index) arg_index = (cl_uint)(index+1);
	checkReturn(ret, "Failed setting kernel argument " + ::to_string(arg_index));
}

void Kernel::addArgument(cl_mem &arg_ptr) { this->addArgument(&arg_ptr); }

void Kernel::addArgument(cl_mem *arg_ptr) {
	this->setArgument(++arg_index, arg_ptr);
}


void Kernel::setArgument(unsigned int index, cl_mem &arg_ptr) {
	this->setArgument(index, sizeof(cl_mem), (const void*)&arg_ptr);
}

void Kernel::setArgument(unsigned int index, cl_mem* arg_ptr) {
	this->setArgument(index, sizeof(cl_mem), (const void*)arg_ptr);
}

void Kernel::setArgumentLocalMem(unsigned int index, size_t size) {
#if _FLEXCL_DEBUG_SWITCH_ == 1
	cout << "OpenCL::Kernel::setArgument(" << (index) << "" << size << " Bytes,LOCAL)" << endl;
#endif
	cl_int ret;
	
	ret = clSetKernelArg(kernel, index, size, NULL);
	if( (unsigned int)(index) >= arg_index) arg_index = (cl_uint)(index+1);
	checkReturn(ret, "Failed setting kernel argument " + ::to_string(arg_index));
}

void Kernel::addArgumentLocalMem(size_t size) {
#if _FLEXCL_DEBUG_SWITCH_ == 1
	cout << "OpenCL::Kernel::setArgument(" << (arg_index+1) << "," << size << " Bytes,LOCAL)" << endl;
#endif
	cl_int ret;
	
	ret = clSetKernelArg(kernel, arg_index++, size, NULL);
	checkReturn(ret, "Failed setting kernel argument " + ::to_string(arg_index));
}

void Kernel::setArgument(unsigned int index, float arg) {
	setArgument(index, sizeof(float), &arg);
}

void Kernel::setArgument(unsigned int index, double arg){
	setArgument(index, sizeof(double), &arg);
}

void Kernel::setArgument(unsigned int index, int arg){
	setArgument(index, sizeof(int), &arg);
}

void Kernel::setArgument(unsigned int index, long arg){
	setArgument(index, sizeof(long), &arg);
}

void Kernel::addArgument(float arg) {
	addArgument(sizeof(float), &arg);
}

void Kernel::addArgument(double arg) {
	addArgument(sizeof(double), &arg);
}

void Kernel::addArgument(int arg) {
	addArgument(sizeof(int), &arg);
}

void Kernel::addArgument(long arg) {
	addArgument(sizeof(long), &arg);
}

void Kernel::setArgument(unsigned int index, unsigned char arg) {
	setArgument(index, sizeof(unsigned char), &arg);
}

void Kernel::setArgument(unsigned int index, unsigned long arg) {
	setArgument(index, sizeof(unsigned long), &arg);
}


inline bool Kernel::isProfiling(void) {
	return this->program->context->isProfiling();
}

inline cl_command_queue Kernel::command_queue() {
	return this->program->context->command_queue;
}

void Kernel::enqueue() {
#if _FLEXCL_DEBUG_SWITCH_ == 1
	cout << "OpenCL::Kernel::enqueue()" << endl;
#endif
	cl_int ret;
	
	profile_infos_collected = false;
	if(isProfiling()) 
		ret = clEnqueueTask(this->command_queue(), this->kernel, 0, NULL, &perf_event);
	else
		ret = clEnqueueTask(this->command_queue(), this->kernel, 0, NULL, NULL);
	checkReturn(ret, "Enqueue kernel failed");
	
}

void Kernel::enqueueNDRange(unsigned int work_dim, const size_t *global_work_size) {
	this->enqueueNDRange(work_dim, global_work_size, NULL);
}

void Kernel::enqueueNDRange(unsigned int work_dim, const size_t *global_work_size, const size_t *local_work_size) {
	const bool profiling = isProfiling();
	
#if _FLEXCL_DEBUG_SWITCH_ == 1
	cout << "OpenCL::Kernel::enqueueNDRange(DIMS=" << work_dim << ",{";
	bool first = true;
	for(unsigned int i=0;i<work_dim;i++) {
		if(first)
			first = false;
		else
			cout << ",";
		cout << global_work_size[i];
	}
	cout << "}";
	if(local_work_size != NULL) {
		cout << ",{";
		first = true;
		for(unsigned int i=0;i<work_dim;i++) {
			if(first)
				first = false;
			else
				cout << ",";
			cout << local_work_size[i];
		}
	}
	cout << "})" << endl;
#endif
	cl_int ret;
	
	profile_infos_collected = false;
	if(profiling) 
		ret = clEnqueueNDRangeKernel(this->command_queue(), kernel, (cl_uint)(work_dim), NULL, global_work_size, local_work_size, 0, NULL, &perf_event);
	else
		ret = clEnqueueNDRangeKernel(this->command_queue(), kernel, (cl_uint)(work_dim), NULL, global_work_size, local_work_size, 0, NULL, NULL);
	checkReturn(ret, "Enqueue kernel failed");
}

void Kernel::enqueueNDRange(size_t dim1) {
	size_t global_work_size[1];
	global_work_size[0] = dim1;
	this->enqueueNDRange(1, global_work_size);
}
void Kernel::enqueueNDRange(size_t dim1, size_t dim2) {
	size_t global_work_size[2];
	global_work_size[0] = dim1;
	global_work_size[1] = dim2;
	this->enqueueNDRange(2, global_work_size);
}

void Kernel::enqueueNDRange(size_t dim1, size_t dim2, size_t dim3) {
	size_t global_work_size[3];
	global_work_size[0] = dim1;
	global_work_size[1] = dim2;
	global_work_size[2] = dim3;
	this->enqueueNDRange(3, global_work_size);
}

void Kernel::collect_profile_infos(void) {
	if(!isProfiling()) return;
	clWaitForEvents(1, &perf_event);
	
	profiling_times[0] = _flexcl_profile_info(perf_event, CL_PROFILING_COMMAND_QUEUED);
	profiling_times[1] = _flexcl_profile_info(perf_event, CL_PROFILING_COMMAND_SUBMIT);
	profiling_times[2] = _flexcl_profile_info(perf_event, CL_PROFILING_COMMAND_START);
	profiling_times[3] = _flexcl_profile_info(perf_event, CL_PROFILING_COMMAND_END);
	
	profile_infos_collected = true;
}

unsigned long Kernel::runtime(void) {
	if(!isProfiling()) return 0L;
	if(!profile_infos_collected) collect_profile_infos();
	return (profiling_times[3] - profiling_times[2]);
}

unsigned long Kernel::total_runtime(void) {
	if(!isProfiling()) return 0L;
	if(!profile_infos_collected) collect_profile_infos();
	return (profiling_times[3] - profiling_times[0]);
}



Program* Kernel::getProgram() {
	return this->program;
}

Context* Kernel::getContext() {
	if(this->program == NULL) return NULL;
	return this->program->getContext();
}

size_t Kernel::getKernelWorkGroupSize(void) {
	cl_int ret;
	size_t result;

	ret = clGetKernelWorkGroupInfo(this->kernel, this->program->context->_device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), & result, NULL);
	checkReturn(ret, "Error querying kernel work group size");
	return result;
}
size_t Kernel::getLocalMemSize(void) {
	cl_int ret;
	size_t result;

	ret = clGetKernelWorkGroupInfo(this->kernel, this->program->context->_device_id, CL_KERNEL_LOCAL_MEM_SIZE, sizeof(size_t), & result, NULL);
	checkReturn(ret, "Error querying kernel work group size");
	return result;
}
size_t Kernel::getPreferredWorkGroupSizeMultiple(void) {
	cl_int ret;
	size_t result;

	ret = clGetKernelWorkGroupInfo(this->kernel, this->program->context->_device_id, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), & result, NULL);
	checkReturn(ret, "Error querying kernel work group size");
	return result;
}


static inline string flexCL_platform_info(cl_platform_id id, cl_platform_info param_name) {
	const int BUF_SIZE = 1024 * 10;
	
	cl_int ret;
	char buffer[BUF_SIZE];
	
	ret = clGetPlatformInfo(id, param_name, BUF_SIZE, buffer, NULL);
	checkReturn(ret, "Error getting platform info");
	return string(buffer);
}

PlatformInfo::PlatformInfo(cl_platform_id platform_id) {
	this->_platform_id = platform_id;
	this->_profile = flexCL_platform_info(platform_id, CL_PLATFORM_PROFILE);
	this->_version = flexCL_platform_info(platform_id, CL_PLATFORM_VERSION);
	this->_name = flexCL_platform_info(platform_id, CL_PLATFORM_NAME);
	this->_vendor = flexCL_platform_info(platform_id, CL_PLATFORM_VENDOR);
	this->_extensions = flexCL_platform_info(platform_id, CL_PLATFORM_EXTENSIONS);
}

PlatformInfo::~PlatformInfo() {}

cl_platform_id PlatformInfo::platform_id() { return this->_platform_id; }
string PlatformInfo::profile() { return this->_profile; }
string PlatformInfo::version() { return this->_version; }
string PlatformInfo::name() { return this->_name; }
string PlatformInfo::vendor() { return this->_vendor; }
string PlatformInfo::extensions() { return this->_extensions; }


vector<DeviceInfo> PlatformInfo::devices() {
	vector<DeviceInfo> result;
	cl_int ret;
	
	cl_device_id* devices = NULL;
	cl_uint num_devices;
	
	ret  = clGetDeviceIDs( this->_platform_id, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
	devices = (cl_device_id*) malloc(sizeof(cl_device_id) * num_devices);
	try {
		ret |= clGetDeviceIDs( this->_platform_id, CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
		checkReturn(ret, "Querying device failed");
		
		for(cl_uint i=0;i<num_devices;i++) {
			result.push_back( DeviceInfo(devices[i]) );
		}
		
	} catch (...) {
		free(devices);
		throw;
	}
	free(devices);
	return result;
}


/* Specific device defines for the internal usage of device info */
#define _FLEXCL_DEVICE_CPU_ 0x1
#define _FLEXCL_DEVICE_GPU_ 0x2
#define _FLEXCL_DEVICE_ACCELERATED_ 0x4

static string flexCL_device_info(cl_device_id id, cl_platform_info param_name) {
	const int BUF_SIZE = 1024 * 10;		// For sure enough memory
	
	cl_int ret;
	char buffer[BUF_SIZE];
	
	ret = clGetDeviceInfo(id, param_name, BUF_SIZE, buffer, NULL);
	checkReturn(ret, "Error getting device info");
	return string(buffer);
}

/*
 * A full list of available options is found at
 * https://www.khronos.org/registry/cl/sdk/1.0/docs/man/xhtml/clGetDeviceInfo.html
 */
DeviceInfo::DeviceInfo(cl_device_id device_id) {
	cl_int ret;
	this->_device_id = device_id;
	
	// CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_ACCELERATOR, or CL_DEVICE_TYPE_DEFAULT.
	cl_device_type device_type;
	ret = clGetDeviceInfo(device_id, CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, NULL);
	checkReturn(ret, "Error getting device info");
	if((device_type & CL_DEVICE_TYPE_CPU) != 0) this->_device_type |= _FLEXCL_DEVICE_CPU_;
	if((device_type & CL_DEVICE_TYPE_GPU) != 0) this->_device_type |= _FLEXCL_DEVICE_GPU_;
	if((device_type & CL_DEVICE_TYPE_ACCELERATOR) != 0) this->_device_type |= _FLEXCL_DEVICE_ACCELERATED_;
}

DeviceInfo::~DeviceInfo() {}

cl_device_id DeviceInfo::device_id() { return this->_device_id; }

string DeviceInfo::getDeviceInfo(cl_device_info param_name) {
	const int BUF_SIZE = 1024 * 10;		// For sure enough memory
	
	cl_int ret;
	char buffer[BUF_SIZE];
	
	ret = clGetDeviceInfo(this->_device_id, param_name, BUF_SIZE, buffer, NULL);
	buffer[BUF_SIZE-1] = '\0';
	checkReturn(ret, "Error getting device info");
	return string(buffer);
}

cl_uint DeviceInfo::getDeviceInfo_ui(cl_device_info param_name) {
	cl_uint result;
	cl_int ret;
	ret = clGetDeviceInfo(this->_device_id, param_name, sizeof(cl_uint), &result, NULL);
	checkReturn(ret, "Error getting device info");
	return result;
}
cl_int DeviceInfo::getDeviceInfo_i(cl_device_info param_name) {
	cl_int result;
	cl_int ret;
	ret = clGetDeviceInfo(this->_device_id, param_name, sizeof(cl_int), &result, NULL);
	checkReturn(ret, "Error getting device info");
	return result;
}
cl_ulong DeviceInfo::getDeviceInfo_ul(cl_device_info param_name) {
	cl_ulong result;
	cl_int ret;
	ret = clGetDeviceInfo(this->_device_id, param_name, sizeof(cl_ulong), &result, NULL);
	checkReturn(ret, "Error getting device info");
	return result;
}
cl_long DeviceInfo::getDeviceInfo_l(cl_device_info param_name) {
	cl_long result;
	cl_int ret;
	ret = clGetDeviceInfo(this->_device_id, param_name, sizeof(cl_long), &result, NULL);
	checkReturn(ret, "Error getting device info");
	return result;
}

size_t DeviceInfo::getDeviceInfo_size_t(cl_device_info param_name) {
	size_t result;
	cl_int ret;
	ret = clGetDeviceInfo(this->_device_id, param_name, sizeof(size_t), &result, NULL);
	checkReturn(ret, "Error getting device info");
	return result;
}
	
unsigned long DeviceInfo::maxMemAllocSize() { return (unsigned long)this->getDeviceInfo_ul(CL_DEVICE_MAX_MEM_ALLOC_SIZE); }
unsigned int DeviceInfo::maxComputeUnits() { return (unsigned int)this->getDeviceInfo_ui(CL_DEVICE_MAX_COMPUTE_UNITS); }
string DeviceInfo::deviceVersion() { return flexCL_device_info(this->_device_id, CL_DEVICE_VERSION); }
string DeviceInfo::driverVersion() { return flexCL_device_info(this->_device_id, CL_DRIVER_VERSION); }
string DeviceInfo::deviceOpenCLVersion() { return flexCL_device_info(this->_device_id, CL_DEVICE_OPENCL_C_VERSION); }
unsigned int DeviceInfo::addressBits() { return (unsigned int)this->getDeviceInfo_ui(CL_DEVICE_ADDRESS_BITS); }
unsigned long DeviceInfo::globalMemSize() { return (unsigned long)this->getDeviceInfo_ul(CL_DEVICE_GLOBAL_MEM_SIZE); }
unsigned long DeviceInfo::globalMemCacheSize() { return (unsigned long)this->getDeviceInfo_ul(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE); }
unsigned long DeviceInfo::localMemSize() { return (unsigned long)this->getDeviceInfo_ul(CL_DEVICE_LOCAL_MEM_SIZE); }
//string DeviceInfo::local_mem_type() { return flexCL_device_info(this->_device_id, CL_DEVICE_LOCAL_MEM_TYPE); }
size_t DeviceInfo::maxWorkGroupSize(void) { return this->getDeviceInfo_size_t(CL_DEVICE_MAX_WORK_GROUP_SIZE); }

bool DeviceInfo::isCPU(void) { return (this->_device_type & _FLEXCL_DEVICE_CPU_) != 0; }
bool DeviceInfo::isGPU(void) { return (this->_device_type & _FLEXCL_DEVICE_GPU_) != 0; }
bool DeviceInfo::isAccelerator(void)  { return (this->_device_type & _FLEXCL_DEVICE_ACCELERATED_) != 0; }

string DeviceInfo::name() { return flexCL_device_info(this->_device_id, CL_DEVICE_NAME); }
string DeviceInfo::vendor() { return flexCL_device_info(this->_device_id, CL_DEVICE_VENDOR); }
string DeviceInfo::extensions() { return flexCL_device_info(this->_device_id, CL_DEVICE_EXTENSIONS); }

unsigned long DeviceInfo::timerResolution() {
	cl_int ret;
	size_t timer_resolution;
	ret = clGetDeviceInfo(this->_device_id, CL_DEVICE_PROFILING_TIMER_RESOLUTION, sizeof(size_t), &timer_resolution, NULL);
	checkReturn(ret, "Error getting device info (Timer resolution)");
	
	if(timer_resolution < 0L) return 0L;
	else return (unsigned long)timer_resolution;
}

bool DeviceInfo::hasImageSupport(void) { return this->getDeviceInfo_i(CL_DEVICE_IMAGE_SUPPORT) > 0; }
unsigned int DeviceInfo::maxClockFrequency(void) { return (unsigned int)this->getDeviceInfo_ui(CL_DEVICE_MAX_CLOCK_FREQUENCY); }
unsigned int DeviceInfo::maxConstantArguments(void) { return (unsigned int)this->getDeviceInfo_ui(CL_DEVICE_MAX_CONSTANT_ARGS); }
size_t DeviceInfo::maxParameterSize(void) { return (size_t)this->getDeviceInfo_l(CL_DEVICE_MAX_PARAMETER_SIZE); }





#if _FLEXCL_OPENGL_SUPPORT_ == 1

cl_mem Context::createSharedOpenGLBuffer(GLuint buffer, bool readAccess, bool writeAccess) {
	cl_int ret;
	cl_mem_flags flags;
	if(readAccess && writeAccess)
		flags = CL_MEM_READ_WRITE;
	else {
		if(readAccess)
			flags = CL_MEM_READ_ONLY;
		else if(writeAccess)
			flags = CL_MEM_WRITE_ONLY;
		else
			flags = CL_MEM_READ_WRITE;		// Fallback on error
	}
	cl_mem result = clCreateFromGLBuffer(context, flags, buffer, &ret);
	checkReturn(ret, "Error creating OpenCL buffer from shared OpenGL buffer");
	return result;
}

OpenGLBuffer::~OpenGLBuffer() {
	this->close();
}

void OpenGLBuffer::close(void) {
	if(_closed) return;
	
	// Release the object
	if(_aquired) this->release();
	if(_created) {
		// Release OpenGL buffer
		glDeleteBuffers(1, &this->_buffer);
	}
	this->_context->releaseBuffer(this->_mem);
	_closed = true;
}


OpenGLBuffer::OpenGLBuffer(Context* context, GLuint buffer) {
	this->_context = context;
	this->_buffer = buffer;
	this->_created = false;
	this->_aquired = false;
	this->_mem = _context->createSharedOpenGLBuffer(buffer);
	_closed = false;
}

OpenGLBuffer::OpenGLBuffer(Context* context, GLuint buffer, cl_mem mem) {
	this->_context = context;
	this->_buffer = buffer;
	this->_mem = mem;
	this->_created = false;
	this->_aquired = false;
	_closed = false;
}

void OpenGLBuffer::aquire(void) {
	cl_int ret;
	ret = clEnqueueAcquireGLObjects(this->_context->command_queue, 1, &this->_mem, 0, NULL, NULL);
	checkReturn(ret, "Error aquiring OpenGL buffer");
	_aquired = true;
}

void OpenGLBuffer::release(void) {
	cl_int ret;
	ret = clEnqueueReleaseGLObjects(this->_context->command_queue, 1, &this->_mem, 0, NULL, NULL);
	checkReturn(ret, "Error releasing OpenGL buffer"); 
	_aquired = false;
}

bool OpenGLBuffer::isAquired(void) { return this->_aquired; }

OpenGLBuffer* Context::createGLBuffer(GLuint buffer) {
	OpenGLBuffer *bufObj = new OpenGLBuffer(this, buffer);
	openglBuffers.push_back(bufObj);
	return bufObj;
}

OpenGLBuffer* Context::createGLBuffer(size_t size, const GLvoid* data, bool isStatic) {
	GLuint buffer;
	GLenum usage;
	if(isStatic)
		usage = GL_STATIC_DRAW;
	else
		usage = GL_DYNAMIC_DRAW;
	
	glGenBuffers(1, &buffer);
	glBindBuffer(GL_ARRAY_BUFFER, buffer);
	glBufferData(GL_ARRAY_BUFFER, size, data, usage);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	OpenGLBuffer *bufObj = this->createGLBuffer(buffer);
	bufObj->_created = true;
	return bufObj;
}


void Context::releaseBuffer(OpenGLBuffer *buffer) {
	buffer->close();
	removeFromVector(this->openglBuffers, buffer);
}

void OpenGLBuffer::readBuffer(size_t size, void* data, bool blocking) {
	if(_closed) throw OpenCLException("Buffer is closed");
	if(!_aquired) this->aquire();
	this->_context->readBuffer(this->_mem, size, data, blocking);
	
}

void OpenGLBuffer::writeBuffer(size_t size, void* data, bool blocking) {
	if(_closed) throw OpenCLException("Buffer is closed");
	if(!_aquired) this->aquire();
	this->_context->writeBuffer(this->_mem, size, data, blocking);
}

#endif




#undef _FLEXCL_DEVICE_CPU_
#undef _FLEXCL_DEVICE_GPU_
#undef _FLEXCL_DEVICE_ACCELERATED_


