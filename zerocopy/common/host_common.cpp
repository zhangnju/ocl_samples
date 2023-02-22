

#include "host_common.h"
#include <memory>

//basic components of any cl sample application
cl_context g_clContext;
cl_device_id *g_clDevices = NULL;
cl_command_queue g_clCommandQueue;
cl_program g_clProgram;
char *g_clProgramString;


//data structures specific to this sample solution
float *g_f32_resultImage = NULL; //fimg
unsigned char *g_img = NULL;
unsigned int g_h,g_w;
unsigned int g_numSubSamples = 1; //for speed, go low, accuracy, go high
unsigned int g_imageSize;
bool g_bAlignedAlloc = ALIGNED_ALLOCATION;

cl_mem g_cl_mem_resultImage;
cl_mem g_cl_mem_spheres;
cl_mem g_cl_mem_planes;
cl_kernel cl_kernel_one_pixel;

//scene specific data
sphere g_spheres[3];
plane g_plane;

void* aligned_malloc(size_t size, size_t alignment)
{
	size_t offset = alignment - 1 + sizeof(void*);
	void * originalP = malloc(size + offset);
	size_t originalLocation = reinterpret_cast<size_t>(originalP);
	size_t realLocation = (originalLocation + offset) & ~(alignment - 1);
	void * realP = reinterpret_cast<void*>(realLocation);
	size_t originalPStorage = realLocation - sizeof(void*);
	*reinterpret_cast<void**>(originalPStorage) = originalP;
	return realP;
}

void aligned_free(void* p)
{
	size_t originalPStorage = reinterpret_cast<size_t>(p) - sizeof(void*);
	free(*reinterpret_cast<void**>(originalPStorage));
}

//if an error occurs we exit
//it would be better to cleanup state then exit, for sake of simplicity going to omit the cleanup
void testStatus(int status, const char *errorMsg)
{
	if(status != SUCCESS)
	{
		if(errorMsg == NULL)
		{
			printf("Error\n");
		}
		else
		{
			printf("Error: %s", errorMsg);
		}
		exit(EXIT_FAILURE);
	}
}

int HandleCompileError(void)
{
	cl_int logStatus;
	char *buildLog = NULL;
	size_t buildLogSize = 0;
	//in this tutorial i only have one device
	logStatus = clGetProgramBuildInfo( g_clProgram, g_clDevices[0], CL_PROGRAM_BUILD_LOG, buildLogSize, buildLog, &buildLogSize);
	if(logStatus != CL_SUCCESS)
	{
		printf("logging error\n");
		exit(EXIT_FAILURE);
	}

	buildLog = (char *)malloc(buildLogSize);
	if(buildLog == NULL)
	{
		printf("ERROR TO ALLOCATE MEM FOR BUILDLOG\n");
		exit(EXIT_FAILURE);
	}

	memset(buildLog, 0, buildLogSize);

	logStatus = clGetProgramBuildInfo (g_clProgram, g_clDevices[0], CL_PROGRAM_BUILD_LOG, buildLogSize, buildLog, NULL);
	if(logStatus != CL_SUCCESS)
	{
		free(buildLog);
		return FAIL;
	}

	printf("\nBUILD LOG\n");
	printf("************************************************************************\n");
	printf("%s END \n", buildLog);
	printf("************************************************************************\n");
	free(buildLog);
	return SUCCESS;
}


//convert to string needs to take a string a file as input and write a char * to output
int convertToStringBuf(const char *fileName)
{
	FILE *fp = NULL;
	int status;

	fp = fopen(fileName, "r");
	if(fp == NULL)
	{
		printf("Error opening %s, check path\n", fileName);
		exit(EXIT_FAILURE);
	}

	status = fseek(fp, 0, SEEK_END);
	if(status != 0)
	{
		printf("Error finding end of file\n");
		exit(EXIT_FAILURE);
	}

	int len = ftell(fp);
	if(len == -1L)
	{
		printf("Error reporting position of file pointer\n");
		exit(EXIT_FAILURE);
	}
	rewind(fp);
	g_clProgramString = (char *)malloc((len * sizeof(char))+1);
	if(g_clProgramString == NULL)
	{
		printf("Error in allocation when converting CL source file to a string\n");
		exit(EXIT_FAILURE);
	}
	memset(g_clProgramString, '\0', len+1);
	fread(g_clProgramString, sizeof(char), len, fp);
	status = ferror(fp);
	if(status != 0)
	{
		printf("Error reading into the program string from file\n");
		exit(EXIT_FAILURE);
	}
	fclose(fp);
	
	return SUCCESS;
}

void initializeScene(void)
{
	g_spheres[0].center.x = -2.0f;
	g_spheres[0].center.y = 0.0f;
	g_spheres[0].center.z = -3.5f;
	g_spheres[0].radius = 0.5f;

	g_spheres[1].center.x = -0.5f;
	g_spheres[1].center.y = 0.0f;
	g_spheres[1].center.z = -3.0f;
	g_spheres[1].radius = 0.5f;

	g_spheres[2].center.x = 1.0f;
	g_spheres[2].center.y = 0.0f;
	g_spheres[2].center.z = -2.2f;
	g_spheres[2].radius = 0.5f;

	g_plane.p.x = 0.0f;
	g_plane.p.y = -0.5f;
	g_plane.p.z = 0.0f;

	g_plane.n.x = 0.0f;
	g_plane.n.y = 1.0f;
	g_plane.n.z = 0.0f;

}

int initializeHost(void)
{
	g_clDevices = NULL;
	g_clProgramString = NULL;
	g_f32_resultImage = NULL;
	g_img = NULL;
	
	return SUCCESS;
}
 

int initializeDeviceCode()
{
	cl_int status;

	//load CL file, build CL program object, create CL kernel object
	const char *filename = CL_KERNELS;
	status = convertToStringBuf(filename);

	size_t sourceSize = strlen(g_clProgramString);

	g_clProgram = clCreateProgramWithSource(g_clContext, 1, (const char **)&g_clProgramString, &sourceSize, &status);
	testStatus(status, "clCreateProgramWithSource error");

	status = clBuildProgram(g_clProgram, 1, g_clDevices, NULL, NULL, NULL);
	if(status != CL_SUCCESS)
	{
		if(status == CL_BUILD_PROGRAM_FAILURE)
		{
			HandleCompileError();
		} //end if BUILD_PROGRAM_FAILURE
	} //end if CL_SUCCESS

	cl_kernel_one_pixel = clCreateKernel(g_clProgram, "traceOnePixel", &status);
	testStatus(status, "clCreateKernel error");

	return SUCCESS;

}

int initializeCL()
{
	cl_int status = 0;
	cl_uint numPlatforms= 0;

	//[1] get the platform
	cl_platform_id platformToUse = NULL;
	status = clGetPlatformIDs(0, NULL, &numPlatforms);
	testStatus(status, "clGetPlatformIDs error");
	
	cl_platform_id *platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * numPlatforms);
	if(platforms == NULL)
	{
		printf("Error when allocating space for the platforms\n");
		exit(EXIT_FAILURE);
	}

	status = clGetPlatformIDs(numPlatforms, platforms, NULL);
	testStatus(status, "clGetPlatformIDs error");

	for(unsigned int i=0;i<numPlatforms;i++)
	{
		char pbuf[100];
		status = clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, sizeof(pbuf), pbuf, NULL);
		testStatus(status, "clGetPlatformInfo error");

		//leaving this strcmp for AMD, used when debugging and is the precise AMD platform name
		//if(!strcmp(pbuf, "Advanced Micro Devices, Inc.")) 
		//We found the Intel Platform, this is the one we wantto use
		//if(!strcmp(pbuf, "Intel(R) Corporation"))
		//{
		//	printf("Great! We found an Intel OpenCL platform.\n");
			platformToUse = platforms[i];
		//	break;
		//}
	}
		
	//if(platformToUse == NULL)
	//{
	//	printf("We have not found an Intel(r) OpenCL implementation, exiting application\n");
	//	exit(EXIT_FAILURE);
	//}

	//[2] get device ids for the platform i have obtained
	cl_uint num_devices = -1; //yeah yeah, i know uint 
	cl_device_info devTypeToUse = CL_DEVICE_TYPE_ALL;  //set as CPU or GPU from host_common.h

	//get # of devices of this type on this platform and allocate space in g_clDevices (better be 1 for this tutorial)
	status = clGetDeviceIDs(platformToUse, devTypeToUse, 0, g_clDevices, &num_devices);
	testStatus(status, "clGetDeviceIDs error, might need to set value CL_DEVICE_TYPE_CPU_OR_GPU");
	//allocate space
	g_clDevices = (cl_device_id *)malloc(sizeof(cl_device_id)*num_devices);
	if(g_clDevices == NULL)
	{
		printf("Error when creating space for devices\n");
		exit(EXIT_FAILURE);
	}
	//we know we have an intel platform, get the device we want to use
	status = clGetDeviceIDs(platformToUse, devTypeToUse, num_devices, g_clDevices, 0);
	testStatus(status, "clGetDeviceIDs error");

	//print out the device type just to make sure we got it right
	cl_device_type device_type;
    char vendorName[255];
	memset(vendorName, '\0', 255);

	clGetDeviceInfo(g_clDevices[0], CL_DEVICE_TYPE, sizeof(cl_device_type), (void *)&device_type, NULL);

	clGetDeviceInfo(g_clDevices[0], CL_DEVICE_VENDOR, (sizeof(char)*256), vendorName, NULL);


	if(device_type == CL_DEVICE_TYPE_CPU)
	{
		printf("Device type is CPU, Vendor is %s\n", vendorName);
	}
	else if(device_type == CL_DEVICE_TYPE_GPU)
	{
		printf("Device type is GPU, Vendor is %s\n", vendorName);
	}
	else 
	{
		printf("device type is unknown\n");
	}

	cl_context_properties cps[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platformToUse, 0 };

	//create an OCL context
	g_clContext = clCreateContext(cps, 1, g_clDevices, NULL, NULL, &status);
	testStatus(status, "clCreateContext error");

	//create an openCL commandqueue
	/*
	cl_queue_properties props[]=
	{
      CL_QUEUE_PROPERTIES,CL_QUEUE_PROFILING_ENABLE |
	  CL_QUEUE_ON_DEVICE | CL_QUEUE_ON_DEVICE_DEFAULT,
	  0
	};
	*/
	g_clCommandQueue = clCreateCommandQueueWithProperties(g_clContext, g_clDevices[0], NULL, &status);
	//g_clCommandQueue = clCreateCommandQueue(g_clContext, g_clDevices[0], 0, &status);
	testStatus(status, "clCreateCommandQueue error");
	
	//create device side program, compile and create program objects
	status = initializeDeviceCode();
	testStatus(status, "error in initializeDevice()");

	//create device side buffers
	status = initializeDeviceData();
	testStatus(status, "error in initializeDeviceData()");
	
	if(numPlatforms > 0)
	{
		free(platforms);
	}

	if(g_clDevices != NULL)
	{
		free(g_clDevices);
	}

	return SUCCESS;
}


int cleanupCL()
{
	//cleanup all CL queues, contexts, programs, mem_objs
	cl_int status;

	status = clReleaseKernel(cl_kernel_one_pixel);
	testStatus(status, "Error releasing kernel");

	status = clReleaseProgram(g_clProgram);
	testStatus(status, "Error releasing program");

	status = clReleaseMemObject(g_cl_mem_resultImage);
	testStatus(status, "Error releasing mem object");

	status = clReleaseMemObject(g_cl_mem_spheres);
	testStatus(status, "Error releasing mem object");

	status = clReleaseMemObject(g_cl_mem_planes);
	testStatus(status, "Error releasing mem object");

	status = clReleaseCommandQueue(g_clCommandQueue);
	testStatus(status, "Error releasing mem object");

	status = clReleaseContext(g_clContext);
	testStatus(status, "Error releasing mem object");

	return status;
}



unsigned char clamp(float f)
{
	int i = (int)(f * 255.5f);

	if(i > 255)
		i = 255;
	else if(i < 0)
		i = 0;
	
	return (unsigned char) i;
}


void savePPM()
{
	const char *fname = "AO.ppm";
	FILE *fp = NULL;

	fp=fopen(fname, "wb");

	fprintf(fp, "P6\n");
	fprintf(fp, "%d %d\n", g_w, g_h);
	fprintf(fp, "255\n");
	fwrite(g_img, g_w * g_h * 3, 1, fp);
	fclose(fp);
}

unsigned int verifyZeroCopyPtr(void *ptr, unsigned int sizeOfContentsOfPtr)
{
	int status; //so we only have one exit point from function
	if((long int)ptr % 4096 == 0)
	{
		if(sizeOfContentsOfPtr % 64 == 0)
		{
			status = 1;  
		}
		else status = 0; 
	}
	else status = 0;
	return status;
}
