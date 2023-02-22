
#ifndef HOST_COMMON_H
#define HOST_COMMON_H

#include <CL/cl.h>
#include <string.h>
#include <cstdlib>
#include <iostream>
#include <string>
#include <fstream>

#include "scene.h"

#include <chrono>
using namespace std;

#define SUCCESS 0
#define FAIL -1
#define IMAGE_WIDTH 1024
#define IMAGE_HEIGHT 1024
#define CL_KERNELS "../../common/kernels.cl"
#define PERF_TEST_ZERO_COPY 0 //don't forget to also decide if you want aligned or unaligned malloc
#define ALIGNED_ALLOCATION 1

//#define CL_DEVICE_CPU_OR_GPU CL_DEVICE_TYPE_GPU
//#define CL_DEVICE_CPU_OR_GPU CL_DEVICE_TYPE_CPU

void* aligned_malloc(size_t size, size_t alignment);
void aligned_free(void* p);

//basic components of any cl sample application
extern cl_context g_clContext;
extern cl_device_id *g_clDevices;
extern cl_command_queue g_clCommandQueue;
extern cl_program g_clProgram;
extern char *g_clProgramString;


//data structures specific to this sample solution
extern float *g_f32_resultImage; 
extern unsigned char *g_img;
extern unsigned int g_h,g_w;
extern unsigned int g_numSubSamples;
extern unsigned int g_imageSize;
extern bool g_bAlignedAlloc;
extern cl_mem g_cl_mem_resultImage;
extern cl_mem g_cl_mem_spheres;
extern cl_mem g_cl_mem_planes;
extern cl_kernel cl_kernel_one_pixel;

//scene graph
extern sphere g_spheres[3];
extern plane g_plane;

//functions for any CL Sample application
int initializeHost(void);
int initializeCL(void);
int initializeCL2(void);
int runCLKernels(void);
int cleanupCL(void);

//lower level functions used to setup the application
//create buffers to be used on the device
int initializeDeviceData();
//compile program and create kernels to be used on device side
int initializeDeviceCode();
int initializeHost(void);
void initializeScene(void);

//Debugging functions
void print1DArray(const std::string arrayName, const unsigned int *arrayData, const unsigned int length);
int HandleCompileError(void);
int convertToString(const char *filename, char *buf);

int ReportPlatformInfo(FILE *fp);
int ReportDeviceInfo(FILE *fp);

//utility functions
//testStatus is intentionally extremely tiny to minimize extraneous code in a sample
void testStatus(int status, const char *errorMsg);
int HandleCompileError(void);
std::string convertToString(const char *fileName);
unsigned char clamp(float f);
void savePPM();
unsigned int verifyZeroCopyPtr(void *ptr, unsigned int sizeOfContentsOfPtr);


#endif 
