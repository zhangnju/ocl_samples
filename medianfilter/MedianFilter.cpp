
#ifndef __linux__
#include "stdafx.h"
#else
#include <string.h>
#endif

#include <iostream>
#include "basic.hpp"
#include "oclobject.hpp"
#include "utils.h"

using namespace std;

#define PAD_LINES 2

void generateInput(cl_uint* p_input, size_t width, size_t height)
{

    srand(12345);

    // random initialization of input
    for (cl_uint i = 0; i <  width * (height+4); ++i)
    {
        p_input[i] = (rand() | (rand()<<15) ) & 0xFFFFFF;
    }
}

// inline functions for reference kernel
#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif

inline unsigned c4max(const unsigned& l, const unsigned& r)
{
    unsigned ur;
    unsigned char* res = (unsigned char*)&ur;
    unsigned char* left = (unsigned char*)&l;
    unsigned char* right = (unsigned char*)&r;

    res[0] = max(left[0],right[0]);
    res[1] = max(left[1],right[1]);
    res[2] = max(left[2],right[2]);
    res[3] = max(left[3],right[3]);

    return ur;
};

inline unsigned c4min(const unsigned& l, const unsigned& r)
{
    unsigned ur;
    unsigned char* res = (unsigned char*)&ur;
    unsigned char* left = (unsigned char*)&l;
    unsigned char* right = (unsigned char*)&r;

    res[0] = min(left[0],right[0]);
    res[1] = min(left[1],right[1]);
    res[2] = min(left[2],right[2]);
    res[3] = min(left[3],right[3]);

    return ur;
};

void ExecuteMedianFilterReference(cl_uint* p_input, cl_uint* p_output, cl_int width, cl_uint height)
{
    memset(p_output, 0,   width * (height+4));

    // do the Median
    for(cl_uint y = 0; y < height; y++)        // rows loop
    {
        int iOffset = (y+PAD_LINES) * width;
        int iPrev = iOffset - width;
        int iNext = iOffset + width;

        for(int x = 0; x < width; x++)        // columns loop
        {
            unsigned uiRGBA[9];

            //get pixels within aperture
            uiRGBA[0] = p_input[iPrev + x - 1];
            uiRGBA[1] = p_input[iPrev + x];
            uiRGBA[2] = p_input[iPrev + x + 1];

            uiRGBA[3] = p_input[iOffset + x - 1];
            uiRGBA[4] = p_input[iOffset + x];
            uiRGBA[5] = p_input[iOffset + x + 1];

            uiRGBA[6] = p_input[iNext + x - 1];
            uiRGBA[7] = p_input[iNext + x];
            uiRGBA[8] = p_input[iNext + x + 1];

            unsigned uiMin = c4min(uiRGBA[0], uiRGBA[1]);
            unsigned uiMax = c4max(uiRGBA[0], uiRGBA[1]);
            uiRGBA[0] = uiMin;
            uiRGBA[1] = uiMax;

            uiMin = c4min(uiRGBA[3], uiRGBA[2]);
            uiMax = c4max(uiRGBA[3], uiRGBA[2]);
            uiRGBA[3] = uiMin;
            uiRGBA[2] = uiMax;

            uiMin = c4min(uiRGBA[2], uiRGBA[0]);
            uiMax = c4max(uiRGBA[2], uiRGBA[0]);
            uiRGBA[2] = uiMin;
            uiRGBA[0] = uiMax;

            uiMin = c4min(uiRGBA[3], uiRGBA[1]);
            uiMax = c4max(uiRGBA[3], uiRGBA[1]);
            uiRGBA[3] = uiMin;
            uiRGBA[1] = uiMax;

            uiMin = c4min(uiRGBA[1], uiRGBA[0]);
            uiMax = c4max(uiRGBA[1], uiRGBA[0]);
            uiRGBA[1] = uiMin;
            uiRGBA[0] = uiMax;

            uiMin = c4min(uiRGBA[3], uiRGBA[2]);
            uiMax = c4max(uiRGBA[3], uiRGBA[2]);
            uiRGBA[3] = uiMin;
            uiRGBA[2] = uiMax;

            uiMin = c4min(uiRGBA[5], uiRGBA[4]);
            uiMax = c4max(uiRGBA[5], uiRGBA[4]);
            uiRGBA[5] = uiMin;
            uiRGBA[4] = uiMax;

            uiMin = c4min(uiRGBA[7], uiRGBA[8]);
            uiMax = c4max(uiRGBA[7], uiRGBA[8]);
            uiRGBA[7] = uiMin;
            uiRGBA[8] = uiMax;

            uiMin = c4min(uiRGBA[6], uiRGBA[8]);
            uiMax = c4max(uiRGBA[6], uiRGBA[8]);
            uiRGBA[6] = uiMin;
            uiRGBA[8] = uiMax;

            uiMin = c4min(uiRGBA[6], uiRGBA[7]);
            uiMax = c4max(uiRGBA[6], uiRGBA[7]);
            uiRGBA[6] = uiMin;
            uiRGBA[7] = uiMax;

            uiMin = c4min(uiRGBA[4], uiRGBA[8]);
            uiMax = c4max(uiRGBA[4], uiRGBA[8]);
            uiRGBA[4] = uiMin;
            uiRGBA[8] = uiMax;

            uiMin = c4min(uiRGBA[4], uiRGBA[6]);
            uiMax = c4max(uiRGBA[4], uiRGBA[6]);
            uiRGBA[4] = uiMin;
            uiRGBA[6] = uiMax;

            uiMin = c4min(uiRGBA[5], uiRGBA[7]);
            uiMax = c4max(uiRGBA[5], uiRGBA[7]);
            uiRGBA[5] = uiMin;
            uiRGBA[7] = uiMax;

            uiMin = c4min(uiRGBA[4], uiRGBA[5]);
            uiMax = c4max(uiRGBA[4], uiRGBA[5]);
            uiRGBA[4] = uiMin;
            uiRGBA[5] = uiMax;

            uiMin = c4min(uiRGBA[6], uiRGBA[7]);
            uiMax = c4max(uiRGBA[6], uiRGBA[7]);
            uiRGBA[6] = uiMin;
            uiRGBA[7] = uiMax;

            uiMin = c4min(uiRGBA[0], uiRGBA[8]);
            uiMax = c4max(uiRGBA[0], uiRGBA[8]);
            uiRGBA[0] = uiMin;
            uiRGBA[8] = uiMax;

            uiRGBA[4] = c4max(uiRGBA[0], uiRGBA[4]);
            uiRGBA[5] = c4max(uiRGBA[1], uiRGBA[5]);

            uiRGBA[6] = c4max(uiRGBA[2], uiRGBA[6]);
            uiRGBA[7] = c4max(uiRGBA[3], uiRGBA[7]);

            uiRGBA[4] = c4min(uiRGBA[4], uiRGBA[6]);
            uiRGBA[5] = c4min(uiRGBA[5], uiRGBA[7]);

            // convert and copy to output
            p_output[(y+PAD_LINES) * width + x] = c4min(uiRGBA[4], uiRGBA[5]);
        }
    }
}

float ExecuteMedianFilterKernel(cl_uint* p_input, cl_uint* p_output, cl_int width, cl_uint height, OpenCLBasic& ocl, OpenCLProgramOneKernel& executable)
{
    double perf_start;
    double perf_stop;

    cl_int err = CL_SUCCESS;
    cl_uint numStages = 0;

    // allocate the buffer with some padding (to avoid boundaries checking)
    cl_mem cl_input_buffer =
        clCreateBuffer
        (
            ocl.context,
            CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
            zeroCopySizeAlignment(sizeof(cl_uint) * width * (height+2*PAD_LINES)),
            p_input,
            &err
        );
    SAMPLE_CHECK_ERRORS(err);
    if (cl_input_buffer == (cl_mem)0)
        throw Error("Failed to create Input Buffer!");

    // allocate the buffer with some padding (to avoid boundaries checking)
    cl_mem cl_output_buffer =
        clCreateBuffer
        (
            ocl.context,
            CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
            zeroCopySizeAlignment(sizeof(cl_uint) * width * (height+2*PAD_LINES)),
            p_output,
            &err
        );
    SAMPLE_CHECK_ERRORS(err);
    if (cl_output_buffer == (cl_mem)0)
        throw Error("Failed to create Output Buffer!");

    err = clSetKernelArg(executable.kernel, 0, sizeof(cl_mem), (void *) &cl_input_buffer);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(executable.kernel, 1, sizeof(cl_mem), (void *) &cl_output_buffer);
    SAMPLE_CHECK_ERRORS(err);

    size_t global_work_size[2] = { (size_t)width, (size_t)height};
    size_t offset[2]= { 0, PAD_LINES};

    // execute kernel
    perf_start=time_stamp();
    err = clEnqueueNDRangeKernel(ocl.queue, executable.kernel, 2, offset, global_work_size, NULL, 0, NULL, NULL);
    SAMPLE_CHECK_ERRORS(err);
    err = clFinish(ocl.queue);
    SAMPLE_CHECK_ERRORS(err);
    perf_stop=time_stamp();

    void* tmp_ptr = NULL;
    tmp_ptr = clEnqueueMapBuffer(ocl.queue, cl_output_buffer, true, CL_MAP_READ, 0, sizeof(cl_uint) * width * (height+2*PAD_LINES), 0, NULL, NULL, NULL);
    if(tmp_ptr!=p_output)
    {
        throw Error("clEnqueueMapBuffer failed to return original pointer");
    }

    err = clFinish(ocl.queue);
    SAMPLE_CHECK_ERRORS(err);

    err = clEnqueueUnmapMemObject(ocl.queue, cl_output_buffer, tmp_ptr, 0, NULL, NULL);
    SAMPLE_CHECK_ERRORS(err);

    err = clReleaseMemObject(cl_input_buffer);
    SAMPLE_CHECK_ERRORS(err);
    err = clReleaseMemObject(cl_output_buffer);
    SAMPLE_CHECK_ERRORS(err);

    // retrieve perf. counter frequency
    return (float)(perf_stop - perf_start);
}

// main execution routine - performs median filtering with 3x3 kernel size
int main (int argc, const char** argv)
{
    //return code
    int ret = EXIT_SUCCESS;
    // pointer to the HOST buffers
    cl_uint* p_input = NULL;
    cl_uint* p_output = NULL;
    cl_uint* p_ref = NULL;

    int width = 4096;
    int height= 4096;
    int max_error_count = 0;
    
    // validate user input parameters
    {
            if(width < 4 || height < 4 || width > 8192 || height > 8192 )
            {
                throw Error("Input size in each dimension should be in the range [4, 8192]!");
            }

            if((width & (width-1)) || (height & (height-1)))
            {
                throw Error("Input size should be (2^N)!");
            }
    }

    // Create the necessary OpenCL objects up to device queue.
    OpenCLBasic oclobjects;

    // Build kernel
    OpenCLProgramOneKernel executable(oclobjects,L"../MedianFilter.cl","","MedianFilterBitonic");

    printf("Input size is %d X %d\n", width, height);

    // allocate buffers with some padding (to avoid boundaries checking)
    cl_uint dev_alignment = zeroCopyPtrAlignment(oclobjects.device);
    size_t aligned_size = zeroCopySizeAlignment(sizeof(cl_uint) * width * (height+2*PAD_LINES), oclobjects.device);
    printf("OpenCL data alignment is %d bytes.\n", dev_alignment);
    p_input = (cl_uint*)aligned_malloc(aligned_size, dev_alignment);
    p_output = (cl_uint*)aligned_malloc(aligned_size, dev_alignment);
    p_ref = (cl_uint*)malloc(sizeof(cl_uint) * width * (height+2*PAD_LINES));

    if(!(p_input && p_output && p_ref))
    {
        throw Error("Could not allocate buffers on the HOST!");
    }

    // random input
    generateInput(p_input, width, height);
    SaveImageAsBMP( p_input, (int)width, (int)height, "MedianFilterInput.bmp");

    // median filtering
    printf("Executing OpenCL kernel...\n");
    float ocl_time = ExecuteMedianFilterKernel(p_input, p_output, width, height, oclobjects, executable);
    SaveImageAsBMP( p_output + PAD_LINES*width, width, height, "MedianFilterOutput.bmp");

        printf("Executing reference...\n");
        ExecuteMedianFilterReference(p_input, p_ref, width, height);
        SaveImageAsBMP( p_ref + PAD_LINES*width, width, height, "MedianFilterOutputReference.bmp");

        printf("Performing verification...\n");
        bool    result = true;
        int     error_count = 0;
        for(int i = 0; i < height*width; i++)
        {
            unsigned int REF = (p_ref+PAD_LINES*width)[i] & 0xFFFFFF;
            unsigned int RES = (p_output+PAD_LINES*width)[i] & 0xFFFFFF;

            if((abs((int)(REF & 0xFF)- (int)(RES & 0xFF))>1)||(abs((int)(REF & 0xFF00)- (int)(RES & 0xFF00))>1)||(abs((int)(REF & 0xFF0000)- (int)(RES & 0xFF0000))>1))
            {
                printf("y = %d, x = %d, REF = %x %x %x, RES = %x %x %x\n",
                    i/width, i%width,
                    (REF & 0xFF),
                    (REF & 0xFF00) >> 8,
                    (REF & 0xFF0000) >> 16,

                    (RES & 0xFF),
                    (RES & 0xFF00) >> 8,
                    (RES & 0xFF0000) >> 16 );
                result = false;
                if(max_error_count>=0 && (++error_count >= max_error_count))
                    break;
            }
        }
        if(!result)
        {
            printf("ERROR: Verification failed.\n");
            ret = EXIT_FAILURE;
        }
        else
        {
            printf("Verification succeeded.\n");
        }

        printf("NDRange perf. counter time %f ms.\n", ocl_time*1000);

    

    free( p_ref );
    aligned_free( p_input );
    aligned_free( p_output );

    return ret;
}

