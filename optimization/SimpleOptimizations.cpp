
#include <iostream>

#include "math.h"

#include "basic.hpp"
#include "oclobject.hpp"

using namespace std;

#pragma warning( push )

typedef struct PerfSetting{
    size_t       global_size;
    size_t       task_size;
    size_t       local_size;
    bool         relaxed_math;
    bool         use_host_ptr;
    bool         ocl_profiling;
    bool         warming;
    bool         vector_kernel;
    int          iterations;
}PerfSetting;

void ExecuteNative(cl_float* p_input, cl_float* p_ref, const PerfSetting& settings)
{
    printf("Executing reference...");
    for (size_t i = 0; i < settings.task_size ; ++i)
    {
        p_ref[i] = sinf(fabs(p_input[i]));
    }
    printf("Done\n\n");
}

void ExecuteKernel(
    cl_float* p_input,
    cl_float* p_output,
    OpenCLBasic& ocl,
    OpenCLProgramOneKernel& executable,
    PerfSetting& settings,
    float* p_time_device,
    float* p_time_host,
    float* p_time_read)
{
    double   perf_ndrange_start;
    double   perf_ndrange_stop;
    double   perf_read_start;
    double   perf_read_stop;
    cl_event        cl_perf_event = NULL;
    cl_int          err;



    // allocate buffers
    const cl_mem_flags  flag = settings.use_host_ptr?CL_MEM_USE_HOST_PTR: CL_MEM_COPY_HOST_PTR;
    size_t              size = sizeof(cl_float)*settings.task_size;
    size_t              alignedSize = zeroCopySizeAlignment(size, ocl.device);

    cl_mem cl_input_buffer = clCreateBuffer(ocl.context, CL_MEM_READ_ONLY|flag, alignedSize, p_input, &err);
    SAMPLE_CHECK_ERRORS(err);
    if (cl_input_buffer == (cl_mem)0)
        throw Error("Failed to create Input Buffer!");
    cl_mem cl_output_buffer = clCreateBuffer(ocl.context, CL_MEM_WRITE_ONLY|flag, alignedSize, p_output, &err);
    SAMPLE_CHECK_ERRORS(err);
    if (cl_output_buffer == (cl_mem)0)
        throw Error("Failed to create Output Buffer!");

    size_t global_size = settings.global_size;
    size_t local_size  = settings.local_size;

    // Set kernel arguments
    err = clSetKernelArg(executable.kernel, 0, sizeof(cl_mem), (void *) &cl_input_buffer);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(executable.kernel, 1, sizeof(cl_mem), (void *) &cl_output_buffer);
    SAMPLE_CHECK_ERRORS(err);

    printf("Global work size %lu\n", global_size);
    if(local_size)
    {
        printf("Local work size %lu\n", local_size);
    }
    else
    {
        printf("Run-time determines optimal local size\n\n");
    }

    {// get maximum workgroup size
        size_t local_size_max;
        err = clGetKernelWorkGroupInfo(executable.kernel, ocl.device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), (void *)&local_size_max, NULL);
        SAMPLE_CHECK_ERRORS(err);
        printf("Maximum workgroup size for this kernel  %lu\n\n",local_size_max );
    }

    if(settings.warming)
    {
        printf("Warming up OpenCL execution...");
        err= clEnqueueNDRangeKernel(ocl.queue, executable.kernel, 1, NULL, &global_size, local_size? &local_size:NULL, 0, NULL, NULL);
        SAMPLE_CHECK_ERRORS(err);
        err = clFinish(ocl.queue);
        SAMPLE_CHECK_ERRORS(err);
        printf("Done\n");
    }


    printf("Executing %s OpenCL kernel...",settings.vector_kernel?"vector":"scalar");
    perf_ndrange_start=time_stamp();
    // execute kernel, pls notice g_bAutoGroupSize
    err= clEnqueueNDRangeKernel(ocl.queue, executable.kernel, 1, NULL, &global_size, local_size? &local_size:NULL, 0, NULL, &cl_perf_event);
    SAMPLE_CHECK_ERRORS(err);
    err = clWaitForEvents(1, &cl_perf_event);
    SAMPLE_CHECK_ERRORS(err);
    perf_ndrange_stop=time_stamp();
    p_time_host[0] = (float)(perf_ndrange_stop - perf_ndrange_start);

    printf("Done\n");

    if(settings.ocl_profiling)
    {
        cl_ulong start = 0;
        cl_ulong end = 0;

        // notice that pure HW execution time is END-START
        err = clGetEventProfilingInfo(cl_perf_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
        SAMPLE_CHECK_ERRORS(err);
        err = clGetEventProfilingInfo(cl_perf_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
        SAMPLE_CHECK_ERRORS(err);
        p_time_device[0] = (float)(end - start)*1e-9f;
    }

    if(settings.use_host_ptr)
    {
        perf_read_start=time_stamp();
        void* tmp_ptr = clEnqueueMapBuffer(ocl.queue, cl_output_buffer, true, CL_MAP_READ, 0, size , 0, NULL, NULL, &err);
        SAMPLE_CHECK_ERRORS(err);
        if(tmp_ptr!=p_output)
        {// the pointer have to be same because CL_MEM_USE_HOST_PTR option was used in clCreateBuffer
            throw Error("clEnqueueMapBuffer failed to return original pointer");
        }
        err=clFinish(ocl.queue);
        SAMPLE_CHECK_ERRORS(err);
        perf_read_stop=time_stamp();

        err = clEnqueueUnmapMemObject(ocl.queue, cl_output_buffer, tmp_ptr, 0, NULL, NULL);
        SAMPLE_CHECK_ERRORS(err);
    }
    else
    {
        perf_read_start=time_stamp();
        err = clEnqueueReadBuffer(ocl.queue, cl_output_buffer, CL_TRUE, 0, size , p_output, 0, NULL, NULL);
        SAMPLE_CHECK_ERRORS(err);
        err=clFinish(ocl.queue);
        SAMPLE_CHECK_ERRORS(err);
        perf_read_stop=time_stamp();
    }
    p_time_read[0] = (float)(perf_read_stop - perf_read_start);

    err = clReleaseMemObject(cl_output_buffer);
    SAMPLE_CHECK_ERRORS(err);
    err = clReleaseMemObject(cl_input_buffer);
    SAMPLE_CHECK_ERRORS(err);
    err = clReleaseEvent(cl_perf_event);
    SAMPLE_CHECK_ERRORS(err);
}

// main execution routine - perform simple math on float vectors
int main (int argc, const char** argv)
{
    // pointer to the HOST buffers
    cl_float*   p_input = NULL;
    cl_float*   p_output = NULL;
    cl_float*   p_ref = NULL;
    //return code
    int ret = EXIT_SUCCESS;
    int max_error_count = 0;

    PerfSetting settings;
    settings.global_size  = 16*1024*1024;
    settings.task_size    = 16*1024*1024;
    settings.local_size   = 0;//OpenCL work group size. Set to 0 for work group size auto selection
    settings.relaxed_math = false;
    settings.use_host_ptr = false;
    settings.ocl_profiling= false;
    settings.warming      = false;
    settings.vector_kernel= true;
    settings.iterations   = 1000;

    if(settings.vector_kernel == true)
      settings.global_size = settings.global_size/4;

    // Create the necessary OpenCL objects up to device queue.
    OpenCLBasic oclobjects;


    // Build kernel
    string build_options;
    build_options += "-D ITER_NUM="+to_str(settings.iterations);
    if(settings.relaxed_math)
            build_options += " -cl-fast-relaxed-math";

    OpenCLProgramOneKernel executable(
        oclobjects,
        L"../SimpleOptimizations.cl",
        "",
        settings.vector_kernel?"SimpleKernel4":"SimpleKernel",
        build_options);

    // allocate buffers
    cl_uint     dev_alignment = zeroCopyPtrAlignment(oclobjects.device);
    size_t      size = sizeof(cl_float) * settings.task_size;
    size_t      alignedSize = zeroCopySizeAlignment(size, oclobjects.device);
    p_input = (cl_float*)aligned_malloc(alignedSize, dev_alignment);
    p_output = (cl_float*)aligned_malloc(alignedSize, dev_alignment);
    p_ref = (cl_float*)aligned_malloc(alignedSize, dev_alignment);

    if(!(p_input && p_output && p_ref))
    {
        printf("Could not allocate buffers on the HOST!");
        return -1;
    }

    // set input array to random legal values
    srand(2011);
    for (size_t i = 0; i < settings.task_size ; i++)
    {
        p_input[i] = rand_uniform_01<cl_float>()*512.0f - 256.0f;
    }

    // do simple math
    float ocl_time_device = 0;
    float ocl_time_host = 0;
    float ocl_time_read = 0;

    ExecuteKernel(
            p_input,p_output,
            oclobjects,executable,
            settings,
            &ocl_time_device,
            &ocl_time_host,
            &ocl_time_read);

    ExecuteNative(p_input,p_ref,settings);


    printf("NDRange perf. counter time %f ms.\n", 1000.0f*ocl_time_host);

    if(settings.ocl_profiling)
    {
            printf("NDRange event profiling time %f ms.\n", 1000.0f*ocl_time_device);
    }

    printf("%s buffer perf. counter time %f ms.\n\n", settings.use_host_ptr?"Map":"Read", 1000.0f*ocl_time_read);


    // Do verification
    printf("Performing verification...\n");
    int     error_count = 0;
    for(size_t i = 0; i < settings.task_size ; i++)
    {
            // Compare the data
            if( fabsf(p_output[i] - p_ref[i]) > 0.01f )
            {
                printf("Error at location %d,  outputArray = %f, refArray = %f \n", i, p_output[i], p_ref[i]);
                error_count++;
                ret = EXIT_FAILURE;
                if(max_error_count>0 && error_count >= max_error_count)
                {
                    break;
                }
            }
        }
        printf("%s", (error_count>0)?"ERROR: Verification failed.\n":"Verification succeeded.\n");


    aligned_free( p_ref );
    aligned_free( p_input );
    aligned_free( p_output );

    return ret;
}

#pragma warning( pop )
