

#include <CL/cl.h>

#include "basic.hpp"
#include "multidevice.hpp"

using namespace std;


cl_program create_program (cl_context context)
{
    // Create a synthetic kernel.
    const char* source =
        "   kernel void simple (                "
        "       global const float* a,          "
        "       global const float* b,          "
        "       global float* c                 "
        "   )                                   "
        "   {                                   "
        "       int i = get_global_id(0);       "
        "       float tmp = 0;                  "
        "       for(int j = 0; j < 100000; ++j) "
        "           tmp += a[i] + b[i];         "
        "       c[i] = tmp;                     "
        "   }                                   "
    ;

    cl_int err = 0;
    cl_program program = clCreateProgramWithSource(context, 1, &source, 0, &err);
    SAMPLE_CHECK_ERRORS(err);
    return program;
}
