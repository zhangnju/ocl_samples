
#include <iostream>
#include <ctime>
#include <limits>
#include <cmath>

#include <CL/cl.h>

#include "basic.hpp"
#include "oclobject.hpp"

using namespace std;

typedef struct gemm_settings{
    size_t size;
    int iterations;

    string arithmetic;
    bool arithmetic_float;
    bool arithmetic_double;

    string kernel;
    bool kernel_nt;
    bool kernel_nn;

    bool validation;

    size_t tile_size_M;
    size_t tile_group_M;

    size_t tile_size_N;
    size_t tile_group_N;

    size_t tile_size_K;
}gemm_settings;

// Check validity for multiplication of square matrices: Cresult == alpha*A*Btransposed(B) + beta*Cinit.
// Samll simplification here: the procedure assumes that initial C values are all zeros,
// so beta is not actually used.
template <class T>
bool checkValidity (
    const T* A,     // left input matrix, column-major
    const T* B,     // right input matrix, column-major or row-major depending on Btransposed argument
    const T* C,     // output matrix, column-major
    size_t size,    // number of column in each of the matrices
    size_t ldabc,   // row stride: the number of T elements in one row (including optional padding)
    bool Btransposed,
    T alpha, T beta // coefficient to compute
)
{
    cout << "Validate output..." << flush;

    // Btransposed == false, lstride = 1
    size_t lstride = Btransposed ? ldabc : 1;
    size_t jstride = Btransposed ? 1 : ldabc;

    // Estimate error tolerance for a given type T and relying on the fact
    // that initial matrix values are from [0, 1]
    T max_value = 1;
    T error_tol = T(2) * alpha * max_value * max_value * T(2) * size * numeric_limits<T>::epsilon();

    for(size_t i = 0; i < size; ++i)
    {
        for(size_t j = 0; j < size; ++j)
        {
            // compute golden value for c[i][j] element
            T accum = 0;
            for(size_t l = 0; l < size; ++l)
            {
                accum += A[l*ldabc + i] * B[l*lstride + j*jstride];
            }

            T golden = alpha*accum;

            T absdiff = abs(C[j*ldabc+i] - golden);
            if(absdiff > error_tol)
            {
                cout << " FAILED\n";
                cerr.precision(std::numeric_limits<T>::digits10);
                cerr << "\nVALIDATION FAILED!!!\n    reference" << "[" << i << ", " << j << "] = "
                     << golden << ",\n    calculated" << "[" << i << ", " << j << "] = "
                     << C[j*ldabc+i]
                     << ",\n    absolute difference" << "[" << i << ", " << j << "] = " << absdiff << "\n"
                     << "Further validation was stopped\n\n";
                return false;
            }
        }
    }

    std::cout << " PASSED\n";
    return true;
}


// The main GEMM function with all application specific
// OpenCL host side code.
template <typename T>
void gemm (
    gemm_settings& settings,
    OpenCLBasic& oclobjects,
    OpenCLProgramOneKernel& executable
)
{
    // -----------------------------------------------------------------------
    // Calculating, allocating and initializing host-side memory
    // -----------------------------------------------------------------------

    // Query for necessary alignment for each row
    // Each row is aligned by requirements of OpenCL to achieve better
    // performance in comparison to not aligned data
    size_t rowAlignment = requiredOpenCLAlignment(oclobjects.device);

    // a couple of sanity checks to ensure correctness of the further math with the returned value
    assert(rowAlignment >= sizeof(T)); // must be
    assert((rowAlignment & (rowAlignment - 1)) == 0); // test for power of 2

    // the next call checks for various OpenCL bounds to proactively
    // handle possible errors like out of memory
    //cmdparser.validateParameters(oclobjects, executable, sizeof(T), rowAlignment);

    size_t size = settings.size;

    cout
        << "Running gemm_" << settings.kernel
        << " kernel with matrix size: " << size << "x" << size << "\n";

    // Ensures that each matrix memory row is aligned
    size_t stride = (size*sizeof(T) + rowAlignment - 1) & ~(rowAlignment - 1);
    cout << "Memory row stride to ensure necessary alignment: " << stride << " bytes\n";
    // calculate row stride in elements of T
    stride /= sizeof(T);
    assert(size <= stride);

    if(stride/sizeof(T) > size_t(numeric_limits<cl_int>::max()))
    {
        cout<<
            "Memory row stride in elements " << to_str(stride/sizeof(T)) <<
            " cannot be represented as type int, which can be maximum " <<
            to_str(numeric_limits<cl_int>::max()) + ".";
        return;
    }

    size_t matrix_memory_size = size*stride*sizeof(T);
    cout << "Size of memory region for one matrix: " << matrix_memory_size << " bytes\n";

    // Allocate aligned memory for matrices to use them in
    // buffers with CL_MEM_USE_HOST_PTR.
    // OpenCLDeviceAndHostMemory is used just for
    // convenient resource deallocation:
    // a pair of pointer and cl_mem object; cl_mem object is
    // be creater later.

    size_t alignmentForPtr = zeroCopyPtrAlignment(oclobjects.device);
    size_t alignedSize = zeroCopySizeAlignment(matrix_memory_size, oclobjects.device);

    OpenCLDeviceAndHostMemory<T> matrix_A;
    matrix_A.host = (T*)aligned_malloc(alignedSize, alignmentForPtr);

    OpenCLDeviceAndHostMemory<T> matrix_B;
    matrix_B.host = (T*)aligned_malloc(alignedSize, alignmentForPtr);

    OpenCLDeviceAndHostMemory<T> matrix_C;
    matrix_C.host = (T*)aligned_malloc(alignedSize, alignmentForPtr);

    // Initialize matrices row by row.
    for(size_t i = 0; i < size; ++i)
    {
        T* row_A = matrix_A.host + i*stride;
        T* row_B = matrix_B.host + i*stride;
        T* row_C = matrix_C.host + i*stride;

        // Fill the rows with random values from range [0, 1]
        fill_rand_uniform_01(row_A, size);
        fill_rand_uniform_01(row_B, size);

        // To simplify validation a bit, we initialize C matrix with all zeros.
        // It should not affect performance, which should be identical to
        // the general case.
        std::fill(row_C, row_C + size, T(0));
    }

    // -----------------------------------------------------------------------
    // Allocating device-side resources for matrices
    // -----------------------------------------------------------------------

    cl_int err = 0; // OpenCL error code

    // Create OpenCL buffers for the matrices based on allocated memory regions
    // Create buffers with CL_MEM_USE_HOST_PTR to minimize copying and
    // model situation when matrices are hosted by some native library that
    // uses OpenCL to accelerate calculations.

    matrix_A.device = clCreateBuffer(
        oclobjects.context,
        CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
        matrix_memory_size,
        matrix_A.host,
        &err
    );
    SAMPLE_CHECK_ERRORS(err);

    matrix_B.device = clCreateBuffer(
        oclobjects.context,
        CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
        matrix_memory_size,
        matrix_B.host,
        &err
    );
    SAMPLE_CHECK_ERRORS(err);

    matrix_C.device = clCreateBuffer(
        oclobjects.context,
        CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
        matrix_memory_size,
        matrix_C.host,
        &err
    );
    SAMPLE_CHECK_ERRORS(err);

    T alpha = rand_uniform_01<T>();
    T beta = rand_uniform_01<T>();
    cout << "Using alpha = " << alpha << " and beta = " << beta << "\n";
    cl_int cl_size = static_cast<int>(size);  // kernel requires int value
    cl_int ldabc = static_cast<int>(stride);  // kernel requires int value

    // -----------------------------------------------------------------------
    // Setting kernel arguments
    // -----------------------------------------------------------------------

    err = clSetKernelArg(executable.kernel, 0, sizeof(cl_mem), &matrix_A.device);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(executable.kernel, 1, sizeof(cl_int), &ldabc);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(executable.kernel, 2, sizeof(cl_mem), &matrix_B.device);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(executable.kernel, 3, sizeof(cl_int), &ldabc);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(executable.kernel, 4, sizeof(cl_mem), &matrix_C.device);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(executable.kernel, 5, sizeof(cl_int), &ldabc);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(executable.kernel, 6, sizeof(cl_int), &cl_size);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(executable.kernel, 7, sizeof(T), &alpha);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(executable.kernel, 8, sizeof(T), &beta);
    SAMPLE_CHECK_ERRORS(err);

    // -----------------------------------------------------------------------
    // Define ndrange iteration space: global and local sizes based on
    // parameters obtained from user.

    // Refer to the sample documentation for clarification about
    // how work is devided among work-groups and work-items.
    // -----------------------------------------------------------------------

    size_t global_size[2] = {
        size / settings.tile_size_M,
        size / settings.tile_size_N
    };

    size_t local_size[2] = {
        settings.tile_group_M,
        settings.tile_group_N
    };

    // theoretical number of floating point operations (addition and multiplication) for one kernel execution
    // needed for performance calculations (GFLOPS) at every iteration below
    double flops = double(size)*size*(
        size + // multiplications
        size + // additions
        2      // multiplication by alpha and beta
    );

    // -----------------------------------------------------------------------
    // Loop with the kernel invocation
    // -----------------------------------------------------------------------

    for(int i = 0; i < settings.iterations; ++i)
    {
        // Here we start measuring host time for kernel execution
        double start = time_stamp();

        err = clEnqueueNDRangeKernel(
            oclobjects.queue,
            executable.kernel,
            2,
            0,
            global_size,
            local_size,
            0, 0, 0
        );
        SAMPLE_CHECK_ERRORS(err);

        err = clFinish(oclobjects.queue);
        SAMPLE_CHECK_ERRORS(err);

        // It is important to measure end host time after clFinish call
        double end = time_stamp();

        double time = end - start;
        cout << "Host time: " << time << " sec.\n";
        cout << "Host perf: " << flops/time/1e9 << " GFLOPS\n";
        cout.flush();

        if(i == 0 && settings.validation)
        {
            // Validate result for the first iteration only and
            // only if user wants this.
            // Please note, validation procedure cannot be run at
            // futher iterations after the very first iteration,
            // as the results are being accumulated in C matrix
            // every iteration but validation procedures assumes that
            // C initial values are all zeros.

            clEnqueueMapBuffer(
                oclobjects.queue,
                matrix_C.device,
                CL_TRUE,    // blocking map
                CL_MAP_READ,
                0,
                matrix_memory_size,
                0, 0, 0,
                &err
            );
            SAMPLE_CHECK_ERRORS(err);

            // After map call, host-memory area for matrix C is
            // automatically updated with the latest bits from the device
            // So we just use it by original pointer as well as input matrices:
            if(
                !checkValidity(
                    matrix_A.host,
                    matrix_B.host,
                    matrix_C.host,
                    size,
                    stride,
                    settings.kernel_nt,    // whether B is transposed or not
                    alpha,
                    beta
                )
            )
            {
                throw Error("Validation procedure reported failures");
            }

            cout.flush();

            err = clEnqueueUnmapMemObject(
                oclobjects.queue,
                matrix_C.device,
                matrix_C.host,
                0, 0, 0
            );
            SAMPLE_CHECK_ERRORS(err);

            // Finish here is only required for correct time measurment on the next iteration
            // It does not affect correctness of calculations because you use the in-order OpenCL queue here.
            err = clFinish(oclobjects.queue);
            SAMPLE_CHECK_ERRORS(err);
        }
    }

    // All resources are deallocated automatically.
}


// Entry point for sample application, command-line parsing,
// generic OpenCL resources allocation and deallocation.
int main (int argc, const char** argv)
{
    gemm_settings settings;

    settings.size = 3968;
    settings.iterations = 10;

    settings.arithmetic = "float";
    settings.arithmetic_float = true;
    settings.arithmetic_double= false;

    settings.kernel = "nn";
    settings.kernel_nt= false;
    settings.kernel_nn= true;

    settings.validation = false;
    settings.tile_size_M  = 1;
    settings.tile_group_M = 16;
    settings.tile_size_N  = 128;
    settings.tile_group_N = 1;
    settings.tile_size_K  = 8;

    // Create the necessary OpenCL objects up to device queue.
    OpenCLBasic oclobjects;

    // Form build options string from given parameters: macros definitions to pass into kernels
    string build_options =
            "-DT=" + settings.arithmetic +
            (settings.arithmetic_double ? " -DSAMPLE_NEEDS_DOUBLE" : "") +
            " -DTILE_SIZE_M=" + to_str(settings.tile_size_M) +
            " -DTILE_GROUP_M=" + to_str(settings.tile_group_M) +
            " -DTILE_SIZE_N=" + to_str(settings.tile_size_N) +
            " -DTILE_GROUP_N=" + to_str(settings.tile_group_N) +
            " -DTILE_SIZE_K=" + to_str(settings.tile_size_K);

    cout << "Build program options: " << inquotes(build_options) << "\n";

    // Build kernel
    OpenCLProgramOneKernel executable(
        oclobjects,
        L"../gemm.cl",
        "",
        "gemm_" + settings.kernel,
        build_options
    );

    // Call gemm with required type of elements
    if(settings.arithmetic_float)
    {
        gemm<float>(settings, oclobjects, executable);
    }
    else if(settings.arithmetic_double)
    {
        gemm<double>(settings, oclobjects, executable);
    }

    // All resource deallocations happen in destructors of helper objects.

    return 0;
    
}
