#define DEFAULT_MATRIX_SIZE 512 
#define TEST_ITERATIONS 2000


//Uncomment to check results (can take a long time for large matrixes)
//#define CHECK_RESULT_CORRECTNESS

//Uncomment to use out of order queue property (can help improve efficiency for smaller matrix sizes)
//#define USE_OUT_OF_ORDER_QUEUE


#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cstring>
#include <ctype.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <string.h>
#include <vector>
#include <CL/cl.h>
#include <immintrin.h>

// Linux-specific definitions
#if defined(__linux__)
#include <cstdint>
typedef int64_t __int64;
#define  _aligned_malloc(bufsz, alignsz) _mm_malloc(bufsz, alignsz);
#define  _aligned_free(ref) _mm_free(ref);


typedef timespec simpleTime;

void simpleGetTime(simpleTime* timestamp)
{
    clock_gettime(CLOCK_REALTIME, timestamp);
}

double simpleTimeDiffMsec(simpleTime tfinish, simpleTime tstart)
{
    double result;
    long long elapsed_nsec = tfinish.tv_nsec - tstart.tv_nsec;
    long long elapsed_sec = tfinish.tv_sec - tstart.tv_sec;

    //timespec uses two fields -- check if borrowing necessary
    if (elapsed_nsec < 0) {
        elapsed_sec -= 1;
        elapsed_nsec += 1000000000;
    }
    //return total converted to milliseconds
    result = (double)elapsed_sec *1000.0;
    result += (double)elapsed_nsec / 1000000;

    return result;
}
#if 0
float half_to_float(unsigned short in)
{
	float cnv;
	cnv = _cvtsh_ss(in);
	return cnv;
}

unsigned short float_to_half(float in)
{
	unsigned short cnv = 0;
	cnv = _cvtss_sh(in, 0);
	return cnv;
}
#else
unsigned short float_to_half(float f)
{
  int hs, he, hm, *ptr;
  short rlt;

  ptr = (int *)&f;
  hs = ((*ptr)&0x80000000) >> 16;

  he = ((*ptr)&0x7f800000) >> 23;
  he = he - 0x70;
  he = he << 10;

  hm = ((*ptr)&0x007fffff) >> 13;

  rlt = hs | he | hm;

  return *((cl_half *)&rlt);
}

float half_to_float(unsigned short h)
{
  short *ptr;
  int fs, fe, fm, rlt;

  ptr = (short *)&h;

  fs = ((*ptr)&0x8000) << 16;

  fe = ((*ptr)&0x7c00) >> 10;
  fe = fe + 0x70;
  fe = fe << 23;

  fm = ((*ptr)&0x03ff) << 13;

  rlt = fs | fe | fm;
  return *((float *)&rlt);
}
#endif 

#endif //__linux__


// Windows-specific definitions
#if defined(_WIN32)
#include <windows.h>
typedef LARGE_INTEGER simpleTime;

void simpleGetTime(simpleTime* timestamp)
{
    QueryPerformanceCounter(timestamp);
}

double simpleTimeDiffMsec(simpleTime tfinish, simpleTime tstart)
{
    static LARGE_INTEGER tFreq = { 0 };

    if (!tFreq.QuadPart) QueryPerformanceFrequency(&tFreq);

    double freq = (double)tFreq.QuadPart;
    return 1000.0 * ((double)tfinish.QuadPart - (double)tstart.QuadPart) / freq;
}

float half_to_float(unsigned short in)
{
	int tmp[4] = { in,0,0,0 };
	__m128 out = _mm_cvtph_ps(*((__m128i*)tmp));

	return ((float*)&out)[0];
}

unsigned short float_to_half(float in)
{

	float tmp[4] = { in,0,0,0 };
	__m128i out = _mm_cvtps_ph(*((__m128*)tmp), 0);

	return ((unsigned short*)&out)[0];
}
#endif   // _WIN32


struct device_info {
   cl_ulong maxSLM;
   size_t max_wrk_grp_size;
};

#define CHK_ERR(x) assert(x == CL_SUCCESS)

#define maxval(a,b)            (((a) > (b)) ? (a) : (b))
#define minval(a,b)            (((a) < (b)) ? (a) : (b))


cl_device_id getDeviceID(int devtype, device_info * dev_info)
{
   int rval;
   cl_platform_id platform[2];
   cl_device_id device;
   cl_ulong maxSLM;
   size_t max_wrk_grp_size;

   rval = clGetPlatformIDs(2, platform, NULL);
   if (rval != CL_SUCCESS)
   {
      printf("failed clGetPlatformIDs (%d)\n", rval);
      exit(-1);
   }
   rval = clGetDeviceIDs(platform[0], devtype, 1, &device, NULL);
   if (rval != CL_SUCCESS)
      rval = clGetDeviceIDs(platform[1], devtype, 1, &device, NULL);
   if (rval != CL_SUCCESS)
   {
      printf("# Valid device ids:\n");
      printf("# %d - DEFAULT \n%d - CPU \n%d - GPU \n%d - ACCELERATOR\n",
             CL_DEVICE_TYPE_DEFAULT, CL_DEVICE_TYPE_CPU,
             CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_ACCELERATOR);
      printf("failed GetDeviceIDs (%d)\n", rval);
      exit(-1);
   }
   char deviceName[1024];
   size_t size_ret;
   clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(deviceName), deviceName,
                   &size_ret);
   printf("# device name: %s\n", deviceName);

   clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(maxSLM),
                   &maxSLM, &size_ret);
   printf("# device slm size: %i\n", (int) (maxSLM));

   clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                   sizeof(max_wrk_grp_size), &max_wrk_grp_size, &size_ret);
   printf("# device max work group size: %i\n", (int) (max_wrk_grp_size));

   if (dev_info != NULL)
   {
      dev_info->maxSLM = maxSLM;
      dev_info->max_wrk_grp_size = max_wrk_grp_size;
   }

   return device;
}


cl_program getProgram(cl_device_id dev, cl_context ctx, const char *filename,
                      const char *buildOptions)
{
   FILE *fp;
   char *my_program;
   size_t fsize;
   int e;

   fp = fopen(filename, "rb");
   if (!fp)
   {
	   printf("Could not find OpenCL kernel file %s\n", filename);
	   exit(1);
   }
   

   fseek(fp, 0L, SEEK_END);
   fsize = ftell(fp);
   fseek(fp, 0L, SEEK_SET);     // Note this resets fp to beginning.


   my_program = (char *) malloc(fsize);
   assert(my_program);
   e = fread(my_program, 1, fsize, fp);
   assert(e == fsize);

   cl_program program =
       clCreateProgramWithSource(ctx, 1, (const char **) &my_program,
                                 &fsize, &e);
   CHK_ERR(e);
   
   e = clBuildProgram(program, 1, &dev, buildOptions, NULL, NULL);

   if (e != CL_SUCCESS)
   {
      /* Print out build log */
      char build_log[65536];
      size_t logsize = sizeof(build_log) - 1;
      e = clGetProgramBuildInfo(program, dev,
                                CL_PROGRAM_BUILD_LOG, logsize,
                                build_log, &logsize);
      build_log[logsize] = 0;

      printf("# Build Failed:\n%s\n", build_log);
      exit(-1);
   }

   return program;
}



cl_kernel getKernel(cl_program program, const char *kernelname)
{
   cl_int e;
   cl_kernel kernel = clCreateKernel(program, kernelname, &e);
   //CHK_ERR(e);
   return kernel;
}





static int dimM = DEFAULT_MATRIX_SIZE;
static int dimK = DEFAULT_MATRIX_SIZE;
static int dimN = DEFAULT_MATRIX_SIZE;

static int h = dimM;            // M
static int w = dimK;            // K
static int w1 = dimN;           // N
static size_t sz_src0 = h * w * sizeof(float);
static size_t sz_src1 = w * w1 * sizeof(float);
static size_t sz_dst = h * w1 * sizeof(float);



std::string buildoptions = " -cl-mad-enable -cl-fast-relaxed-math ";



static cl_device_id device = NULL;
static cl_context context = NULL;
static cl_program program = NULL;
static cl_command_queue queue = NULL;

// fp32
static bool have_fp32_ref = false;
static float *src0 = NULL;
static float *src1 = NULL;
static float *gpu_dst = NULL;
static float *cpu_dst = NULL;
static cl_mem cl_src0 = NULL;
static cl_mem cl_src1 = NULL;
static cl_mem im_src0 = NULL;
static cl_mem im_src1 = NULL;
static cl_mem mi_src0 = NULL;
static cl_mem mi_src1 = NULL;
static cl_mem cl_dst = NULL;
static cl_mem mi_dst = NULL;

// fp16
static bool have_fp16_ref = false;
static unsigned short *hsrc0 = NULL;
static unsigned short *hsrc1 = NULL;
static unsigned short *hgpu_dst = NULL;
static unsigned short *hcpu_dst0 = NULL;
static float *hcpu_dst1 = NULL;
static cl_mem cl_hsrc0 = NULL;
static cl_mem cl_hsrc1 = NULL;
static cl_mem im_hsrc0 = NULL;
static cl_mem im_hsrc1 = NULL;
static cl_mem mi_hsrc0 = NULL;
static cl_mem mi_hsrc1 = NULL;
static cl_mem cl_hdst = NULL;
static cl_mem mi_hdst = NULL;

static float max_gpu_clock_frequency_in_mhz = 0.0f;
static cl_uint max_compute_units_gpu = 0;
static double theoretical_peak_float_perf = 0.0;

enum TEST_TYPE {
   TEST_TYPE_INVALID = 0,
   TEST_TYPE_ALL = 100,
   TEST_TYPE_UNOPTIMIZED,
   TEST_TYPE_SIMD_4x8x8,
   TEST_TYPE_SIMD_IMAGESRW_2x32,
   TEST_TYPE_SIMD_IMAGES_1x16_2_FP16,
   LAST_TEST,
};



void fillMatrices(void)
{
   for (int y = 0; y < h; ++y)
   {
      for (int x = 0; x < w; ++x)
      {
         src0[y * w + x] = 2.f + ((float) rand()) / ((float) RAND_MAX);
         hsrc0[y * w + x] = float_to_half(src0[y * w + x]);
      }
   }

   for (int y = 0; y < w; ++y)
   {
      for (int x = 0; x < w1; ++x)
      {
         src1[y * w1 + x] = 2.f + ((float) rand()) / ((float) RAND_MAX);
         hsrc1[y * w1 + x] = float_to_half(src1[y * w1 + x]);
      }
   }
}

void cpuMul(void)
{
   assert(have_fp32_ref == false);
   const int K = w;             // since matrices are square
   for (int y = 0; y < h; ++y)
   {
      for (int x = 0; x < w1; ++x)
      {

         float sum = 0.f;
         for (int i = 0; i < w; ++i)
         {
            sum += src0[i + y * w] * src1[x + i * w1];
         }

         cpu_dst[x + y * w1] = sum;
      }
   }
   have_fp32_ref = true;

}

void cpuMul_fp16(void)
{
   assert(have_fp16_ref == false);
   const int K = w;             // since matrices are square
   for (int y = 0; y < h; ++y)
   {
      for (int x = 0; x < w1; ++x)
      {
         unsigned short hsum = 0;
         float fsum = 0.0f;
         for (int i = 0; i < w; ++i)
         {
            float s0 = half_to_float(hsrc0[i + y * w]);
            float s1 = half_to_float(hsrc1[x + i * w1]);
            hsum = float_to_half(half_to_float(hsum) + half_to_float(float_to_half(s0 * s1)));
            fsum = fsum + s0 * s1;
         }
         hcpu_dst0[x + y * w1] = hsum;
         hcpu_dst1[x + y * w1] = fsum;
      }
   }
   have_fp16_ref = true;
}

int check(void)
{
   const float tolerance = 0.001f;
   if (!have_fp32_ref)
   {
      cpuMul();
   }

   int ne = 0;
   float err = 0.f;
   for (int i = 0; i < h * w1; ++i)
   {
      float test = gpu_dst[i];
      float ref = cpu_dst[i];
      float localErr = fabs(test - ref) / maxval(fabs(test), fabs(ref));
      if (localErr >= tolerance && ne < 10)
      {
         ne++;
         printf("Error, index %d: Wanted %f, got %f\n", i, ref, test);
      }
      err = maxval(localErr, err);
   }
   printf(" MaxErr = %f ->", err);
   return err < tolerance;
   return 0;
}

int check_fp16(void)
{
   const float tolerance = 0.005f;
   if (!have_fp16_ref)
   {
      cpuMul_fp16();
   }

   int ne = 0;
   float err = 0.f;
   for (int i = 0; i < h * w1; ++i)
   {
      float test = half_to_float(hgpu_dst[i]);
      float ref0 = half_to_float(hcpu_dst0[i]);
      float ref1 = hcpu_dst1[i];
      float localErr0 = fabs(test - ref0) / maxval(fabs(test), fabs(ref0));
      float localErr1 = fabs(test - ref1) / maxval(fabs(test), fabs(ref1));
      float localErr = minval(localErr0, localErr1);
      if (localErr >= tolerance && ne < 10)
      {
         ne++;
         printf("Error, index %d: Wanted %f or %f, got %f\n", i, ref0,
                ref1, test);
      }
      err = maxval(localErr, err);

   }
   printf(" MaxErr = %f ->", err);
   return err < tolerance;
   return 0;
}


void runKernel(cl_kernel kernel, cl_mem dst, const size_t * global,
	       const size_t * local, unsigned int iter)
{
   cl_int err;


   /* Run the kernels */

   simpleTime tStart,tEnd;
   simpleGetTime(&tStart);

   for (int i = 0; i < iter; ++i)
   {
     err =
       clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global,
			      local, 0, NULL, NULL);
     CHK_ERR(err);
   }
   clFinish(queue);

   simpleGetTime(&tEnd);
   const double cpu_msec=simpleTimeDiffMsec(tEnd,tStart);


   void *p = NULL;
   if (dst == cl_dst || dst == cl_hdst)
   {
      p = clEnqueueMapBuffer(queue,
                             dst,
                             CL_TRUE,
                             CL_MAP_READ, 0, sz_dst, 0, NULL, NULL, &err);
      CHK_ERR(err);
   }
   else
   {
      const
      size_t origin[3] = { 0, 0, 0 };
      size_t wsize = (dst == mi_hdst ? size_t(w1) / 2 : size_t(w1));
      const
      size_t region[3] = { wsize, size_t(h), 1 };
      size_t imageRowPitch = 0;
      p = clEnqueueMapImage(queue,
                            dst,
                            CL_TRUE,
                            CL_MAP_READ,
                            origin,
                            region, &imageRowPitch, NULL, 0, NULL, NULL, &err);
      CHK_ERR(err);
   }

   printf
     ("%7.1lf %7.1lf %7.1lf %%",
      (double) cpu_msec / (double) (iter),
      (double) (2.0 * w * h * w1 * iter) / (double) (cpu_msec /
						     1000.) * 1e-9f,
      ((double) (2.0 * w * h * w1 * iter) /
       (double) (cpu_msec / 1000.) * 1e-9f /
       (double) (theoretical_peak_float_perf)) * 100.0);
   fflush(stdout);   


#ifdef CHECK_RESULT_CORRECTNESS
   printf(" ..checking.. "); fflush(stdout);

   int
    success = (dst == cl_hdst || dst == mi_hdst) ? check_fp16() : check();

   if (success)
      printf(" [PASSED]\n");
   else
   {
      assert(0);
      printf(" [FAILED]\n");
   }
#else
   printf("\n");
#endif
   fflush(stdout);

   if (p)
   {
      err = clEnqueueUnmapMemObject(queue, dst, p, 0, NULL, NULL);
      CHK_ERR(err);
   }
}


void blockMatrixMultiplication(const char *kernelName, cl_mem src0, cl_mem src1,
                          cl_mem dst, size_t lx, size_t ly, size_t dx,
                          size_t dy)
{
   cl_int err;
   printf("%-36s ", kernelName);
   cl_kernel kernel = getKernel(program, kernelName);
   if (kernel == NULL)
   {
      printf(" [invalid]\n");
      return;
   }
   size_t maxlws;
   clGetKernelWorkGroupInfo(kernel, NULL, CL_KERNEL_WORK_GROUP_SIZE,
                            sizeof(maxlws), &maxlws, NULL);
   if (maxlws < lx * ly)
   {
      printf(" [not supported]\n");
      return;
   }
   err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &src0);
   CHK_ERR(err);
   err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &src1);
   CHK_ERR(err);
   err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &dst);
   CHK_ERR(err);
   err = clSetKernelArg(kernel, 3, sizeof(int), &w);
   CHK_ERR(err);
   err = clSetKernelArg(kernel, 4, sizeof(int), &w1);
   CHK_ERR(err);
   const size_t global[] = { w1 / dx, h / dy };
   const
   size_t local[] = { lx, ly };

   if (0==strncmp(kernelName,"Unoptimized",11)) 
   {
     //unoptimized version is slow and needs fewer iterations
     runKernel(kernel, dst, global, local, 10);
   }
   else
   {
     runKernel(kernel, dst, global, local, TEST_ITERATIONS);
   }
}

void runTests(cl_uint start, cl_uint end)
{

   printf("# name                                 time(ms) GFLOPS  Efficiency\n");
   fflush(stdout);


   for (cl_uint test = start; test <= end; test++)
   {
      switch (test)
      {
      case TEST_TYPE_UNOPTIMIZED:
         blockMatrixMultiplication("Unoptimized", cl_src0, cl_src1, cl_dst,
                                   8, 8, 1, 1);
         break;
      case TEST_TYPE_SIMD_4x8x8:
         blockMatrixMultiplication("L3_SIMD_4x8x8", cl_src0, cl_src1,
                                   cl_dst, 8, 8, 4, 8);
         break;
      case TEST_TYPE_SIMD_IMAGESRW_2x32:
         blockMatrixMultiplication("MediaBlockRW_SIMD_2x32", mi_src0,
                                   mi_src1, mi_dst, 8, 1, 2, 32);
         break;
      case TEST_TYPE_SIMD_IMAGES_1x16_2_FP16:
         blockMatrixMultiplication("MediaBlockRead_SIMD_1x16_2_fp16",
                                   mi_hsrc0, mi_hsrc1, cl_hdst, 16, 1, 1, 16);
         break;
      default:
         break;
      }
   }
}

void help(int argc, char **argv)
{
   printf
       ("Usage: %s [kernel name] [matrix size] [kernel build option] [max gpu frequency in MHz]\n",
        argv[0]);
   printf("  kernel name             : all\n");
   printf
       ("  matrix size             : %d (square mat) or %dx%dx%d (non-square mat)\n",
        dimM, dimM, dimK, dimN);
   printf("  kernel build option     : -cl-mad-enable\n");
   printf("  max gpu frequency in MHz: 0\n");
}


int main(int argc, char **argv)
{

   cl_uint test_type = TEST_TYPE_ALL;

   if (argc > 1)
   {
      if (!strcmp(argv[1], "help"))
      {
         help(argc, argv);
         return 0;
      }
      else
      {
         test_type = TEST_TYPE_INVALID;
         std::string tmp = argv[1];
         if (tmp.compare("all") == 0)
            test_type = TEST_TYPE_ALL;
         if (tmp.compare("unoptimized") == 0)
            test_type = TEST_TYPE_UNOPTIMIZED;
         if (tmp.compare("SIMD_4x8x8") == 0)
            test_type = TEST_TYPE_SIMD_4x8x8;
         if (tmp.compare("SIMD_ImagesRW_2x32") == 0)
            test_type = TEST_TYPE_SIMD_IMAGESRW_2x32;
         if (tmp.compare("SIMD_Images_1x16_2_fp16") == 0)
            test_type = TEST_TYPE_SIMD_IMAGES_1x16_2_FP16;
         if (test_type == TEST_TYPE_INVALID)
         {
            printf("invalid test type specified.  Using SIMD_4x8x8\n");
            test_type = TEST_TYPE_SIMD_4x8x8;
         }
      }
   }
   if (argc > 2)
   {
      if (sscanf(argv[2], "%dx%dx%d", &dimM, &dimK, &dimN) != 3)
      {
         int d = strtol(argv[2], NULL, 10);
         if (d != 0)
         {
            dimM = dimK = dimN = d;
         }
      }

      if ((dimM<64) || (dimK<64) || (dimN<64))
      {
	puts("Minimum matrix dimension is 64");
	exit(1);
      }

      if ( (dimM != (dimM & (~dimM + 1))) ||
	   (dimK != (dimK & (~dimK + 1))) ||
	   (dimN != (dimN & (~dimN + 1))) )
      {
	puts("Matrix dimensions must be a power of two");
	exit(1);
      }

      h = dimM;
      w = dimK;
      w1 = dimN;
      sz_src0 = h * w * sizeof(float);
      sz_src1 = w * w1 * sizeof(float);
      sz_dst = h * w1 * sizeof(float);
   }


   /*Setup OCL */
   cl_int err;


   device = getDeviceID(CL_DEVICE_TYPE_GPU, NULL);


   cl_uint frequency;
   clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint),
                   &frequency, 0);
   max_gpu_clock_frequency_in_mhz = (float) frequency;

   clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint),
                   &max_compute_units_gpu, 0);

   if (max_gpu_clock_frequency_in_mhz != 0.0f)
   {
      theoretical_peak_float_perf = max_compute_units_gpu * (max_gpu_clock_frequency_in_mhz / 1000.0) * 16;     // peak GFLOP

      printf("# Max compute units  (GPU): %d\n", max_compute_units_gpu);
      printf("# Max clock freqency (GPU): %lf\n",
             max_gpu_clock_frequency_in_mhz);
      printf("# Peak float perf    (GPU): %lf\n", theoretical_peak_float_perf);
   }

   context = clCreateContext(0, 1, &device, NULL, NULL, &err);
   CHK_ERR(err);
   if (test_type == TEST_TYPE_INVALID)
   {
      printf("Invalid test name!\n");
      return -1;
   }


   printf("# build options: %s\n", buildoptions.c_str());
   program =
       getProgram(device, context, "../Opt_MatrixMultiplication.cl",
                  buildoptions.c_str());

#ifdef USE_OUT_OF_ORDER_QUEUE
  cl_queue_properties properties[]={CL_QUEUE_PROPERTIES,  CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,0};
  queue = clCreateCommandQueueWithProperties(context, device, properties, &err); 
  if (CL_SUCCESS != err) 
  {
      puts("Could not create command queue with CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE property.  Using default queue settings.");
      queue = clCreateCommandQueueWithProperties(context, device, NULL, &err);
  }
#else
   queue = clCreateCommandQueueWithProperties(context, device, NULL, &err);
#endif

   CHK_ERR(err);

   src0 = (float *) _aligned_malloc(sz_src0, 4096);
   src1 = (float *) _aligned_malloc(sz_src1, 4096);
   gpu_dst = (float *) _aligned_malloc(sz_dst, 4096);
   cpu_dst = (float *) _aligned_malloc(sz_dst, 4096);

   hsrc0 = (unsigned short *) _aligned_malloc(sz_src0, 4096);
   hsrc1 = (unsigned short *) _aligned_malloc(sz_src1, 4096);
   hgpu_dst = (unsigned short *) _aligned_malloc(sz_dst, 4096);
   hcpu_dst0 = (unsigned short *) _aligned_malloc(sz_dst, 4096);
   hcpu_dst1 = (float *) _aligned_malloc(sz_dst, 4096);

   printf("# matrix size: %dx%dx%d\n", h, w, w1);
   fflush(stdout);
   fillMatrices();
   
	  
   cl_src0 = clCreateBuffer(context, CL_MEM_USE_HOST_PTR, sz_src0, src0, &err);
   CHK_ERR(err);
   cl_src1 = clCreateBuffer(context, CL_MEM_USE_HOST_PTR, sz_src1, src1, &err);
   CHK_ERR(err);
   cl_dst = clCreateBuffer(context, CL_MEM_USE_HOST_PTR, sz_dst, gpu_dst, &err);
   CHK_ERR(err);

   cl_image_format imageFormat;
   imageFormat.image_channel_data_type = CL_FLOAT;
   imageFormat.image_channel_order = CL_RGBA;

   cl_image_desc desc;
   memset(&desc, 0, sizeof(desc));
   desc.image_type = CL_MEM_OBJECT_IMAGE2D;


   desc.image_width = w / 4;
   desc.image_height = h;
   desc.image_row_pitch = w * sizeof(float);
   im_src0 =
       clCreateImage(context, CL_MEM_USE_HOST_PTR, &imageFormat, &desc,
                     src0, &err);
   CHK_ERR(err);
   desc.image_width = w1 / 4;
   desc.image_height = w;
   desc.image_row_pitch = w1 * sizeof(float);
   im_src1 =
       clCreateImage(context, CL_MEM_USE_HOST_PTR, &imageFormat, &desc,
                     src1, &err);
   CHK_ERR(err);

   cl_image_format mbr_imageFormat;
   mbr_imageFormat.image_channel_data_type = CL_UNSIGNED_INT8;
   mbr_imageFormat.image_channel_order = CL_RGBA;

   desc.image_width = w;
   desc.image_height = h;
   desc.image_row_pitch = w * sizeof(float);

   mi_src0 =
       clCreateImage(context, CL_MEM_USE_HOST_PTR, &mbr_imageFormat, &desc,
                     src0, &err);
   CHK_ERR(err);


   desc.image_width = w1;
   desc.image_height = w;
   desc.image_row_pitch = w1 * sizeof(float);
   mi_src1 =
       clCreateImage(context, CL_MEM_USE_HOST_PTR, &mbr_imageFormat, &desc,
                     src1, &err);
   CHK_ERR(err);


   desc.image_width = w1;
   desc.image_height = h;
   desc.image_row_pitch = w1 * sizeof(float);
   mi_dst =
       clCreateImage(context, CL_MEM_USE_HOST_PTR, &mbr_imageFormat, &desc,
                     gpu_dst, &err);
   CHK_ERR(err);

   cl_hsrc0 =
       clCreateBuffer(context, CL_MEM_USE_HOST_PTR, sz_src0, hsrc0, &err);
   CHK_ERR(err);
   cl_hsrc1 =
       clCreateBuffer(context, CL_MEM_USE_HOST_PTR, sz_src1, hsrc1, &err);
   CHK_ERR(err);
   cl_hdst =
       clCreateBuffer(context, CL_MEM_USE_HOST_PTR, sz_dst, hgpu_dst, &err);
   CHK_ERR(err);

   // For the half (fp16) media block images, we can either create an image 
   // with a 16-bit image format, or an image with a 32-bit image format and
   // half the width.  Since the image is byte addressed, it's not clear
   // which is the better way to go.  For now, we'll go with a 32-bit image
   // and half the width, since this has the (slight) benefit of correct
   // bounds checking.

   memset(&desc, 0, sizeof(desc));
   desc.image_type = CL_MEM_OBJECT_IMAGE2D;
   desc.image_width = w / 2;
   desc.image_height = h;
   desc.image_row_pitch = w / 2 * sizeof(float);
   mi_hsrc0 =
       clCreateImage(context, CL_MEM_USE_HOST_PTR, &mbr_imageFormat, &desc,
                     hsrc0, &err);
   CHK_ERR(err);

   desc.image_width = w1 / 2;
   desc.image_height = w;
   desc.image_row_pitch = w1 / 2 * sizeof(float);
   mi_hsrc1 =
       clCreateImage(context, CL_MEM_USE_HOST_PTR, &mbr_imageFormat, &desc,
                     hsrc1, &err);
   CHK_ERR(err);

   desc.image_width = w1 / 2;
   desc.image_height = h;
   desc.image_row_pitch = w1 / 2 * sizeof(float);
   mi_hdst =
       clCreateImage(context, CL_MEM_USE_HOST_PTR, &mbr_imageFormat, &desc,
                     hgpu_dst, &err);
   CHK_ERR(err);

   cl_image_format h_imageFormat;
   h_imageFormat.image_channel_data_type = CL_HALF_FLOAT;
   h_imageFormat.image_channel_order = CL_RGBA;

   memset(&desc, 0, sizeof(desc));
   desc.image_type = CL_MEM_OBJECT_IMAGE2D;
   desc.image_width = w / 4;
   desc.image_height = h;
   desc.image_row_pitch = (w / 2) * sizeof(float);
   im_hsrc0 =
       clCreateImage(context, CL_MEM_USE_HOST_PTR, &h_imageFormat, &desc,
                     hsrc0, &err);
   CHK_ERR(err);

   desc.image_width = w1 / 4;
   desc.image_height = w;
   desc.image_row_pitch = (w1 / 2) * sizeof(float);
   im_hsrc1 =
       clCreateImage(context, CL_MEM_USE_HOST_PTR, &h_imageFormat, &desc,
                     hsrc1, &err);
   CHK_ERR(err);



   /* Run various implementations */
   switch (test_type)
   {
   case TEST_TYPE_ALL:
      runTests(TEST_TYPE_ALL, LAST_TEST);
      break;
   default:
      runTests(test_type, test_type);
      break;
   }


   if (src0)
   {
      _aligned_free(src0);
      src0 = NULL;
   }
   if (src1)
   {
      _aligned_free(src1);
      src1 = NULL;
   }
   if (gpu_dst)
   {
      _aligned_free(gpu_dst);
      gpu_dst = NULL;
   }
   if (cpu_dst)
   {
      _aligned_free(cpu_dst);
      cpu_dst = NULL;
   }
   if (hsrc0)
   {
      _aligned_free(hsrc0);
      hsrc0 = NULL;
   }
   if (hsrc1)
   {
      _aligned_free(hsrc1);
      hsrc1 = NULL;
   }
   if (hgpu_dst)
   {
      _aligned_free(hgpu_dst);
      hgpu_dst = NULL;
   }
   if (hcpu_dst0)
   {
      _aligned_free(hcpu_dst0);
      hcpu_dst0 = NULL;
   }
   if (hcpu_dst1)
   {
      _aligned_free(hcpu_dst1);
      hcpu_dst1 = NULL;
   }

   if (cl_src0)
   {
      clReleaseMemObject(cl_src0);
      cl_src0 = NULL;
   }
   if (cl_src1)
   {
      clReleaseMemObject(cl_src1);
      cl_src1 = NULL;
   }
   if (cl_dst)
   {
      clReleaseMemObject(cl_dst);
      cl_dst = NULL;
   }
   if (im_src0)
   {
      clReleaseMemObject(im_src0);
      im_src0 = NULL;
   }
   if (im_src1)
   {
      clReleaseMemObject(im_src1);
      im_src1 = NULL;
   }
   if (mi_src0)
   {
      clReleaseMemObject(mi_src0);
      mi_src0 = NULL;
   }
   if (mi_src1)
   {
      clReleaseMemObject(mi_src1);
      mi_src1 = NULL;
   }
   if (mi_dst)
   {
      clReleaseMemObject(mi_dst);
      mi_dst = NULL;
   }
   if (cl_hsrc0)
   {
      clReleaseMemObject(cl_hsrc0);
      cl_hsrc0 = NULL;
   }
   if (cl_hsrc1)
   {
      clReleaseMemObject(cl_hsrc1);
      cl_hsrc1 = NULL;
   }
   if (cl_hdst)
   {
      clReleaseMemObject(cl_hdst);
      cl_hdst = NULL;
   }
   if (im_hsrc0)
   {
      clReleaseMemObject(im_hsrc0);
      im_hsrc0 = NULL;
   }
   if (im_hsrc1)
   {
      clReleaseMemObject(im_hsrc1);
      im_hsrc1 = NULL;
   }
   if (mi_hsrc0)
   {
      clReleaseMemObject(mi_hsrc0);
      mi_hsrc0 = NULL;
   }
   if (mi_hsrc1)
   {
      clReleaseMemObject(mi_hsrc1);
      mi_hsrc1 = NULL;
   }

   return 0;
}
