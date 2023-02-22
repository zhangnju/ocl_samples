
//ZeroCopyUseHostPtr

#include <stdio.h>
#include "host_common.h"
#include "scene.h"					//data structs and utilities associated with the AOBench scene 

//this tests the zero copy performance
void PerfTestZeroCopy();

int initializeDeviceData()
{
	//initiate device side data objects (buffers and values)
	cl_int status = 0;

	//initialize host side data
	g_h = IMAGE_HEIGHT;
	g_w = IMAGE_WIDTH;
	g_bAlignedAlloc = ALIGNED_ALLOCATION; //use aligned allocation for output image buffer

	g_imageSize = sizeof(float) * g_w * g_h * 3;
	
	if(g_bAlignedAlloc == true)
	{
		//note the alignment and size requirements for current Intel Processor Graphics: requires 4096 byte page alignment and the size a multiple of 64 bytes
		g_f32_resultImage = (float *)aligned_malloc(g_imageSize, 4096);
        //validate the pointer
	    if(g_f32_resultImage == NULL)
	    {
		  printf("Failed to allocate space on host for result image.\n");
		  exit(EXIT_SUCCESS);
	    }
		//verify we actually created (or did not create if that was our goal for testing for example) a zero copy pointer
	    //on Intel Processor Graphics
	    status = verifyZeroCopyPtr(g_f32_resultImage, g_imageSize);
	    if(status == false)
	    {
		   printf("Using malloc, will not likely result in a valid zero copy buffer\n");
	    }
	    else
	    {
		   printf("Pointer follows rules for zero copy on Intel Processor Graphics\n");
	    }
	}
	else
	{
		g_f32_resultImage = (float *)malloc(g_imageSize);
		//validate the pointer
	    if(g_f32_resultImage == NULL)
	    {
		  printf("Failed to allocate space on host for result image.\n");
		  exit(EXIT_SUCCESS);
	    }
	}

	memset((void *)g_f32_resultImage, 0, g_imageSize);

	//note unsigned char, not float, for ppm file format
	//alignment not required here, just to write image to disk
	g_img = (unsigned char *)malloc( g_w * g_h * 3);
	memset((void *)g_img, 0, g_w * g_h * 3);

	initializeScene();
	
	//allocate the OCL buffer memory objects 
	g_cl_mem_resultImage = clCreateBuffer(g_clContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, g_imageSize, g_f32_resultImage, &status);
	testStatus(status, "Failed at clCreateBuffer\n");

	//spheres
	g_cl_mem_spheres = clCreateBuffer(g_clContext, CL_MEM_READ_ONLY, sizeof(g_spheres) * 3, NULL, &status);
	testStatus(status, "Failed at clCreateBuffer\n");

	//planes
	g_cl_mem_planes = clCreateBuffer(g_clContext, CL_MEM_READ_ONLY, sizeof(g_plane), NULL, &status);
	testStatus(status, "Failed at clCreateBuffer\n");

	return status;

}

void perfTestZeroCopy()
{
	//float totalTimeInSeconds = 0.0f;

    auto start = chrono::system_clock::now();
	//int iNumMapCalls = 100; //enough to amortize cost over enqueueing overhead in API
	//for(int i=0;i<iNumMapCalls;i++)
	{
		g_f32_resultImage = (float *)clEnqueueMapBuffer(g_clCommandQueue, g_cl_mem_resultImage, CL_TRUE, CL_MAP_READ, 0, g_imageSize, 0, NULL, NULL, NULL);
		clEnqueueUnmapMemObject(g_clCommandQueue, g_cl_mem_resultImage, g_f32_resultImage, 0, NULL, NULL); 
		clFinish(g_clCommandQueue);  //just to make sure
	}
	auto end   = chrono::system_clock::now();
	auto duration = chrono::duration_cast< chrono::microseconds >(end - start);

    
	//totalCounts.QuadPart = countAtEnd.QuadPart - countAtStart.QuadPart;
	//totalTimeInSeconds = (float) ((float)totalCounts.QuadPart / (float)countsPerSecond.QuadPart);
	if(g_bAlignedAlloc == true)
	{
		printf("Aligned Allocation: ");
	}
	else
	{
		printf("Unaligned Allocation: ");
	}
	cout <<  "it takes " << double( duration.count() / 1000.0 ) << " ms" << endl;
	//printf("%f seconds is total time, %f is time per map/unmap pair\n", totalTimeInSeconds, (float)(totalTimeInSeconds/iNumMapCalls));
	//convert fp values to integer for final image
	//doing on CPU for illustration only, could also be done on GPU!
	for(unsigned int y=0;y< g_h; y++)
	{
			for(unsigned int x=0;x<g_w;x++)
			{
				g_img[3 * (y * g_w + x) + 0] = clamp(g_f32_resultImage[3 * (y * g_w + x) + 0]);
				g_img[3 * (y * g_w + x) + 1] = clamp(g_f32_resultImage[3 * (y * g_w + x) + 1]);
				g_img[3 * (y * g_w + x) + 2] = clamp(g_f32_resultImage[3 * (y * g_w + x) + 2]);
			}
	}
}

int runCLKernels(void)
{
	cl_int status;

	//leaving these for now, as a render loop would also want these to be map/unmap
	status = clEnqueueWriteBuffer(g_clCommandQueue, g_cl_mem_spheres, CL_TRUE, 0, sizeof(g_spheres), g_spheres, 0, NULL, NULL);
	testStatus(status, NULL);

	status = clEnqueueWriteBuffer(g_clCommandQueue, g_cl_mem_planes, CL_TRUE, 0, sizeof(g_plane), &g_plane, 0, NULL, NULL);
	testStatus(status, NULL);

	status = clSetKernelArg(cl_kernel_one_pixel, 0, sizeof(cl_mem), (void*)&g_cl_mem_resultImage);
	testStatus(status, "clSetKernelArg error");
	status = clSetKernelArg(cl_kernel_one_pixel, 1, sizeof(cl_mem), (void*)&g_cl_mem_spheres);
	testStatus(status, "clSetKernelArg error");
	status = clSetKernelArg(cl_kernel_one_pixel, 2, sizeof(cl_mem), (void*)&g_cl_mem_planes);
	testStatus(status, "clSetKernelArg error");
	status = clSetKernelArg(cl_kernel_one_pixel, 3, sizeof(cl_int), (void*)&g_h);
	testStatus(status, "clSetKernelArg error");
	status = clSetKernelArg(cl_kernel_one_pixel, 4, sizeof(cl_int), (void*)&g_w);
	testStatus(status, "clSetKernelArg error");
	status = clSetKernelArg(cl_kernel_one_pixel, 5, sizeof(cl_int), (void*)&g_numSubSamples);
	testStatus(status, "clSetKernelArg error");


	//Create the NDRange
	size_t global_dim[2];
	global_dim[0] = g_h;
	global_dim[1] = g_w;

	//launch Kernel, letting runtime select the wg size
	status = clEnqueueNDRangeKernel(g_clCommandQueue, cl_kernel_one_pixel, 2, NULL, global_dim, NULL, 0, NULL, NULL);
	testStatus(status, "clEnqueueNDRangeKernel error");

	//consider putting a flush here, ensures i already put compute aspect
	clFinish(g_clCommandQueue);
	
	//this is the code to verify improved performance of zero copy, not needed 
	if(PERF_TEST_ZERO_COPY)
	{
		perfTestZeroCopy();
	}
	else 
	{
		auto start = chrono::system_clock::now();
		g_f32_resultImage = (float *)clEnqueueMapBuffer(g_clCommandQueue, g_cl_mem_resultImage, CL_TRUE, CL_MAP_READ, 0, g_imageSize, 0, NULL, NULL, NULL);
		auto end   = chrono::system_clock::now();
	    auto duration = chrono::duration_cast< chrono::microseconds >(end - start);
        cout <<  "it takes " << double( duration.count() / 1000.0 ) << " ms" << endl;
		
        clEnqueueUnmapMemObject(g_clCommandQueue, g_cl_mem_resultImage, g_f32_resultImage, 0, NULL, NULL);
	    clFinish(g_clCommandQueue);
		//convert fp values to integer for final image
		//doing on CPU for illustration only, could also be done on GPU!
		for(unsigned int y=0;y< g_h; y++)
		{
				for(unsigned int x=0;x<g_w;x++)
				{
					g_img[3 * (y * g_w + x) + 0] = clamp(g_f32_resultImage[3 * (y * g_w + x) + 0]);
					g_img[3 * (y * g_w + x) + 1] = clamp(g_f32_resultImage[3 * (y * g_w + x) + 1]);
					g_img[3 * (y * g_w + x) + 2] = clamp(g_f32_resultImage[3 * (y * g_w + x) + 2]);
				}
	    }
	}

	return status;
}


int cleanupHost()
{
	//cleanup the mallocd buffers
	if(g_clProgramString != NULL)
	{
		free(g_clProgramString);
		g_clProgramString = NULL;
	}

	if(g_bAlignedAlloc == true)
	{
		aligned_free(g_f32_resultImage);
	}
	else
	{
		free(g_f32_resultImage);
	}

	if(g_img != NULL)
	{
		free(g_img);
		g_img = NULL;
	}
	

	return SUCCESS;

}

int main()
{
	if(initializeHost() != SUCCESS)
	{
		printf("Error when initializing host\n");
		exit(EXIT_FAILURE);
	}

	if(initializeCL() != SUCCESS)
	{
		printf("Error when initializing OpenCL\n");
		exit(EXIT_FAILURE);
	}

	if(runCLKernels() != SUCCESS)
	{
		printf("Error when running CL kernels\n");
		exit(EXIT_FAILURE);
	}
	
	savePPM();

	if(cleanupCL() != SUCCESS)
	{
		printf("Error when cleaning up OpenCL\n");
		exit(EXIT_FAILURE);
	}

	if(cleanupHost() != SUCCESS)
	{
		printf("Error when cleaning up host\n");
		exit(EXIT_FAILURE);
	}

	printf("Success! Exiting now...\n");

	return 0;
}