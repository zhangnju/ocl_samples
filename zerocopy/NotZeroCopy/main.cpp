

// NotZeroCopy

#include <stdio.h>
#include "host_common.h"
#include "scene.h"					//data structs and utilities associated with the AOBench scene 

int initializeDeviceData()
{
	//initialize host side data
	g_h = IMAGE_HEIGHT;
	g_w = IMAGE_WIDTH;

	g_bAlignedAlloc = false;

	g_imageSize = sizeof(float) * g_w * g_h * 3;
	g_f32_resultImage = (float *)malloc(g_imageSize);
	if(g_f32_resultImage == NULL)
	{
		printf("Error in initializeDeviceData(), can't allocate space for resulting image\n");
		exit(EXIT_FAILURE);
	}
	memset((void *)g_f32_resultImage, 0, g_imageSize);

	//note unsigned char, not float, for ppm file format
	g_img = (unsigned char *)malloc( g_w * g_h * 3);
	if(g_img == NULL)
	{
		printf("Error in initializeDeviceData(), can't allocate space for resulting image\n");
		exit(EXIT_FAILURE);
	}
	memset((void *)g_img, 0, g_w * g_h * 3);

	initializeScene();
	
	//initiate device side data objects (buffers and values)
	cl_int status = 0;

	//allocate the OCL buffer memory objects 
	g_cl_mem_resultImage = clCreateBuffer(g_clContext, CL_MEM_READ_WRITE, g_imageSize, NULL, &status);
	testStatus(status, "clCreateBuffer error");

	//spheres
	g_cl_mem_spheres = clCreateBuffer(g_clContext, CL_MEM_READ_ONLY, sizeof(g_spheres) * 3, NULL, &status);
	testStatus(status, "clCreateBuffer error");

	//planes
	g_cl_mem_planes = clCreateBuffer(g_clContext, CL_MEM_READ_ONLY, sizeof(g_plane), NULL, &status);
	testStatus(status, "clCreateBuffer error");

	return status;

}


int runCLKernels(void)
{
	cl_int status;

	memset((void *)g_f32_resultImage, 0, g_imageSize);

	status = clEnqueueWriteBuffer(g_clCommandQueue, g_cl_mem_resultImage, CL_TRUE, 0, g_imageSize, g_f32_resultImage, 0, NULL, NULL);
	testStatus(status, "clEnqueueWriteBuffer error");

	status = clEnqueueWriteBuffer(g_clCommandQueue, g_cl_mem_spheres, CL_TRUE, 0, sizeof(g_spheres), g_spheres, 0, NULL, NULL);
	testStatus(status, "clEnqueueWriteBuffer error");

	status = clEnqueueWriteBuffer(g_clCommandQueue, g_cl_mem_planes, CL_TRUE, 0, sizeof(g_plane), &g_plane, 0, NULL, NULL);
	testStatus(status, "clEnqueueWriteBuffer error");

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
	size_t wg_dim[2];
	global_dim[0] = g_h;
	global_dim[1] = g_w;
	//note these are not strintly the best way to launch on intel, use NULL if not using local memory in final code
	wg_dim[0] = 4;
	wg_dim[1] = 4;

	//launch Kernel
	status = clEnqueueNDRangeKernel(g_clCommandQueue, cl_kernel_one_pixel, 2, NULL, global_dim, wg_dim, 0, NULL, NULL);
	testStatus(status, "clEnqueueNDRangeKernel error");

	//consider putting a flush here, ensures i already put compute aspect
    clFinish(g_clCommandQueue);

    auto start = chrono::system_clock::now();
	//read back image
	status = clEnqueueReadBuffer(g_clCommandQueue, g_cl_mem_resultImage, CL_TRUE, 0, g_imageSize, g_f32_resultImage, 0, NULL, NULL);
	testStatus(status, "clEnqueueReadBuffer error");
    auto end   = chrono::system_clock::now();
	auto duration = chrono::duration_cast< chrono::microseconds >(end - start);
	cout <<  "it takes " << double( duration.count() / 1000.0 ) << " ms" << endl;
	//convert fp values to integer for final image
	//doing on CPU for illustration only, could also be done on GPU!
	for(unsigned int y=0;y< g_h; y++)
	{
			for(unsigned int x=0;x<g_w;x++)
			{
				g_img[3 * (y * g_w + x) + 0] = clamp(g_f32_resultImage[3 * (y * g_w + x) + 0]);
				g_img[3 * (y * g_w + x) + 1] = clamp(g_f32_resultImage[3 * (y * g_w + x) + 1]);
				g_img[3 * (y * g_w + x) + 2] = clamp(g_f32_resultImage[3 * (y * g_w + x) + 2]);
		//		printf("y=%d x=%d\t%.2f %.2f %.2f\n",y,x,g_f32_resultImage[3 * (y * g_w + x) + 0], g_f32_resultImage[3 * (y * g_w + x) + 1], g_f32_resultImage[3 * (y * g_w + x) + 2]);
		//		printf("y=%d x=%d\t%d %d %d\n",y,x,g_img[3 * (y * g_w + x) + 0], g_img[3 * (y * g_w + x) + 1], g_img[3 * (y * g_w + x) + 2]);
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

	
	if(g_f32_resultImage != NULL)
	{
		free(g_f32_resultImage);
		g_f32_resultImage = NULL;
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