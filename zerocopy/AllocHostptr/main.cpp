

//ZeroCopyAllocHostPtr

#include <stdio.h>
#include "host_common.h"
#include "scene.h"					//data structs and utilities associated with the AOBench scene 

int initializeDeviceData()
{	
	//initiate device side data objects (buffers and values)
	cl_int status = 0;

	//initialize host side data
	g_h = IMAGE_HEIGHT;
	g_w = IMAGE_WIDTH;
	g_bAlignedAlloc = false; //using aligned allocation for the output image?

	g_imageSize = sizeof(float) * g_w * g_h * 3;

	//note unsigned char, not float, for ppm file format
	g_img = (unsigned char *)malloc( g_w * g_h * 3);
	if(g_img == NULL)
	{
		printf("Error in initializeDeviceData(), can't allocate space for resulting image\n");
		exit(EXIT_FAILURE);
	}
	memset((void *)g_img, 0, g_w * g_h * 3);

	initializeScene();

	//allocate the OCL buffer memory objects 
	//notice the NULL here, not passing in a pointer so it is wherever driver wants to put it 
	//intel driver will allocate so it can be zero copy
	g_cl_mem_resultImage = clCreateBuffer(g_clContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, g_imageSize, NULL, &status);
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

	float *mappedBuffer = NULL;

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
	global_dim[0] = g_h;
	global_dim[1] = g_w;

	//launch Kernel, letting runtime select the wg size
	status = clEnqueueNDRangeKernel(g_clCommandQueue, cl_kernel_one_pixel, 2, NULL, global_dim, NULL, 0, NULL, NULL);
	testStatus(status, "clEnqueueNDRangeKernel error");

	clFinish(g_clCommandQueue);

	//consider putting a flush here, ensures i already put compute aspect
	auto start = chrono::system_clock::now();
	mappedBuffer = (float *)clEnqueueMapBuffer(g_clCommandQueue, g_cl_mem_resultImage, CL_TRUE, CL_MAP_READ, 0, g_imageSize, 0, NULL, NULL, NULL);
    auto end   = chrono::system_clock::now();
	auto duration = chrono::duration_cast< chrono::microseconds >(end - start);
	cout <<  "it takes " << double( duration.count() / 1000.0 ) << " ms" << endl;
	//convert fp values to integer for final image
	//doing on CPU for illustration only, could also be done on GPU!
	for(unsigned int y=0;y< g_h; y++)
	{
			for(unsigned int x=0;x<g_w;x++)
			{
				g_img[3 * (y * g_w + x) + 0] = clamp(mappedBuffer[3 * (y * g_w + x) + 0]);
				g_img[3 * (y * g_w + x) + 1] = clamp(mappedBuffer[3 * (y * g_w + x) + 1]);
				g_img[3 * (y * g_w + x) + 2] = clamp(mappedBuffer[3 * (y * g_w + x) + 2]);

				//printf("y=%d x=%d\t%.2f %.2f %.2f\n",y,x,mappedBuffer[3 * (y * g_w + x) + 0], mappedBuffer[3 * (y * g_w + x) + 1], mappedBuffer[3 * (y * g_w + x) + 2]);
				//printf("y=%d x=%d\t%d %d %d\n",y,x,g_img[3 * (y * g_w + x) + 0], g_img[3 * (y * g_w + x) + 1], g_img[3 * (y * g_w + x) + 2]);
			}
	}

	clEnqueueUnmapMemObject(g_clCommandQueue, g_cl_mem_resultImage, mappedBuffer, 0, NULL, NULL);
	clFinish(g_clCommandQueue);

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

	//note there is no pointer to free here, via aligned_free or regular free for the original image object
	//TODO: clarify with aaron, he was saying otherwise? might be right...think about it

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