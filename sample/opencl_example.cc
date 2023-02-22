#define __CL_ENABLE_EXCEPTIONS

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

const char * helloStr  = "__kernel void "
		"hello(void) "
		"{ "
		"  "
		"} ";

int main(void)
{
	cl_int err = CL_SUCCESS;
	try {

		std::vector<cl::Platform> platforms;
		cl::Platform::get(&platforms);
		if (platforms.size() == 0) {
			std::cout << "Platform size 0\n";
			return -1;
		}

		// Print number of platforms and list of platforms
		std::cout << "Platform number is: " << platforms.size() << std::endl;
		std::string platformVendor;
		for (unsigned int i = 0; i < platforms.size(); ++i) {
			platforms[i].getInfo((cl_platform_info)CL_PLATFORM_VENDOR, &platformVendor);
			std::cout << "Platform is by: " << platformVendor << std::endl;
		}

		cl_context_properties properties[] =
		{
				CL_CONTEXT_PLATFORM,
				(cl_context_properties)(platforms[0])(),
				0
		};
		cl::Context context(CL_DEVICE_TYPE_ALL, properties);

		std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

		// Print number of devices and list of devices
		std::cout << "Device number is: " << devices.size() << std::endl;
		for (unsigned int i = 0; i < devices.size(); ++i) {
			std::cout << "Device #" << i << ": " << devices[i].getInfo<CL_DEVICE_NAME>() << std::endl;
		}

		cl::Program::Sources source(1,
				std::make_pair(helloStr, strlen(helloStr)));
		cl::Program program_ = cl::Program(context, source);
		program_.build(devices);

		cl::Kernel kernel(program_, "hello", &err);

		cl::Event event;
		cl::CommandQueue queue(context, devices[0], 0, &err);
		queue.enqueueNDRangeKernel(
				kernel,
				cl::NullRange,
				cl::NDRange(4,4),
				cl::NullRange,
				NULL,
				&event);

		event.wait();
	}
	catch (cl::Error err) {
		std::cerr
		<< "ERROR: "
		<< err.what()
		<< "("
		<< err.err()
		<< ")"
		<< std::endl;
	}

	return EXIT_SUCCESS;

}
