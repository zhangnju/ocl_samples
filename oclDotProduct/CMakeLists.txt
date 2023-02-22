# Minimal version of CMake
cmake_minimum_required (VERSION 3.11.4)
set(CMAKE_CXX_STANDARD 11) 
 
# Build type
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	message(STATUS "Setting build type to 'Debug' as none was specified.")
	set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)
	# Set the possible values of build type for cmake-gui
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
endif ()
 
# Define project name
project (OpenCL_Example)

set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/")
 
find_package( OpenCL REQUIRED )
find_package( OpenGL REQUIRED )
find_package( GLUT REQUIRED )
find_package( GLEW REQUIRED )

include_directories( ${OPENCL_INCLUDE_DIR} )
include_directories( ${OPENGL_INCLUDE_DIRS}  ${GLUT_INCLUDE_DIRS} ${GLEW_INCLUDE_DIRS} )
include_directories( include )

# Source code of application		
set (opencl_example_src src/oclDotProduct.cpp src/oclUtils.cpp src/shrUtils.cpp src/cmd_arg_reader.cpp)
 
# Compiler flags
if (CMAKE_COMPILER_IS_GNUCC)
    set (CMAKE_CXX_FLAGS "-D_REETRANT -Wall -Wextra -pedantic -Wno-long-long")
	if (CMAKE_BUILD_TYPE STREQUAL "Debug")
   	    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb -O0")
	elseif( CMAKE_BUILD_TYPE STREQUAL "Release" )
	    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG -O3 -fno-strict-aliasing")
	endif ()
endif (CMAKE_COMPILER_IS_GNUCC)

set(GLLIBS ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} )
 
# Set up executable
add_executable (opencl_example ${opencl_example_src})
target_link_libraries(opencl_example ${GLLIBS} ${OPENCL_LIBRARIES})
