cmake_minimum_required(VERSION 2.8.3)

if(NOT DEFINED HIP_PATH)
    if(NOT DEFINED ENV{HIP_PATH})
        set(HIP_PATH /opt/rocm/hip CACHE PATH Path to which HIP has been installed)
    else()
        set(HIP_PATH {HIP_PATH} CACHE PATH Path to which HIP has been installed)
    endif()
endif()
set(CMAKE_MODULE_PATH /public/software/compiler/dtk-23.04/hip/cmake )

project(12_cmake)

find_package(HIP QUIET)
if(HIP_FOUND)
    message(STATUS Found HIP:  )
else()
    message(FATAL_ERROR Could not find HIP. Ensure that HIP is either installed in /opt/rocm/hip or the variable HIP_PATH is set to point to the right location.)
endif()

set(MY_SOURCE_FILES MatrixTranspose.cpp)
set(MY_TARGET_NAME MatrixTranspose)
set(MY_HIPCC_OPTIONS)
set(MY_HCC_OPTIONS)
set(MY_NVCC_OPTIONS)

hip_add_executable(${MY_TARGET_NAME} ${MY_SOURCE_FILES} HIPCC_OPTIONS ${MY_HIPCC_OPTIONS} HCC_OPTIONS ${MY_HCC_OPTIONS} NVCC_OPTIONS ${MY_NVCC_OPTIONS})
set_target_properties(${MY_TARGET_NAME} PROPERTIES LINKER_LANGUAGE HIP)
target_link_libraries(${MY_TARGET_NAME} PRIVATE hip::device)
