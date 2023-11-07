cmake_minimum_required(VERSION 3.16)

if(NOT DEFINED HIP_PATH)
    if(NOT DEFINED ENV{HIP_PATH})
        set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed")
    else()
        set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
    endif()
endif()

if(NOT DEFINED ROCM_PATH)
    if(DEFINED ENV{ROCM_PATH})
        set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Path to which ROCM has been installed")
    elseif(DEFINED ENV{HIP_PATH})
        set(ROCM_PATH "$ENV{HIP_PATH}/.." CACHE PATH "Path to which ROCM has been installed")
    else()
        set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to which ROCM has been installed")
    endif()
endif()

if(NOT DEFINED HCC_PATH)
    if(DEFINED ENV{HCC_PATH})
        set(HCC_PATH $ENV{HCC_PATH} CACHE PATH "Path to which HCC has been installed")
    else()
        set(HCC_PATH "${ROCM_PATH}/hcc" CACHE PATH "Path to which HCC has been installed")
    endif()
    set(HCC_HOME "${HCC_PATH}")
endif()

if(NOT DEFINED HIP_CLANG_PATH)
    if(NOT DEFINED ENV{HIP_CLANG_PATH})
        set(HIP_CLANG_PATH "${ROCM_PATH}/llvm/bin" CACHE PATH "Path to which HIP compatible clang binaries have been installed")
    else()
        set(HIP_CLANG_PATH $ENV{HIP_CLANG_PATH} CACHE PATH "Path to which HIP compatible clang binaries have been installed")
    endif()
endif()

set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})
list(APPEND CMAKE_PREFIX_PATH
    "${HIP_PATH}/lib/cmake"
    "${HIP_PATH}/../lib/cmake" # hopefully catches all extra HIP dependencies
)

project(curand_cmake_test)

find_package(HIP QUIET)
if(HIP_FOUND)
    message(STATUS "Found HIP: " ${HIP_VERSION})
else()
    message(FATAL_ERROR "Could not find HIP. Ensure that HIP is either installed in /opt/rocm/hip or the variable HIP_PATH is set to point to the right location.")
endif()
find_package(hip REQUIRED)

#For ROCm >=3.5, wipe hip-clang specific interface options which are propagated
set_target_properties(hip::device PROPERTIES INTERFACE_COMPILE_OPTIONS "-fPIC")
set_target_properties(hip::device PROPERTIES INTERFACE_LINK_LIBRARIES "hip::host")

set(MY_SOURCE_FILES device_api_example.hip)
set(MY_TARGET_NAME device_api_example)
set(MY_HIPCC_OPTIONS "-fPIC -std=c++17 --offload-arch=gfx906")
set(MY_HCC_OPTIONS "-fopenmp --gpu-max-threads-per-block=1024")
set(CMAKE_HIP_ARCHITECTURES gfx906)
set(HIP_USE_PERL_SCRIPTS "0")
# configure Apache Arrow
set(ARROW_ROOT "/work/home/acychkqsp6/software/apache-arrow-13.0.0-hipcc")
list(APPEND CMAKE_PREFIX_PATH "${ARROW_ROOT}")
find_package(Arrow REQUIRED)

# configure Boost
#set(BOOST_ROOT "/work/home/acychkqsp6/software/boost_1_66_0")
#set(BOOST_INCLUDEDIR "/work/home/acychkqsp6/software/boost_1_66_0/install/include")
#set(BOOST_LIBRARYDIR "/work/home/acychkqsp6/software/boost_1_66_0/install/lib")
#list(APPEND CMAKE_PREFIX_PATH "${BOOST_ROOT}")
#set(Boost_USE_STATIC_LIBS OFF) 
#set(Boost_USE_MULTITHREADED ON)  
#set(Boost_USE_STATIC_RUNTIME OFF) 
#find_package(Boost REQUIRED)

#include_directories(${BOOST_INCLUDEDIR})
set_source_files_properties(${MY_SOURCE_FILES} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
hip_add_executable(${MY_TARGET_NAME} ${MY_SOURCE_FILES} HIPCC_OPTIONS ${MY_HIPCC_OPTIONS} HCC_OPTIONS ${MY_HCC_OPTIONS}) 
set_target_properties(${MY_TARGET_NAME} PROPERTIES LINKER_LANGUAGE HIP)
target_link_libraries(${MY_TARGET_NAME} PRIVATE hip::device)
target_link_libraries(${MY_TARGET_NAME} PRIVATE Arrow::arrow_static)
#target_link_libraries(${MY_TARGET_NAME} ${BOOST_LIBRARYDIR}) 
