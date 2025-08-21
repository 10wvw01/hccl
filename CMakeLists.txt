cmake_minimum_required(VERSION 3.16.0)
option(BUILD_OPEN_PROJECT "Build open hccl project." ON)
message(STATUS "KERNEL_MODE=${KERNEL_MODE}")
message(STATUS "BUILD_OPEN_PROJECT=${BUILD_OPEN_PROJECT}")
if(BUILD_OPEN_PROJECT AND KERNEL_MODE)
    # need to switch
    set(CMAKE_SYSTEM_NAME Linux)
    #Change the path to the absolute path of the cross compilation tool after the toolkit package is decompressed
    # set cross compile toolchain dir
    set(TOOLCHAIN_DIR $ENV{TOOLCHAIN_DIR})
    #Specify cross compiler
    message(STATUS "TOOLCHAIN_DIR=${TOOLCHAIN_DIR}")
    if (NOT "x${TOOLCHAIN_DIR}" STREQUAL "x")
        if (NOT IS_DIRECTORY ${TOOLCHAIN_DIR})
            message(FATAL_ERROR "specify cross compile toolchain directory(${TOOLCHAIN_DIR}) is not exist")
        endif()
    endif()
    set(CMAKE_CXX_COMPILER ${TOOLCHAIN_DIR}/bin/aarch64-target-linux-gnu-g++)
    set(CMAKE_C_COMPILER   ${TOOLCHAIN_DIR}/bin/aarch64-target-linux-gnu-gcc)
endif()
project(hccl)

if(BUILD_OPEN_PROJECT)
    include(cmake/config.cmake)
    add_subdirectory(src/domain/collective_communication)

    if(DEFINED ENABLE_TEST)
        add_subdirectory(test)
    endif()

    set(HCCL_HEAD
        inc/hccl/hccl.h
        inc/hccl/hccl_types.h
    )

    set(HCCL_JSON
        cmake/scripts/ccl_kernel.json
    )

    install(FILES
        ${HCCL_HEAD}
        ${HCCL_JSON}
        DESTINATION ${INSTALL_INCLUDE_DIR}/hccl/ OPTIONAL
    )
endif()
# this is a test commit