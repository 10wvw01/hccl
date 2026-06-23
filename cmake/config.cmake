set(DEFAULT_BUILD_TYPE "Release")

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING "Choose the build type: Release/Debug" FORCE)
endif()

# ccl_kernel 无独立 Find 模块，创建本地 IMPORTED 目标
# CANN 9.0.0 devlib/device 无此库，仅在 >= 9.1.0 提供
if(PRODUCT_SIDE STREQUAL "device" AND BUILD_OPEN_PROJECT)
    if(CUSTOM_ASCEND_CANN_PACKAGE_PATH)
        set(_hccl_devlib_dir ${CUSTOM_ASCEND_CANN_PACKAGE_PATH}/devlib/device)
    elseif(DEFINED ASCEND_CANN_PACKAGE_PATH)
        set(_hccl_devlib_dir ${ASCEND_CANN_PACKAGE_PATH}/devlib/device)
    endif()
    if(DEFINED _hccl_devlib_dir AND EXISTS ${_hccl_devlib_dir}/libccl_kernel.so)
        if(NOT TARGET ccl_kernel)
            add_library(ccl_kernel SHARED IMPORTED)
            set_target_properties(ccl_kernel PROPERTIES
                IMPORTED_LOCATION "${_hccl_devlib_dir}/libccl_kernel.so"
            )
        endif()
    endif()
endif()

if(CUSTOM_ASCEND_CANN_PACKAGE_PATH)
    set(ASCEND_CANN_PACKAGE_PATH  ${CUSTOM_ASCEND_CANN_PACKAGE_PATH})
elseif(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_CANN_PACKAGE_PATH  $ENV{ASCEND_HOME_PATH})
elseif(DEFINED ENV{ASCEND_OPP_PATH})
    get_filename_component(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_OPP_PATH}/.." ABSOLUTE)
else()
    set(ASCEND_CANN_PACKAGE_PATH  "/usr/local/Ascend/ascend-toolkit/latest")
endif()

set(ASCEND_MOCKCPP_PACKAGE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

# if (NOT EXISTS "${ASCEND_CANN_PACKAGE_PATH}")
#     message(FATAL_ERROR "${ASCEND_CANN_PACKAGE_PATH} does not exist, please install the cann package and set environment variables.")
# endif()

# if (NOT EXISTS "${THIRD_PARTY_NLOHMANN_PATH}")
#     message(FATAL_ERROR "${THIRD_PARTY_NLOHMANN_PATH} does not exist, please check the setting of THIRD_PARTY_NLOHMANN_PATH.")
# endif()

#execute_process(COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/cmake/scripts/check_version_compatiable.sh
#                             ${ASCEND_CANN_PACKAGE_PATH}
#                             hccl
#                             ${CMAKE_CURRENT_SOURCE_DIR}/version.info
#    RESULT_VARIABLE result
#    OUTPUT_STRIP_TRAILING_WHITESPACE
#    OUTPUT_VARIABLE CANN_VERSION
#    )

#if (result)
#    message(FATAL_ERROR "${CANN_VERSION}")
#else()
#     string(TOLOWER ${CANN_VERSION} CANN_VERSION)
#endif()

if (CMAKE_INSTALL_PREFIX STREQUAL /usr/local)
    set(CMAKE_INSTALL_PREFIX     "${CMAKE_CURRENT_SOURCE_DIR}/output"  CACHE STRING "path for install()" FORCE)
endif ()

set(HI_PYTHON                     "python3"                       CACHE   STRING   "python executor")

message(STATUS "config.cmake KERNEL_MODE=${KERNEL_MODE} BUILD_OPEN_PROJECT=${BUILD_OPEN_PROJECT}")

#Device 构建安装目录
set(HCCL_DEVICE_BUILD_PATH ${CMAKE_BINARY_DIR}/device_build)
set(HCCL_DEVICE_INSTALL_PATH ${CMAKE_BINARY_DIR}/device_install)

set(INSTALL_LIBRARY_DIR ${CMAKE_SYSTEM_PROCESSOR}-linux/lib64)
set(INSTALL_INCLUDE_DIR ${CMAKE_SYSTEM_PROCESSOR}-linux/include)
set(INSTALL_AICPU_KERNEL_JSON_DIR opp/built-in/op_impl/aicpu)
set(INSTALL_DEVICE_TAR_DIR compat)

set(INSTALL_OPGRAPH_LIBRARY_DIR opp/built-in/op_graph/lib/linux/${CMAKE_SYSTEM_PROCESSOR})
set(INSTALL_OPGRAPH_INCLUDE_DIR opp/built-in/op_graph/inc)

if (ENABLE_TEST)
    set(CMAKE_SKIP_RPATH FALSE)
else ()
    set(CMAKE_SKIP_RPATH TRUE)
endif ()
