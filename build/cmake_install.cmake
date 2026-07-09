# Install script for directory: /home/crx/code/hccl_three_sequence_rs/hccl_0701_br_sequence/hccl_2039

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/crx/code/hccl_three_sequence_rs/hccl_0701_br_sequence/hccl_2039/build_out")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xhcclx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/info/hccl/script" TYPE DIRECTORY PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE DIR_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE FILES "/home/crx/code/hccl_three_sequence_rs/hccl_0701_br_sequence/hccl_2039/scripts/package/hccl/scripts/")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xhcclx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/info/hccl/script" TYPE FILE FILES
    "/home/crx/code/hccl_three_sequence_rs/hccl_0701_br_sequence/hccl_2039/build/_deps/cann-cmake-src/scripts/install/check_version_required.awk"
    "/home/crx/code/hccl_three_sequence_rs/hccl_0701_br_sequence/hccl_2039/build/_deps/cann-cmake-src/scripts/install/common_func.inc"
    "/home/crx/code/hccl_three_sequence_rs/hccl_0701_br_sequence/hccl_2039/build/_deps/cann-cmake-src/scripts/install/common_interface.sh"
    "/home/crx/code/hccl_three_sequence_rs/hccl_0701_br_sequence/hccl_2039/build/_deps/cann-cmake-src/scripts/install/common_interface.csh"
    "/home/crx/code/hccl_three_sequence_rs/hccl_0701_br_sequence/hccl_2039/build/_deps/cann-cmake-src/scripts/install/common_interface.fish"
    "/home/crx/code/hccl_three_sequence_rs/hccl_0701_br_sequence/hccl_2039/build/_deps/cann-cmake-src/scripts/install/version_compatiable.inc"
    "/home/crx/code/hccl_three_sequence_rs/hccl_0701_br_sequence/hccl_2039/build/_deps/cann-cmake-src/scripts/package/merge_binary_info_config.py"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xhcclx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/info/hccl" TYPE FILE RENAME "version.info" FILES "/home/crx/code/hccl_three_sequence_rs/hccl_0701_br_sequence/hccl_2039/build/version.hccl.info")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xhcclx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/x86_64-linux/conf" TYPE FILE FILES "/home/crx/code/hccl_three_sequence_rs/hccl_0701_br_sequence/hccl_2039/build/_deps/cann-cmake-src/scripts/package/cfg/path.cfg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xhcclx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/info/hccl/script" TYPE FILE FILES
    "/home/crx/code/hccl_three_sequence_rs/hccl_0701_br_sequence/hccl_2039/build/_deps/cann-cmake-src/scripts/install/install_common_parser.sh"
    "/home/crx/code/hccl_three_sequence_rs/hccl_0701_br_sequence/hccl_2039/build/_deps/cann-cmake-src/scripts/install/common_func_v2.inc"
    "/home/crx/code/hccl_three_sequence_rs/hccl_0701_br_sequence/hccl_2039/build/_deps/cann-cmake-src/scripts/install/common_installer.inc"
    "/home/crx/code/hccl_three_sequence_rs/hccl_0701_br_sequence/hccl_2039/build/_deps/cann-cmake-src/scripts/install/script_operator.inc"
    "/home/crx/code/hccl_three_sequence_rs/hccl_0701_br_sequence/hccl_2039/build/_deps/cann-cmake-src/scripts/install/version_cfg.inc"
    "/home/crx/code/hccl_three_sequence_rs/hccl_0701_br_sequence/hccl_2039/build/_deps/cann-cmake-src/scripts/install/multi_version.inc"
    )
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/crx/code/hccl_three_sequence_rs/hccl_0701_br_sequence/hccl_2039/build/src/cmake_install.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xhcclx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/x86_64-linux/include/hccl" TYPE FILE FILES
    "/home/crx/code/hccl_three_sequence_rs/hccl_0701_br_sequence/hccl_2039/include/hccl.h"
    "/home/crx/code/hccl_three_sequence_rs/hccl_0701_br_sequence/hccl_2039/include/hccl_mc2.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/crx/code/hccl_three_sequence_rs/hccl_0701_br_sequence/hccl_2039/build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
