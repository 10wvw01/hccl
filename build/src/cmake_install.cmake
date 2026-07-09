# Install script for directory: /home/crx/code/hccl_three_sequence_rs/hccl_0701_br_sequence/hccl_2039/src

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
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/opp/built-in/op_impl/aicpu/config" TYPE FILE FILES "/home/crx/code/hccl_three_sequence_rs/hccl_0701_br_sequence/hccl_2039/build/src/libscatter_aicpu_kernel.json")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xhcclx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/x86_64-linux/lib64" TYPE SHARED_LIBRARY FILES "/home/crx/code/hccl_three_sequence_rs/hccl_0701_br_sequence/hccl_2039/build/src/libhccl.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/x86_64-linux/lib64/libhccl.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/x86_64-linux/lib64/libhccl.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/x86_64-linux/lib64/libhccl.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xhcclx" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xhcclx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/opp/built-in/op_graph/inc" TYPE FILE FILES "/home/crx/code/hccl_three_sequence_rs/hccl_0701_br_sequence/hccl_2039/src/common/op_graph/ops_proto_hccl.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xhcclx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/opp/built-in/op_graph/lib/linux/x86_64" TYPE SHARED_LIBRARY FILES "/home/crx/code/hccl_three_sequence_rs/hccl_0701_br_sequence/hccl_2039/build/src/libopgraph_hccl.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/opp/built-in/op_graph/lib/linux/x86_64/libopgraph_hccl.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/opp/built-in/op_graph/lib/linux/x86_64/libopgraph_hccl.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/opp/built-in/op_graph/lib/linux/x86_64/libopgraph_hccl.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xhcclx" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xhcclx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/x86_64-linux/include/es/" TYPE DIRECTORY FILES "/home/crx/code/hccl_three_sequence_rs/hccl_0701_br_sequence/hccl_2039/build/es_output/include/es_hccl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xhcclx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/ops_hccl/es_packages/whl" TYPE DIRECTORY FILES "/home/crx/code/hccl_three_sequence_rs/hccl_0701_br_sequence/hccl_2039/build/es_output/whl/")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xhcclx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/x86_64-linux/lib64" TYPE FILE FILES "/home/crx/code/hccl_three_sequence_rs/hccl_0701_br_sequence/hccl_2039/build/es_output/lib64/libes_hccl.so")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/crx/code/hccl_three_sequence_rs/hccl_0701_br_sequence/hccl_2039/build/src/common/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/crx/code/hccl_three_sequence_rs/hccl_0701_br_sequence/hccl_2039/build/src/ops/cmake_install.cmake")
endif()

