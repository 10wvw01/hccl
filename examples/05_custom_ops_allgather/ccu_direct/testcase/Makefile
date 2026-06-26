# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

ifndef ASCEND_HOME_PATH
    $(error "ASCEND_HOME_PATH is not set, please ensure CANN is properly installed and \
             source environment variables by running `source /path/to/Ascend/cann/set_env.sh`")
endif

CXX := g++
CXXFLAGS := -std=c++14 \
        -Werror \
        -fstack-protector-strong \
        -fPIE \
        -O2 \
        -s \
        -DHOST_COMPILE \
        -D_GLIBCXX_USE_CXX11_ABI=0
LDFLAGS := -pie \
        -Wl,-z,relro \
        -Wl,-z,now \
        -Wl,-z,noexecstack \
        -Wl,--copy-dt-needed-entries

SOURCES = main.cc \
        ../op_host/allgather.cc \
        ../op_host/utils.cc \
        ../op_kernel_ccu/exec_op.cc \
        ../op_kernel_ccu/ccu_kernel.cc
ASCEND_INC_DIR = ${ASCEND_HOME_PATH}/include
ASCEND_PKG_INC_DIR = ${ASCEND_HOME_PATH}/pkg_inc
ASCEND_LIB_DIR = ${ASCEND_HOME_PATH}/lib64
LIBS = -L$(ASCEND_LIB_DIR) -lhcomm -lc_sec -lascendcl -lacl_rt
INCS = -I../inc \
       -I../op_host \
       -I../op_kernel_ccu \
       -I$(ASCEND_INC_DIR) \
       -I$(ASCEND_INC_DIR)/hccl \
       -I$(ASCEND_INC_DIR)/hcomm \
       -I$(ASCEND_INC_DIR)/hcomm/ccu \
       -I$(ASCEND_PKG_INC_DIR) \
       -I$(ASCEND_PKG_INC_DIR)/hcomm \
       -I$(ASCEND_PKG_INC_DIR)/hcomm/ccu
TARGET = custom_allgather_ccu

# Default target
all:
	$(CXX) $(CXXFLAGS) $(SOURCES) $(INCS) $(LDFLAGS) $(LIBS) -o ${TARGET}
	@echo "${TARGET} compile completed"

# Test target
test: 
	./$(TARGET)

# Clean build artifacts
clean:
	rm -f ${TARGET}

# Help
help:
	@echo ""
	@echo "HCCL Example: AllGather"
	@echo "======================="
	@echo ""
	@echo "This example shows how to use HcclAllGather() to run"
	@echo "the AllGather collective operation in a single process."
	@echo ""
	@echo "Targets:"
	@echo "  all       - Build the example (default)"
	@echo "  test      - Build and run test"
	@echo "  clean     - Remove build artifacts"
	@echo "  help      - Show this help"
	@echo ""

.PHONY: all test clean help
