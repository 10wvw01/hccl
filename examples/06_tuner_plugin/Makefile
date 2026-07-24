# HCCL Tuner Plugin 示例 Makefile
#
# 使用方式：
#   source /usr/local/Ascend/cann/set_env.sh
#   make
#
# 产物：hccl_tuner_example.so

ASCEND_HOME_PATH ?= /usr/local/Ascend/ascend-toolkit/latest
PLUGIN_INCLUDE ?= ../../include

CC ?= gcc
CFLAGS := -fPIC -shared -O2 -Wall -Wextra -std=c11
CFLAGS += -I$(ASCEND_HOME_PATH)/include -I$(ASCEND_HOME_PATH)/x86_64-linux/include -I$(PLUGIN_INCLUDE)

TARGET := hccl_tuner_example.so
SRC := plugin.c

.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $<

test:
	$(MAKE) -C test

clean:
	rm -f $(TARGET)
	$(MAKE) -C test clean
