# HCCL Tuner Plugin 单元测试 Makefile
#
# 使用方式：
#   source /usr/local/Ascend/cann/set_env.sh
#   make
#   ./test_plugin

ASCEND_HOME_PATH ?= /usr/local/Ascend/ascend-toolkit/latest
PLUGIN_INCLUDE ?= ../../../include

CC ?= gcc
CFLAGS := -O0 -g -Wall -Wextra -std=c11 -D_GNU_SOURCE
CFLAGS += -I.. -I$(ASCEND_HOME_PATH)/include -I$(ASCEND_HOME_PATH)/x86_64-linux/include -I$(PLUGIN_INCLUDE)
LDFLAGS := -lm

TARGET := test_plugin
SRC := test_plugin.c

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
