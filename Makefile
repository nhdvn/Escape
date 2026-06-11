CC      = gcc
ZKW  ?= 11
CFLAGS  = -O3 -march=native -maes -mavx512dq -std=c11 -Wall -Wextra \
           -Wno-unused-function \
           -D_POSIX_C_SOURCE=200112L \
           -Icompress -Iutils \
           -pthread -fopenmp -DZKW=$(ZKW) $(EXTRA)
LDFLAGS = -pthread -fopenmp -lm -lgmp

BUILD = build

vpath %.c . plhe channel server client compress utils

# Detect AVX-512 IFMA52 from the *actual* CFLAGS -- on IFMA-capable boxes we
# pick _m512.c (uses mont.h primitives), otherwise fall back to _gmpz.c
# (default).  Both files implement the same public symbols from compress.h,
# so only one is linked.
HAS_IFMA := $(shell echo | $(CC) $(CFLAGS) -dM -E -x c - 2>/dev/null | grep -q __AVX512IFMA__ && echo 1)
ifeq ($(HAS_IFMA),1)
  COMPRESS_PATH_OBJ       = $(BUILD)/_m512.o
  COMPRESS_PATH_SMALL_OBJ = $(BUILD)/_m512_small.o
else
  COMPRESS_PATH_OBJ       = $(BUILD)/_gmpz.o
  COMPRESS_PATH_SMALL_OBJ = $(BUILD)/_gmpz_small.o
endif

OBJS = $(BUILD)/plhe.o \
       $(BUILD)/memory.o \
       $(BUILD)/server.o \
       $(BUILD)/client.o \
       $(COMPRESS_PATH_OBJ) \
       $(BUILD)/bench.o

SMALL_FLAGS = -DPLHE_LAMBDA=128 -DPLHE_N1=8 -DPLHE_MR1=16 -DPLHE_MC1=16 \
              -DBLOCK_KIB=1 \
              -DSERVER_THREADS=2 -DAGG_BATCH=2 \
              -DCLIENT_THREADS=2 \
              -DPLHE_SIGMA_X10=32

SMALL_OBJS = $(BUILD)/plhe_small.o \
             $(BUILD)/memory_small.o \
             $(BUILD)/server_small.o \
             $(BUILD)/client_small.o \
             $(COMPRESS_PATH_SMALL_OBJ) \
             $(BUILD)/bench_small.o

.PHONY: all small clean

all: bench

small: bench_small

bench: $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

bench_small: $(SMALL_OBJS)
	$(CC) $(CFLAGS) $(SMALL_FLAGS) -o $@ $^ $(LDFLAGS)

# test_comp -- combined: Mont modmul micro-bench + compress pipeline test.
test_comp: $(COMPRESS_PATH_OBJ) $(BUILD)/test_comp.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# test_noise -- v1 layer-1 noise budget (no Mont / no GMP / no PIR pipeline).
test_noise: $(BUILD)/test_noise.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(BUILD)/%.o: %.c | $(BUILD)
	$(CC) $(CFLAGS) -c -o $@ $<

$(BUILD)/%_small.o: %.c | $(BUILD)
	$(CC) $(CFLAGS) $(SMALL_FLAGS) -c -o $@ $<

$(BUILD):
	mkdir -p $(BUILD)

# Header dependencies
$(BUILD)/plhe.o $(BUILD)/plhe_small.o \
$(BUILD)/memory.o $(BUILD)/memory_small.o \
$(BUILD)/server.o $(BUILD)/server_small.o \
$(BUILD)/client.o $(BUILD)/client_small.o \
$(BUILD)/bench.o $(BUILD)/bench_small.o: plhe/params.h

$(BUILD)/plhe.o $(BUILD)/plhe_small.o \
$(BUILD)/server.o $(BUILD)/server_small.o \
$(BUILD)/client.o $(BUILD)/client_small.o \
$(BUILD)/bench.o $(BUILD)/bench_small.o: plhe/plhe.h

$(BUILD)/memory.o $(BUILD)/memory_small.o \
$(BUILD)/server.o $(BUILD)/server_small.o \
$(BUILD)/client.o $(BUILD)/client_small.o \
$(BUILD)/bench.o $(BUILD)/bench_small.o: channel/channel.h

$(BUILD)/server.o $(BUILD)/server_small.o \
$(BUILD)/bench.o $(BUILD)/bench_small.o: server/server.h

$(BUILD)/client.o $(BUILD)/client_small.o \
$(BUILD)/bench.o $(BUILD)/bench_small.o: client/client.h

$(BUILD)/client.o $(BUILD)/client_small.o \
$(BUILD)/server.o $(BUILD)/server_small.o: utils/barrier.h

$(BUILD)/_m512.o $(BUILD)/_m512_small.o \
$(BUILD)/_gmpz.o $(BUILD)/_gmpz_small.o \
$(BUILD)/server.o $(BUILD)/server_small.o \
$(BUILD)/client.o $(BUILD)/client_small.o \
$(BUILD)/memory.o $(BUILD)/memory_small.o \
$(BUILD)/bench.o $(BUILD)/bench_small.o \
$(BUILD)/test_comp.o: compress/compress.h compress/mont.h compress/mont_n80.h compress/mont_n40.h

clean:
	rm -rf $(BUILD) bench bench_small test_comp test_noise
