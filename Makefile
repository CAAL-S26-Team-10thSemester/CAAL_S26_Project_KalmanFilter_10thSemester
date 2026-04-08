# ─────────────────────────────────────────────
# Toolchain for QEMU Programs
# ─────────────────────────────────────────────
AS_QEMU   = riscv64-linux-gnu-as
LD_QEMU   = riscv64-linux-gnu-ld
GCC_QEMU  = riscv64-linux-gnu-gcc
GDB_QEMU  = gdb-multiarch
QEMU      = /usr/local/bin/qemu-riscv64

# Plugin Configuration
PLUGIN ?= libinsn.so
PLUGIN_SYS ?= /usr/local/lib/qemu/plugins/$(PLUGIN)
PLUGIN_BUILD ?= build/tests/plugin/$(PLUGIN)
PLUGIN_PATH := $(shell \
	if [ -f $(PLUGIN_SYS) ]; then echo $(PLUGIN_SYS); \
	elif [ -f $(PLUGIN_BUILD) ]; then echo $(PLUGIN_BUILD); \
	else echo "NOT_FOUND"; fi)

# ─────────────────────────────────────────────
# Toolchain for Kalman Filter / Matrix ASM
# ─────────────────────────────────────────────
TOOLCHAIN_PREFIX ?= riscv64-linux-gnu
CC       = $(TOOLCHAIN_PREFIX)-gcc
AS       = $(TOOLCHAIN_PREFIX)-gcc     # use gcc as assembler
LD       = $(TOOLCHAIN_PREFIX)-gcc
SPIKE    ?= spike
PK       ?= pk

ARCH     =rv64imfd
ABI      = lp64d
# CFLAGS   = -march=$(ARCH) -mabi=$(ABI) -O0 -g -Wall
# LDFLAGS  = -march=$(ARCH) -mabi=$(ABI) -lm -static
CFLAGS  = -march=rv64imfd -mabi=lp64d -O0 -g -Wall
LDFLAGS = -march=rv64imfd -mabi=lp64d -lm -static

# Source files — Kalman Filter / Matrix
MATRIX_ASM   = matrix_asm.s
MATRIX_HDR   = matrix_asm.h
VERIFY_C     = verify_matrix_asm.c
LKF_ASM      = lkf_asm.s
EKF_ASM      = ekf_asm.s

# Object files
MATRIX_OBJ   = matrix_asm.o
VERIFY_OBJ   = verify_matrix_asm.o
LKF_OBJ      = lkf_asm.o
EKF_OBJ      = ekf_asm.o

# Executables
VERIFY_ELF   = verify_matrix_asm
LKF_ELF      = lkf_runner
EKF_ELF      = ekf_runner

# ─────────────────────────────────────────────
# General Targets
# ─────────────────────────────────────────────
.PHONY: all hello run-hello debug-hello \
        vector run-vector print_c run-print_c \
        plugin-check run-plugin \
        verify run_verify lkf ekf clean

all: verify

# ─────────────────────────────────────────────
# QEMU Assembly Programs
# ─────────────────────────────────────────────
hello: hello.s
	$(AS_QEMU) -o hello.o hello.s
	$(LD_QEMU) -o hello hello.o

run-hello: hello
	$(QEMU) ./hello

debug-hello: hello
	$(QEMU) -g 1234 ./hello

vector: vector.s
	$(AS_QEMU) -march=rv64gcv -o vector.o vector.s
	$(LD_QEMU) -o vector vector.o

run-vector: vector
	$(QEMU) -cpu rv64,v=true,vlen=128 ./vector

print_c: print_c.s
	$(GCC_QEMU) -o print_c print_c.s

run-print_c: print_c
	$(QEMU) ./print_c

plugin-check:
	@echo "Checking plugin support..."
	@$(QEMU) --help | grep plugin

run-plugin: hello
	@if [ "$(PLUGIN_PATH)" = "NOT_FOUND" ]; then \
		echo "❌ Plugin not found in:"; \
		echo "   - $(PLUGIN_SYS)"; \
		echo "   - $(PLUGIN_BUILD)"; \
		echo ""; \
		echo "👉 Fix:"; \
		echo "   1. Rebuild container"; \
		echo "   2. Ensure QEMU built with --enable-plugins"; \
		exit 1; \
	fi
	@echo "✅ Using plugin: $(PLUGIN_PATH)"
	@$(QEMU) -d plugin -plugin "$(PLUGIN_PATH)" -D insn.log ./hello

# ─────────────────────────────────────────────
# Matrix / Kalman Filter Targets
# ─────────────────────────────────────────────
$(MATRIX_OBJ): $(MATRIX_ASM) $(MATRIX_HDR)
	$(AS) $(CFLAGS) -c $< -o $@

$(VERIFY_OBJ): $(VERIFY_C) $(MATRIX_HDR)
	$(CC) $(CFLAGS) -c $< -o $@

$(VERIFY_ELF): $(VERIFY_OBJ) $(MATRIX_OBJ)
	$(LD) -march=$(ARCH) -mabi=$(ABI) $^ -o $@ -lm -static

verify: $(VERIFY_ELF)
	@echo ""
	@echo "--- Running numerical verification ---"
	$(QEMU) ./$(VERIFY_ELF)

run_verify: $(VERIFY_ELF)
	$(QEMU) ./$(VERIFY_ELF)

$(LKF_OBJ): $(LKF_ASM)
	$(AS) $(CFLAGS) -c $< -o $@

lkf: $(LKF_OBJ) $(MATRIX_OBJ)
	$(LD) $(LDFLAGS) $^ -o $(LKF_ELF)
	$(SPIKE) $(PK) ./$(LKF_ELF)

$(EKF_OBJ): $(EKF_ASM)
	$(AS) $(CFLAGS) -c $< -o $@

ekf: $(EKF_OBJ) $(MATRIX_OBJ)
	$(LD) $(LDFLAGS) $^ -o $(EKF_ELF)
	$(SPIKE) $(PK) ./$(EKF_ELF)

# ─────────────────────────────────────────────
# Cleanup
# ─────────────────────────────────────────────
clean:
	rm -f *.o hello vector print_c insn.log \
	      $(VERIFY_ELF) $(LKF_ELF) $(EKF_ELF)