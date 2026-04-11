# =============================================================================
#  Makefile  —  Kalman Filter Milestone-3
#  Targets:
#    make verify      compile matrix_asm.s + verify_matrix_asm.c, run tests
#    make lkf         compile lkf_asm.s + lkf_verify.c, run LKF verification
#    make all         run both verify and lkf
#    make clean       remove all generated files
# =============================================================================

# ─────────────────────────────────────────────
# Toolchain
# ─────────────────────────────────────────────
TOOLCHAIN_PREFIX ?= riscv64-linux-gnu
CC     = $(TOOLCHAIN_PREFIX)-gcc
QEMU   = /usr/local/bin/qemu-riscv64

ARCH   = rv64imfd
ABI    = lp64d
CFLAGS = -march=$(ARCH) -mabi=$(ABI) -O0 -g -Wall

# ─────────────────────────────────────────────
# Source / object / executable names
# ─────────────────────────────────────────────
MATRIX_ASM  = matrix_asm.s
MATRIX_HDR  = matrix_asm.h
MATRIX_OBJ  = matrix_asm.o

VERIFY_C    = verify_matrix_asm.c
VERIFY_OBJ  = verify_matrix_asm.o
VERIFY_ELF  = verify_matrix_asm

LKF_ASM     = lkf_asm.s
LKF_OBJ     = lkf_asm.o
LKF_VER_C   = lkf_verify.c
LKF_VER_OBJ = lkf_verify.o
LKF_VER_ELF = lkf_verify

# CSV file names (must be present in the working directory)
NOISY_CSV   = 3D Full Body Humain Gait Walking Dataset (Noisy Values).csv
REF_CSV     = lkf_results.csv

# ─────────────────────────────────────────────
# Phony targets
# ─────────────────────────────────────────────
.PHONY: all verify lkf clean

all: verify lkf

# ─────────────────────────────────────────────
# matrix_asm.o  — assemble shared matrix library
# ─────────────────────────────────────────────
$(MATRIX_OBJ): $(MATRIX_ASM) $(MATRIX_HDR)
	$(CC) $(CFLAGS) -c $< -o $@

# ─────────────────────────────────────────────
# verify_matrix_asm  — matrix unit tests
# ─────────────────────────────────────────────
$(VERIFY_OBJ): $(VERIFY_C) $(MATRIX_HDR)
	$(CC) $(CFLAGS) -c $< -o $@

$(VERIFY_ELF): $(VERIFY_OBJ) $(MATRIX_OBJ)
	$(CC) $(CFLAGS) $^ -o $@ -lm -static

verify: $(VERIFY_ELF)
	@echo ""
	@echo "=== Running matrix numerical verification ==="
	$(QEMU) ./$(VERIFY_ELF)

# ─────────────────────────────────────────────
# lkf_asm.o  — assemble LKF
# ─────────────────────────────────────────────
$(LKF_OBJ): $(LKF_ASM)
	$(CC) $(CFLAGS) -c $< -o $@

# ─────────────────────────────────────────────
# lkf_verify  — LKF §6 verification harness
# ─────────────────────────────────────────────
$(LKF_VER_OBJ): $(LKF_VER_C)
	$(CC) $(CFLAGS) -c $< -o $@

$(LKF_VER_ELF): $(LKF_VER_OBJ) $(LKF_OBJ) $(MATRIX_OBJ)
	$(CC) $(CFLAGS) $^ -o $@ -lm -static

lkf: $(LKF_VER_ELF)
	@echo ""
	@echo "=== Running LKF §6 verification ==="
	$(QEMU) ./$(LKF_VER_ELF) "$(NOISY_CSV)" "$(REF_CSV)"

# ─────────────────────────────────────────────
# Cleanup
# ─────────────────────────────────────────────
clean:
	rm -f *.o \
	      $(VERIFY_ELF) \
	      $(LKF_VER_ELF) \
	      lkf_asm_results.csv \
	      lkf_asm_verification.csv \
	      insn.log
