# =============================================================================
#  Makefile  —  Kalman Filter Milestone-3
#  Targets:
#    make verify          compile matrix_asm.s + verify_matrix_asm.c, run tests
#    make lkf             compile lkf_asm.s + lkf_verify.c, run LKF verification
#    make ekf             compile ekf_asm.s + ekf_verify.c, run EKF
#    make ekf_verify_ref  run EKF with Python reference for §6 element-wise check
#    make all             run verify + lkf + ekf
#    make clean           remove all generated files
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

EKF_ASM     = ekf_asm.s
EKF_OBJ     = ekf_asm.o
EKF_VER_C   = ekf_verify.c
EKF_VER_OBJ = ekf_verify.o
EKF_VER_ELF = ekf_verify_bin

# ─────────────────────────────────────────────
# CSV file names
#
#  NOISY_CSV   — input: noisy measurements
#  LKF_REF_CSV — input: Python LKF reference (from previous milestone)
#  EKF_REF_CSV — input: Python EKF reference (from kalman-updated.py)
#                       DO NOT overwrite — this is the reference!
#  EKF_ASM_CSV — output: assembly EKF results (compared against EKF_REF_CSV)
# ─────────────────────────────────────────────
NOISY_CSV   = 3D Full Body Humain Gait Walking Dataset (Noisy Values).csv
LKF_REF_CSV = lkf_results.csv
EKF_REF_CSV = ekf_results.csv
EKF_ASM_CSV = ekf_asm_results.csv

# ─────────────────────────────────────────────
# Phony targets
# ─────────────────────────────────────────────
.PHONY: all verify lkf ekf ekf_verify_ref clean

all: verify lkf ekf

# ─────────────────────────────────────────────
# matrix_asm.o  — shared matrix library
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
# lkf_asm.o + lkf_verify
# ─────────────────────────────────────────────
$(LKF_OBJ): $(LKF_ASM)
	$(CC) $(CFLAGS) -c $< -o $@

$(LKF_VER_OBJ): $(LKF_VER_C)
	$(CC) $(CFLAGS) -c $< -o $@

$(LKF_VER_ELF): $(LKF_VER_OBJ) $(LKF_OBJ) $(MATRIX_OBJ)
	$(CC) $(CFLAGS) $^ -o $@ -lm -static

lkf: $(LKF_VER_ELF)
	@echo ""
	@echo "=== Running LKF §6 verification ==="
	$(QEMU) ./$(LKF_VER_ELF) "$(NOISY_CSV)" "$(LKF_REF_CSV)"

# ─────────────────────────────────────────────
# ekf_asm.o + ekf_verify_bin
#
# NOTE: ekf_asm.s calls state_init_F and state_init_Q defined in lkf_asm.s.
#       Therefore lkf_asm.o is included in the EKF link as well.
#
# ekf target:
#   argv[1] = NOISY_CSV    — input measurements
#   argv[2] = EKF_ASM_CSV  — output: assembly results (ekf_asm_results.csv)
#   argv[3] = EKF_REF_CSV  — input:  Python reference  (ekf_results.csv)
#
# The verifier:
#   1. Runs assembly EKF  → saves to ekf_asm_results.csv
#   2. Loads ekf_results.csv (Python reference from kalman-updated.py)
#   3. Compares element-wise → reports pass/fail
# ─────────────────────────────────────────────
$(EKF_OBJ): $(EKF_ASM)
	$(CC) $(CFLAGS) -c $< -o $@

$(EKF_VER_OBJ): $(EKF_VER_C) $(MATRIX_HDR)
	$(CC) $(CFLAGS) -c $< -o $@

$(EKF_VER_ELF): $(EKF_VER_OBJ) $(EKF_OBJ) $(LKF_OBJ) $(MATRIX_OBJ)
	$(CC) $(CFLAGS) $^ -o $@ -lm -static

# Run EKF:
#   - Assembly output → ekf_asm_results.csv  (never overwrites Python ref)
#   - Compares vs     → ekf_results.csv      (Python reference, must exist)
ekf: $(EKF_VER_ELF)
	@echo ""
	@echo "=== Running EKF ==="
	@if [ ! -f "$(EKF_REF_CSV)" ]; then \
	    echo "[ERROR] Python reference not found: $(EKF_REF_CSV)"; \
	    echo "        Run: python3 kalman-updated.py \"$(NOISY_CSV)\" $(EKF_REF_CSV)"; \
	    exit 1; \
	fi
	$(QEMU) ./$(EKF_VER_ELF) "$(NOISY_CSV)" "$(EKF_ASM_CSV)" "$(EKF_REF_CSV)"

# Run EKF with explicit Python reference for §6 element-wise check.
# Same as ekf target but more explicit — useful for debugging.
ekf_verify_ref: $(EKF_VER_ELF)
	@echo ""
	@echo "=== Running EKF with Python reference comparison ==="
	@if [ ! -f "$(EKF_REF_CSV)" ]; then \
	    echo "[ERROR] Python reference not found: $(EKF_REF_CSV)"; \
	    echo "        Run: python3 kalman-updated.py \"$(NOISY_CSV)\" $(EKF_REF_CSV)"; \
	    exit 1; \
	fi
	$(QEMU) ./$(EKF_VER_ELF) "$(NOISY_CSV)" "$(EKF_ASM_CSV)" "$(EKF_REF_CSV)"

# ─────────────────────────────────────────────
# Cleanup
# ─────────────────────────────────────────────
clean:
	rm -f *.o \
	      $(VERIFY_ELF) \
	      $(LKF_VER_ELF) \
	      $(EKF_VER_ELF) \
	      lkf_asm_results.csv \
	      lkf_asm_verification.csv \
	      $(EKF_ASM_CSV) \
	      insn.log

		  