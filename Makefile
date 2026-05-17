# =============================================================================
#  Makefile  —  Kalman Filter Milestone-3 & Milestone-4
#
#  Directory layout:
#    ./                  M3 scalar sources, shared data CSVs, Dockerfile
#    ./milestone4/       M4 vector sources (.s, .c, .h)
#
#  Targets:
#    make verify          compile matrix_asm.s + verify_matrix_asm.c, run tests
#    make lkf             compile lkf_asm.s  + lkf_verify.c,  run LKF §6 verification
#    make ekf             compile ekf_asm.s  + ekf_verify.c,  run EKF §6 verification
#    make all             run verify + lkf + ekf
#
#  Milestone-4 targets (vectorised / RVV):
#    make lkf_vector      compile + run vectorised LKF §7 verification
#    make ekf_vector      compile + run vectorised EKF §7 verification
#    make all_vec          run lkf_vector + ekf_vector
#    make perf             run M3 vs M4 performance comparison
#    make insn_count       run instruction count profiling
#
#    make clean           remove all generated files
# =============================================================================

# ─────────────────────────────────────────────
# Toolchain
# ─────────────────────────────────────────────
TOOLCHAIN_PREFIX ?= riscv64-linux-gnu
CC     = $(TOOLCHAIN_PREFIX)-gcc
QEMU   = /usr/local/bin/qemu-riscv64

ARCH     = rv64imfd
ARCH_V   = rv64gcv
ABI      = lp64d
CFLAGS   = -march=$(ARCH) -mabi=$(ABI) -O0 -g -Wall
CFLAGS_V = -march=$(ARCH_V) -mabi=$(ABI) -O0 -g -Wall -I. -I$(M4DIR)
QEMU_V   = $(QEMU) -cpu rv64,v=true,vlen=128

# ─────────────────────────────────────────────
# Directories
# ─────────────────────────────────────────────
M4DIR = milestone4

# ─────────────────────────────────────────────
# Source / object / executable names — Milestone 3 (root)
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

# ── Milestone-4 vector sources (in milestone4/) ──
MATRIX_VEC_ASM    = $(M4DIR)/matrix_vec.s
MATRIX_VEC_OBJ    = $(M4DIR)/matrix_vec.o

EKF_UTILS_VEC_ASM = $(M4DIR)/ekf_utils_vector.s
EKF_UTILS_VEC_OBJ = $(M4DIR)/ekf_utils_vector.o

LKF_VEC_ASM       = $(M4DIR)/lkf_vector.s
LKF_VEC_OBJ       = $(M4DIR)/lkf_vector.o
LKF_VEC_VER_C     = $(M4DIR)/verify_lkf_vector.c
LKF_VEC_VER_OBJ   = $(M4DIR)/verify_lkf_vector.o
LKF_VEC_VER_ELF   = $(M4DIR)/verify_lkf_vector

EKF_VEC_ASM       = $(M4DIR)/ekf_vector.s
EKF_VEC_OBJ       = $(M4DIR)/ekf_vector.o
EKF_VEC_VER_C     = $(M4DIR)/verify_ekf_vector.c
EKF_VEC_VER_OBJ   = $(M4DIR)/verify_ekf_vector.o
EKF_VEC_VER_ELF   = $(M4DIR)/verify_ekf_vector

# ─────────────────────────────────────────────
# CSV file names (data in root, vec outputs in milestone4/)
# ─────────────────────────────────────────────
NOISY_CSV   = 3D Full Body Humain Gait Walking Dataset (Noisy Values).csv
TRUE_CSV    = 3D Full Body Humain Gait Walking Dataset (True Values).csv
LKF_REF_CSV = lkf_results.csv
EKF_REF_CSV = ekf_results.csv
LKF_ASM_CSV = lkf_asm_results.csv
EKF_ASM_CSV = ekf_asm_results.csv

# ─────────────────────────────────────────────
# Phony targets
# ─────────────────────────────────────────────
.PHONY: all verify lkf ekf clean \
        lkf_vector ekf_vector all_vec \
        perf insn_lkf_scalar insn_lkf_vec insn_ekf_scalar insn_ekf_vec insn_count verify_matrix_vec

all: verify lkf ekf
all_vec: lkf_vector ekf_vector

# ─────────────────────────────────────────────
# matrix_asm.o  — shared matrix library (root)
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
# lkf_asm.o + lkf_verify (M3 scalar)
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
	@if [ ! -f "$(LKF_REF_CSV)" ]; then \
	    echo "[ERROR] LKF Python reference not found: $(LKF_REF_CSV)"; \
	    exit 1; \
	fi
	$(QEMU) ./$(LKF_VER_ELF) "$(NOISY_CSV)" "$(LKF_REF_CSV)"

# ─────────────────────────────────────────────
# ekf_asm.o + ekf_verify_bin (M3 scalar)
# ─────────────────────────────────────────────
$(EKF_OBJ): $(EKF_ASM)
	$(CC) $(CFLAGS) -c $< -o $@

$(EKF_VER_OBJ): $(EKF_VER_C) $(MATRIX_HDR)
	$(CC) $(CFLAGS) -c $< -o $@

$(EKF_VER_ELF): $(EKF_VER_OBJ) $(EKF_OBJ) $(LKF_OBJ) $(MATRIX_OBJ)
	$(CC) $(CFLAGS) $^ -o $@ -lm -static

ekf: $(EKF_VER_ELF)
	@echo ""
	@echo "=== Running EKF §6 verification ==="
	@if [ ! -f "$(EKF_REF_CSV)" ]; then \
	    echo "[ERROR] EKF Python reference not found: $(EKF_REF_CSV)"; \
	    exit 1; \
	fi
	$(QEMU) ./$(EKF_VER_ELF) "$(NOISY_CSV)" "$(EKF_REF_CSV)"

# =============================================================================
#  Milestone-4  —  Vectorised LKF and EKF (sources in milestone4/)
# =============================================================================

# ─────────────────────────────────────────────
# Shared vector objects
# ─────────────────────────────────────────────
$(MATRIX_VEC_OBJ): $(MATRIX_VEC_ASM)
	$(CC) $(CFLAGS_V) -c $< -o $@

$(EKF_UTILS_VEC_OBJ): $(EKF_UTILS_VEC_ASM)
	$(CC) $(CFLAGS_V) -c $< -o $@

# ─────────────────────────────────────────────
# lkf_vector  —  vectorised LKF build + §7 verification
# ─────────────────────────────────────────────
$(LKF_VEC_OBJ): $(LKF_VEC_ASM)
	$(CC) $(CFLAGS_V) -c $< -o $@

$(LKF_VEC_VER_OBJ): $(LKF_VEC_VER_C)
	$(CC) $(CFLAGS_V) -c $< -o $@

$(LKF_VEC_VER_ELF): $(LKF_VEC_VER_OBJ) $(LKF_VEC_OBJ) $(EKF_UTILS_VEC_OBJ) \
                    $(MATRIX_VEC_OBJ) $(MATRIX_OBJ)
	$(CC) $(CFLAGS_V) $^ -o $@ -lm -static

lkf_vector: $(LKF_VEC_VER_ELF)
	@echo ""
	@echo "=== Running LKF-VEC §7 verification (Milestone-4) ==="
	@if [ ! -f "$(LKF_REF_CSV)" ]; then \
	    echo "[ERROR] LKF Python reference not found: $(LKF_REF_CSV)"; \
	    exit 1; \
	fi
	$(QEMU_V) ./$(LKF_VEC_VER_ELF) "$(NOISY_CSV)" "$(LKF_REF_CSV)"

# ─────────────────────────────────────────────
# ekf_vector  —  vectorised EKF build + §7 verification
# ─────────────────────────────────────────────
$(EKF_VEC_OBJ): $(EKF_VEC_ASM)
	$(CC) $(CFLAGS_V) -c $< -o $@

$(EKF_VEC_VER_OBJ): $(EKF_VEC_VER_C)
	$(CC) $(CFLAGS_V) -c $< -o $@

$(EKF_VEC_VER_ELF): $(EKF_VEC_VER_OBJ) $(EKF_VEC_OBJ) $(EKF_UTILS_VEC_OBJ) \
                    $(MATRIX_VEC_OBJ) $(LKF_VEC_OBJ) $(MATRIX_OBJ)
	$(CC) $(CFLAGS_V) $^ -o $@ -lm -static

ekf_vector: $(EKF_VEC_VER_ELF)
	@echo ""
	@echo "=== Running EKF-VEC §7 verification (Milestone-4) ==="
	@if [ ! -f "$(EKF_REF_CSV)" ]; then \
	    echo "[ERROR] EKF Python reference not found: $(EKF_REF_CSV)"; \
	    exit 1; \
	fi
	$(QEMU_V) ./$(EKF_VEC_VER_ELF) "$(NOISY_CSV)" "$(EKF_REF_CSV)"

# =============================================================================
#  Milestone-4  —  Performance Analysis (§8)
# =============================================================================

# ── Matrix vec unit tests ──
VERIFY_MAT_VEC_C   = $(M4DIR)/verify_matrix_vec.c
VERIFY_MAT_VEC_OBJ = $(M4DIR)/verify_matrix_vec.o
VERIFY_MAT_VEC_ELF = $(M4DIR)/verify_matrix_vec

$(VERIFY_MAT_VEC_OBJ): $(VERIFY_MAT_VEC_C)
	$(CC) $(CFLAGS_V) -c $< -o $@

$(VERIFY_MAT_VEC_ELF): $(VERIFY_MAT_VEC_OBJ) $(MATRIX_VEC_OBJ) $(MATRIX_OBJ)
	$(CC) $(CFLAGS_V) $^ -o $@ -lm -static

verify_matrix_vec: $(VERIFY_MAT_VEC_ELF)
	@echo ""
	@echo "=== Running matrix_vec numerical verification ==="
	$(QEMU_V) ./$(VERIFY_MAT_VEC_ELF)

# ── Performance comparison binary ──
PERF_C   = $(M4DIR)/perf_compare.c
PERF_OBJ = $(M4DIR)/perf_compare.o
PERF_ELF = $(M4DIR)/perf_compare

PERF_FRAMES ?= 10

$(PERF_OBJ): $(PERF_C)
	$(CC) $(CFLAGS_V) -c $< -o $@

$(PERF_ELF): $(PERF_OBJ) $(LKF_VEC_OBJ) $(EKF_VEC_OBJ) $(EKF_UTILS_VEC_OBJ) \
             $(MATRIX_VEC_OBJ) $(LKF_OBJ) $(EKF_OBJ) $(MATRIX_OBJ)
	$(CC) $(CFLAGS_V) $^ -o $@ -lm -static

perf: $(PERF_ELF)
	@echo ""
	@echo "=== Milestone-4 Performance Analysis (§8) ==="
	$(QEMU_V) ./$(PERF_ELF) "$(NOISY_CSV)" $(PERF_FRAMES)

# ── Instruction-count profiling via QEMU plugin ──
QEMU_PLUGIN = /usr/local/lib/qemu/plugins/libinsn.so

insn_lkf_scalar: $(LKF_VER_ELF)
	@echo "=== Counting instructions: LKF-M3 (scalar) ==="
	$(QEMU) -plugin $(QEMU_PLUGIN) -d plugin -D insn_lkf_scalar.log \
	    ./$(LKF_VER_ELF) "$(NOISY_CSV)" "$(LKF_REF_CSV)" 2>/dev/null || true
	@tail -5 insn_lkf_scalar.log

insn_lkf_vec: $(LKF_VEC_VER_ELF)
	@echo "=== Counting instructions: LKF-M4 (vector) ==="
	$(QEMU_V) -plugin $(QEMU_PLUGIN) -d plugin -D insn_lkf_vec.log \
	    ./$(LKF_VEC_VER_ELF) "$(NOISY_CSV)" "$(LKF_REF_CSV)" 2>/dev/null || true
	@tail -5 insn_lkf_vec.log

insn_ekf_scalar: $(EKF_VER_ELF)
	@echo "=== Counting instructions: EKF-M3 (scalar) ==="
	$(QEMU) -plugin $(QEMU_PLUGIN) -d plugin -D insn_ekf_scalar.log \
	    ./$(EKF_VER_ELF) "$(NOISY_CSV)" "$(EKF_REF_CSV)" 2>/dev/null || true
	@tail -5 insn_ekf_scalar.log

insn_ekf_vec: $(EKF_VEC_VER_ELF)
	@echo "=== Counting instructions: EKF-M4 (vector) ==="
	$(QEMU_V) -plugin $(QEMU_PLUGIN) -d plugin -D insn_ekf_vec.log \
	    ./$(EKF_VEC_VER_ELF) "$(NOISY_CSV)" "$(EKF_REF_CSV)" 2>/dev/null || true
	@tail -5 insn_ekf_vec.log

insn_count: insn_lkf_scalar insn_lkf_vec insn_ekf_scalar insn_ekf_vec
	@echo ""
	@echo "=== Instruction Count Summary ==="
	@echo "LKF scalar:"; grep -i 'total' insn_lkf_scalar.log 2>/dev/null || echo "  (check insn_lkf_scalar.log)"
	@echo "LKF vector:"; grep -i 'total' insn_lkf_vec.log 2>/dev/null || echo "  (check insn_lkf_vec.log)"
	@echo "EKF scalar:"; grep -i 'total' insn_ekf_scalar.log 2>/dev/null || echo "  (check insn_ekf_scalar.log)"
	@echo "EKF vector:"; grep -i 'total' insn_ekf_vec.log 2>/dev/null || echo "  (check insn_ekf_vec.log)"

# ─────────────────────────────────────────────
# Cleanup
# ─────────────────────────────────────────────
clean:
	rm -f *.o \
	      $(VERIFY_ELF) \
	      $(LKF_VER_ELF) \
	      $(EKF_VER_ELF) \
	      $(LKF_ASM_CSV) \
	      lkf_asm_verification.csv \
	      $(EKF_ASM_CSV) \
	      ekf_asm_verification.csv \
	      insn.log insn_lkf_scalar.log insn_lkf_vec.log \
	      insn_ekf_scalar.log insn_ekf_vec.log
	rm -f $(M4DIR)/*.o \
	      $(LKF_VEC_VER_ELF) \
	      $(EKF_VEC_VER_ELF) \
	      $(VERIFY_MAT_VEC_ELF) \
	      $(PERF_ELF) \
	      $(M4DIR)/lkf_vec_results.csv \
	      $(M4DIR)/lkf_vec_verification.csv \
	      $(M4DIR)/ekf_vec_results.csv \
	      $(M4DIR)/ekf_vec_verification.csv \
	      $(M4DIR)/perf_analysis.csv
	rm -rf plots/
