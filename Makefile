# =============================================================================
#  Makefile  —  Kalman Filter Milestone-3 & Milestone-4
#  Targets:
#    make verify          compile matrix_asm.s + verify_matrix_asm.c, run tests
#    make lkf             compile lkf_asm.s  + lkf_verify.c,  run LKF §6 verification
#    make ekf             compile ekf_asm.s  + ekf_verify.c,  run EKF §6 verification
#    make ekf_verify_ref  run EKF with explicit Python reference comparison
#    make plots           generate all Milestone-3 plots via plot_milestone3.py
#    make all             run verify + lkf + ekf
#
#  Milestone-4 targets (vectorised / RVV):
#    make lkf_vector      compile + run vectorised LKF §7 verification
#    make ekf_vector      compile + run vectorised EKF §7 verification
#    make all_vec          run lkf_vector + ekf_vector
#
#    make clean           remove all generated files
#
#  §6 requirements (verified automatically by lkf/ekf targets):
#    |x_asm[k,i] - x_ref[k,i]| <= 1e-9  for every frame k and state component i
#    Tables A (avg |err| per joint) and B (avg |err| per component) printed to
#    stdout and saved to *_asm_verification.csv
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
CFLAGS_V = -march=$(ARCH_V) -mabi=$(ABI) -O0 -g -Wall
QEMU_V   = $(QEMU) -cpu rv64,v=true,vlen=128

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

# ── Milestone-4 vector sources ──
MATRIX_VEC_ASM    = matrix_vec.s
MATRIX_VEC_OBJ    = matrix_vec.o

EKF_UTILS_VEC_ASM = ekf_utils_vector.s
EKF_UTILS_VEC_OBJ = ekf_utils_vector.o

LKF_VEC_ASM       = lkf_vector.s
LKF_VEC_OBJ       = lkf_vector.o
LKF_VEC_VER_C     = verify_lkf_vector.c
LKF_VEC_VER_OBJ   = verify_lkf_vector.o
LKF_VEC_VER_ELF   = verify_lkf_vector

EKF_VEC_ASM       = ekf_vector.s
EKF_VEC_OBJ       = ekf_vector.o
EKF_VEC_VER_C     = verify_ekf_vector.c
EKF_VEC_VER_OBJ   = verify_ekf_vector.o
EKF_VEC_VER_ELF   = verify_ekf_vector

# ─────────────────────────────────────────────
# CSV file names
#
#  NOISY_CSV   — input: noisy measurements (required)
#  TRUE_CSV    — input: ground-truth measurements  (for plots)
#  LKF_REF_CSV — input: Python LKF reference output (from Milestone-2 / kalman-updated.py)
#  EKF_REF_CSV — input: Python EKF reference output (from kalman-updated.py)
#                       DO NOT overwrite — these are the references!
#  LKF_ASM_CSV — output: assembly LKF estimated state
#  EKF_ASM_CSV — output: assembly EKF estimated state
# ─────────────────────────────────────────────
NOISY_CSV   = 3D Full Body Humain Gait Walking Dataset (Noisy Values).csv
TRUE_CSV    = 3D Full Body Humain Gait Walking Dataset (True Values).csv
LKF_REF_CSV = lkf_results.csv
EKF_REF_CSV = ekf_results.csv
LKF_ASM_CSV = lkf_asm_results.csv
EKF_ASM_CSV = ekf_asm_results.csv

# Joint to focus on for plots (0=pelvis, default)
PLOT_JOINT  = 0

# ─────────────────────────────────────────────
# Phony targets
# ─────────────────────────────────────────────
.PHONY: all verify lkf ekf ekf_verify_ref plots clean \
        lkf_vector ekf_vector all_vec

all: verify lkf ekf
all_vec: lkf_vector ekf_vector

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
#
# §6: lkf_verify.c compares assembly output vs Python reference element-wise.
#     Prints Table A (avg |err| per joint) + Table B (avg |err| per component)
#     + global max/min error.  Saves lkf_asm_verification.csv.
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
	    echo "        Run: python3 kalman-updated.py \"$(NOISY_CSV)\" $(LKF_REF_CSV)"; \
	    exit 1; \
	fi
	$(QEMU) ./$(LKF_VER_ELF) "$(NOISY_CSV)" "$(LKF_REF_CSV)"

# ─────────────────────────────────────────────
# ekf_asm.o + ekf_verify_bin
#
# NOTE: ekf_asm.s calls state_init_F and state_init_Q defined in lkf_asm.s.
#       Therefore lkf_asm.o is included in the EKF link.
#
# §6: ekf_verify.c compares assembly output vs Python reference element-wise.
#     Prints Table A (avg |err| per joint) + Table B (avg |err| per component)
#     + global max/min error.  Saves ekf_asm_verification.csv.
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
	    echo "        Run: python3 kalman-updated.py \"$(NOISY_CSV)\" $(EKF_REF_CSV)"; \
	    exit 1; \
	fi
	$(QEMU) ./$(EKF_VER_ELF) "$(NOISY_CSV)" "$(EKF_REF_CSV)"

# Same as ekf but more explicit label — useful for CI / debugging.
ekf_verify_ref: $(EKF_VER_ELF)
	@echo ""
	@echo "=== Running EKF with Python reference comparison ==="
	@if [ ! -f "$(EKF_REF_CSV)" ]; then \
	    echo "[ERROR] EKF Python reference not found: $(EKF_REF_CSV)"; \
	    echo "        Run: python3 kalman-updated.py \"$(NOISY_CSV)\" $(EKF_REF_CSV)"; \
	    exit 1; \
	fi
	$(QEMU) ./$(EKF_VER_ELF) "$(NOISY_CSV)" "$(EKF_REF_CSV)"

# ─────────────────────────────────────────────
# plots  — generate all Milestone-3 figures
#
# Requires:
#   $(LKF_ASM_CSV)  — produced by 'make lkf'
#   $(EKF_ASM_CSV)  — produced by 'make ekf'
#   $(NOISY_CSV)    — input data
#   $(TRUE_CSV)     — optional ground truth
#   $(LKF_REF_CSV)  — optional M2 LKF reference (for comparison plots)
#   $(EKF_REF_CSV)  — optional M2 EKF reference (for comparison plots)
#
# Output: ./plots/*.png
# ─────────────────────────────────────────────
plots: $(LKF_ASM_CSV) $(EKF_ASM_CSV)
	@echo ""
	@echo "=== Generating Milestone-3 plots ==="
	@python3 plot_milestone3.py \
	    --noisy   "$(NOISY_CSV)"   \
	    --true    "$(TRUE_CSV)"    \
	    --lkf_m3  "$(LKF_ASM_CSV)" \
	    --ekf_m3  "$(EKF_ASM_CSV)" \
	    $(if $(wildcard $(LKF_REF_CSV)),--lkf_m2 "$(LKF_REF_CSV)",) \
	    $(if $(wildcard $(EKF_REF_CSV)),--ekf_m2 "$(EKF_REF_CSV)",) \
	    --joint   $(PLOT_JOINT)
	@echo "=== Plots written to ./plots/ ==="

# =============================================================================
#  Milestone-4  —  Vectorised LKF and EKF
# =============================================================================

# ─────────────────────────────────────────────
# Shared vector objects  (matrix_vec.o, ekf_utils_vector.o)
# ─────────────────────────────────────────────
$(MATRIX_VEC_OBJ): $(MATRIX_VEC_ASM)
	$(CC) $(CFLAGS_V) -c $< -o $@

$(EKF_UTILS_VEC_OBJ): $(EKF_UTILS_VEC_ASM)
	$(CC) $(CFLAGS_V) -c $< -o $@

# ─────────────────────────────────────────────
# lkf_vector  —  vectorised LKF build + §7 verification
#
# Link chain:  verify_lkf_vector.c
#              lkf_vector.s          (vector LKF logic)
#              ekf_utils_vector.s    (mat_joseph_update_vec)
#              matrix_vec.s          (vectorised matrix kernels)
#              matrix_asm.s          (mat_eye, mat_inverse_nxn — scalar)
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
	    echo "        Run: python3 kalman-updated.py \"$(NOISY_CSV)\" $(LKF_REF_CSV)"; \
	    exit 1; \
	fi
	$(QEMU_V) ./$(LKF_VEC_VER_ELF) "$(NOISY_CSV)" "$(LKF_REF_CSV)"

# ─────────────────────────────────────────────
# ekf_vector  —  vectorised EKF build + §7 verification
#
# Link chain:  verify_ekf_vector.c
#              ekf_vector.s          (vector EKF logic)
#              ekf_utils_vector.s    (mat_joseph_update_vec)
#              matrix_vec.s          (vectorised matrix kernels)
#              lkf_asm.s             (state_init_F, state_init_Q — shared)
#              matrix_asm.s          (mat_eye, mat_inverse_nxn, fast_atan2,
#                                     wrap_angle — scalar)
# ─────────────────────────────────────────────
$(EKF_VEC_OBJ): $(EKF_VEC_ASM)
	$(CC) $(CFLAGS_V) -c $< -o $@

$(EKF_VEC_VER_OBJ): $(EKF_VEC_VER_C)
	$(CC) $(CFLAGS_V) -c $< -o $@

$(EKF_VEC_VER_ELF): $(EKF_VEC_VER_OBJ) $(EKF_VEC_OBJ) $(EKF_UTILS_VEC_OBJ) \
                    $(MATRIX_VEC_OBJ) $(LKF_OBJ) $(MATRIX_OBJ)
	$(CC) $(CFLAGS_V) $^ -o $@ -lm -static

ekf_vector: $(EKF_VEC_VER_ELF)
	@echo ""
	@echo "=== Running EKF-VEC §7 verification (Milestone-4) ==="
	@if [ ! -f "$(EKF_REF_CSV)" ]; then \
	    echo "[ERROR] EKF Python reference not found: $(EKF_REF_CSV)"; \
	    echo "        Run: python3 kalman-updated.py \"$(NOISY_CSV)\" $(EKF_REF_CSV)"; \
	    exit 1; \
	fi
	$(QEMU_V) ./$(EKF_VEC_VER_ELF) "$(NOISY_CSV)" "$(EKF_REF_CSV)"


# ─────────────────────────────────────────────
# Cleanup
# ─────────────────────────────────────────────
clean:
	rm -f *.o \
	      $(VERIFY_ELF) \
	      $(LKF_VER_ELF) \
	      $(EKF_VER_ELF) \
	      $(LKF_VEC_VER_ELF) \
	      $(EKF_VEC_VER_ELF) \
	      $(LKF_ASM_CSV) \
	      lkf_asm_verification.csv \
	      $(EKF_ASM_CSV) \
	      ekf_asm_verification.csv \
	      lkf_vec_results.csv \
	      lkf_vec_verification.csv \
	      ekf_vec_results.csv \
	      ekf_vec_verification.csv \
	      insn.log
	rm -rf plots/
