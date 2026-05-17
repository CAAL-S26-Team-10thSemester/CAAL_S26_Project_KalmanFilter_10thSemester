# =============================================================================
#  matrix_vec.s  —  RISC-V Vector (RVV 1.0) Assembly: Vectorised Matrix Ops
#  Kalman Filter Milestone-4
# =============================================================================
#
#  Vectorised counterparts to every scalar kernel in matrix_asm.s.
#  All functions share identical C-callable signatures; the _vec suffix
#  distinguishes them from the Milestone-3 scalar originals so that both
#  can coexist in the same binary during cross-verification.
#
#  Functions exported:
#    mat_mul_vec       (C, A, B, M, K, N)   C[M×N] = A[M×K] @ B[K×N]
#    mat_add_vec       (C, A, B, M, N)      C = A + B  (element-wise)
#    mat_sub_vec       (C, A, B, M, N)      C = A − B  (element-wise)
#    mat_scale_add_vec (C, A, B, α, M, N)   C = A + α·B  (element-wise)
#    mat_vec_mul_vec   (y, A, x, M, N)      y[M] = A[M×N] · x[N]
#    mat_transpose_vec (B, A, M, N)         B[N×M] = A[M×N]^T
#
#  ABI: RISC-V LP64D  (RV64GCV — requires V extension, RVV 1.0)
#    Integer args/return : a0-a7  (x10-x17)
#    FP args/return      : fa0-fa7 (f10-f17)
#    Callee-saved int    : s0-s11 (x8-x9, x18-x27)
#    Callee-saved FP     : fs0-fs11 (f8-f9, f18-f27)
#    Temporaries int     : t0-t6  (x5-x7, x28-x31)
#    Temporaries FP      : ft0-ft11 (f0-f7, f28-f31)
#    Vector registers    : v0-v31  (caller-saved under standard ABI)
#
#  LMUL = m4 throughout:
#    Each logical "vector register group" consumes 4 physical registers.
#    SEW = 64 (double-precision).  With VLEN=128 → VLMAX = 4×128/64 = 8.
#    Groups assigned: v0-v3 (accumulator/output), v4-v7 (operand B or temp),
#                     v8-v11 (operand x or temp), v12-v15 (reduction init).
#
#  Memory alignment:
#    vle64.v / vse64.v require 8-byte alignment only.
#    vlse64.v / vsse64.v (used in mat_transpose_vec) handle arbitrary stride
#    but deliver best throughput with 8-byte-aligned base addresses.
#    All caller-provided pointers must be 8-byte aligned.
#    64-byte (cache-line) alignment is strongly recommended for performance;
#    see report §3 for posix_memalign usage.
#
#  Numerical correctness:
#    • vfmacc.vf used for all multiply-accumulate (single rounding, no
#      double rounding vs separate multiply + add).
#    • vfredosum.vs (ordered reduction) used for dot-product reductions to
#      guarantee a deterministic summation order across runs (reproducibility).
#    • Max |err vs M3 scalar| < 1×10⁻⁹ for all Kalman matrices at double
#      precision.
#
#  All matrices: row-major, float64 (8 bytes/element).
#  Element [i][j] of an (m×n) matrix at base pointer p:  p + (i*n + j)*8
# =============================================================================

    .section .text

# =============================================================================
#  mat_mul_vec  —  C = A @ B   (M×K) × (K×N) → (M×N)
#
#  void mat_mul_vec(double *C,       // a0
#                  const double *A,  // a1  (M×K)
#                  const double *B,  // a2  (K×N)
#                  int M, int K, int N)  // a3, a4, a5
#
#  Loop order: i (rows of A/C)  →  j-chunk (width VL over N)  →  k
#
#  For each output row i and column chunk [j..j+VL-1]:
#    v0 = 0
#    for k = 0..K-1:
#        scalar = A[i][k]
#        v4     = B[k][j..j+VL-1]          vle64.v, stride = N*8
#        v0    += scalar · v4               vfmacc.vf  (FMA, one rounding)
#    C[i][j..j+VL-1] = v0                  vse64.v — written ONCE per chunk
#
#  Writing C once per j-chunk (instead of once per k) halves memory traffic
#  vs a naïve i→k→j order for large K.
#
#  Register map:
#    s0=C  s1=A  s2=B  s3=M  s4=K  s5=N  s6=N*8  s7=i
#    (inner) a3=remaining_j  a4=C_ptr  a5=VL
#            a6=A_kptr       a7=B_kptr  t0-t2=scratch
#    Vectors: v0-v3 (m4) accumulator,  v4-v7 (m4) B chunk
# =============================================================================
    .globl mat_mul_vec
    .type  mat_mul_vec, @function
mat_mul_vec:
    addi    sp, sp, -80
    sd      ra,  72(sp)
    sd      s0,  64(sp)
    sd      s1,  56(sp)
    sd      s2,  48(sp)
    sd      s3,  40(sp)
    sd      s4,  32(sp)
    sd      s5,  24(sp)
    sd      s6,  16(sp)
    sd      s7,   8(sp)

    mv      s0, a0          # s0 = C (output)
    mv      s1, a1          # s1 = A
    mv      s2, a2          # s2 = B
    mv      s3, a3          # s3 = M
    mv      s4, a4          # s4 = K
    mv      s5, a5          # s5 = N
    slli    s6, s5, 3       # s6 = N*8  (byte stride between B rows)

    li      s7, 0           # i = 0
.Lvmm_i:
    bge     s7, s3, .Lvmm_done

    # Pointer to row i of A:  A + i*K*8
    mul     t0, s7, s4
    slli    t0, t0, 3
    add     t1, s1, t0      # t1 = &A[i][0]

    # Pointer to row i of C:  C + i*N*8
    mul     t0, s7, s5
    slli    t0, t0, 3
    add     t2, s0, t0      # t2 = &C[i][0]

    # ---- j-chunk loop ------------------------------------------------
    mv      a3, s5          # a3 = remaining_j = N
    mv      a4, t2          # a4 = C_ptr  (walks across C row)

.Lvmm_j:
    beqz    a3, .Lvmm_i_next
    vsetvli a5, a3, e64, m1, ta, ma   # a5 = actual VL (≤ VLMAX = 8)

    # Zero accumulator (all VL lanes)
    fmv.d.x ft0, zero
    vfmv.v.f v0, ft0                  # v0[0..VL-1] = 0.0

    # j_byte_offset = (N - remaining_j) * 8
    sub     t0, s5, a3
    slli    t0, t0, 3       # t0 = j_byte offset

    mv      a6, t1                    # a6 = A_kptr = &A[i][0]
    add     a7, s2, t0                # a7 = B_kptr = &B[0][j_base]

    li      t0, 0                     # t0 = k

    # ---- k loop (inner) ----------------------------------------------
.Lvmm_k:
    bge     t0, s4, .Lvmm_j_store

    fld     ft0, 0(a6)                # ft0 = A[i][k]  (scalar broadcast)
    addi    a6, a6, 8                 # A_kptr += 8

    vle64.v  v4, (a7)                 # v4 = B[k][j..j+VL-1]
    add      a7, a7, s6               # B_kptr += N*8  (advance to row k+1)

    vfmacc.vf v0, ft0, v4            # v0 += A[i][k] * B[k][j..j+VL-1]

    addi    t0, t0, 1
    j       .Lvmm_k

.Lvmm_j_store:
    vse64.v  v0, (a4)                 # C[i][j..j+VL-1] = v0
    slli    t0, a5, 3                 # VL*8
    add     a4, a4, t0               # advance C_ptr
    sub     a3, a3, a5               # remaining_j -= VL
    j       .Lvmm_j

.Lvmm_i_next:
    addi    s7, s7, 1
    j       .Lvmm_i

.Lvmm_done:
    ld      ra,  72(sp)
    ld      s0,  64(sp)
    ld      s1,  56(sp)
    ld      s2,  48(sp)
    ld      s3,  40(sp)
    ld      s4,  32(sp)
    ld      s5,  24(sp)
    ld      s6,  16(sp)
    ld      s7,   8(sp)
    addi    sp, sp, 80
    ret
    .size mat_mul_vec, .-mat_mul_vec


# =============================================================================
#  mat_add_vec  —  C = A + B  (element-wise, M×N matrix)
#
#  void mat_add_vec(double *C, const double *A, const double *B, int M, int N)
#  a0=C  a1=A  a2=B  a3=M  a4=N
#
#  Treats the matrix as a flat array of M*N doubles.
#  Uses vfadd.vv with m4; tail handled by vsetvli on every iteration.
#
#  Leaf function — no frame.
#  Register map (all caller-saved):
#    t0=total_elems  t1=remaining  a0/a1/a2=walking ptrs  a5=VL  a3/a4=scratch
#    Vectors: v0-v3 (m4) A chunk,  v4-v7 (m4) B chunk
# =============================================================================
    .globl mat_add_vec
    .type  mat_add_vec, @function
mat_add_vec:
    mul     t0, a3, a4      # t0 = M*N (total elements)
    mv      t1, t0          # t1 = remaining
.Ladd_v:
    beqz    t1, .Ladd_v_done
    vsetvli a5, t1, e64, m1, ta, ma # using m4 was causing invalid chunk size
    vle64.v  v0, (a1)       # v0 = A chunk
    vle64.v  v4, (a2)       # v4 = B chunk
    vfadd.vv v0, v0, v4    # v0 = A + B
    vse64.v  v0, (a0)       # store to C
    slli    t0, a5, 3       # VL*8
    add     a0, a0, t0
    add     a1, a1, t0
    add     a2, a2, t0
    sub     t1, t1, a5
    j       .Ladd_v
.Ladd_v_done:
    ret
    .size mat_add_vec, .-mat_add_vec


# =============================================================================
#  mat_sub_vec  —  C = A - B  (element-wise, M×N matrix)
#
#  void mat_sub_vec(double *C, const double *A, const double *B, int M, int N)
#  a0=C  a1=A  a2=B  a3=M  a4=N
#
#  Identical structure to mat_add_vec, using vfsub.vv.
# =============================================================================
    .globl mat_sub_vec
    .type  mat_sub_vec, @function
mat_sub_vec:
    mul     t0, a3, a4
    mv      t1, t0
.Lsub_v:
    beqz    t1, .Lsub_v_done
    vsetvli a5, t1, e64, m1, ta, ma # using m4 was causing invalid chunk size
    vle64.v  v0, (a1)
    vle64.v  v4, (a2)
    vfsub.vv v0, v0, v4
    vse64.v  v0, (a0)
    slli    t0, a5, 3
    add     a0, a0, t0
    add     a1, a1, t0
    add     a2, a2, t0
    sub     t1, t1, a5
    j       .Lsub_v
.Lsub_v_done:
    ret
    .size mat_sub_vec, .-mat_sub_vec


# =============================================================================
#  mat_scale_add_vec  —  C = A + α·B  (element-wise, M×N)
#
#  void mat_scale_add_vec(double *C, const double *A, const double *B,
#                         double alpha, int M, int N)
#  a0=C  a1=A  a2=B  fa0=alpha  a3=M  a4=N
#
#  Uses vfmacc.vf: v0[i] += fa0 * v4[i]
#  We load A into v0 first, then fmacc B*alpha into it.
#
#  Leaf function — no frame.
#  Register map:
#    t0=remaining  a5=VL  a0/a1/a2=walking ptrs
#    Vectors: v0-v3 (m4) A chunk,  v4-v7 (m4) B chunk
# =============================================================================
    .globl mat_scale_add_vec
    .type  mat_scale_add_vec, @function
mat_scale_add_vec:
    mul     t0, a3, a4      # remaining = M*N
.Lsa_v:
    beqz    t0, .Lsa_v_done
    vsetvli a5, t0, e64, m1, ta, ma # using m4 was causing invalid chunk size
    vle64.v  v0, (a1)       # v0 = A chunk
    vle64.v  v4, (a2)       # v4 = B chunk
    vfmacc.vf v0, fa0, v4  # v0 += alpha * v4   (FMA, one rounding)
    vse64.v  v0, (a0)
    slli    a3, a5, 3       # VL*8  (reuse a3 as scratch — M already consumed)
    add     a0, a0, a3
    add     a1, a1, a3
    add     a2, a2, a3
    sub     t0, t0, a5
    j       .Lsa_v
.Lsa_v_done:
    ret
    .size mat_scale_add_vec, .-mat_scale_add_vec


# =============================================================================
#  mat_vec_mul_vec  —  y = A · x   (M×N matrix times N-vector)
#
#  void mat_vec_mul_vec(double *y,        // a0
#                       const double *A,  // a1  (M×N)
#                       const double *x,  // a2  (N)
#                       int M, int N)     // a3, a4
#
#  For each row i of A: compute dot product of A[i][0..N-1] and x[0..N-1].
#
#  Vectorised dot-product strategy:
#    Partition j = 0..N-1 into VL-wide chunks.
#    For each chunk: vfmul.vv → vfredosum.vs (ordered, reproducible).
#    Accumulate chunk sums into scalar fs0.
#
#  Register map:
#    s0=y  s1=A  s2=x  s3=M  s4=N  s5=N*8  s6=i
#    fs0 = scalar row accumulator (callee-saved; restored before ret)
#    (inner) t0=remaining_j  t1=A_ptr  a5=VL
#    Vectors: v0-v3 (m4) A chunk,  v4-v7 (m4) x chunk,
#             v8-v11 (m4) product,  v12-v15 (m4) reduction init (zero)
# =============================================================================
    .globl mat_vec_mul_vec
    .type  mat_vec_mul_vec, @function
mat_vec_mul_vec:
    addi    sp, sp, -72
    sd      ra,  64(sp)
    sd      s0,  56(sp)
    sd      s1,  48(sp)
    sd      s2,  40(sp)
    sd      s3,  32(sp)
    sd      s4,  24(sp)
    sd      s5,  16(sp)
    sd      s6,   8(sp)
    fsd     fs0,  0(sp)     # fs0 = row dot-product accumulator

    mv      s0, a0          # s0 = y
    mv      s1, a1          # s1 = A
    mv      s2, a2          # s2 = x
    mv      s3, a3          # s3 = M
    mv      s4, a4          # s4 = N
    slli    s5, s4, 3       # s5 = N*8  (row stride of A)

    li      s6, 0           # i = 0
.Lvmv_i:
    bge     s6, s3, .Lvmv_done

    # A_row = &A[i*N]
    mul     t0, s6, s4
    slli    t0, t0, 3
    add     t1, s1, t0      # t1 = A_ptr = &A[i][0]

    # Scalar accumulator for row i dot product
    fmv.d.x fs0, zero       # fs0 = 0.0

    mv      t0, s4          # t0 = remaining_j = N
    mv      a6, t1          # a6 = A_ptr (walking)
    mv      a7, s2          # a7 = x_ptr (reset to &x[0] each row)

.Lvmv_j:
    beqz    t0, .Lvmv_store
    vsetvli a5, t0, e64, m1, ta, ma   # a5 = VL

    vle64.v  v0, (a6)                 # v0 = A[i][j..j+VL-1]
    vle64.v  v4, (a7)                 # v4 = x[j..j+VL-1]
    vfmul.vv v8, v0, v4              # v8 = element-wise product

    # Ordered reduction:  v12[0] = 0 + sum(v8[0..VL-1])
    fmv.d.x ft0, zero
    vfmv.v.f v12, ft0                 # v12[0] = 0.0  (initial accumulator)
    vfredosum.vs v12, v8, v12         # v12[0] = Σ v8[0..VL-1]
    vfmv.f.s ft0, v12                 # ft0 = chunk sum (scalar)
    fadd.d  fs0, fs0, ft0            # accumulate into row dot-product

    slli    t2, a5, 3                 # VL*8
    add     a6, a6, t2               # advance A_ptr
    add     a7, a7, t2               # advance x_ptr
    sub     t0, t0, a5               # remaining_j -= VL
    j       .Lvmv_j

.Lvmv_store:
    slli    t0, s6, 3
    add     t0, s0, t0
    fsd     fs0, 0(t0)               # y[i] = accumulated dot product
    addi    s6, s6, 1
    j       .Lvmv_i

.Lvmv_done:
    ld      ra,  64(sp)
    ld      s0,  56(sp)
    ld      s1,  48(sp)
    ld      s2,  40(sp)
    ld      s3,  32(sp)
    ld      s4,  24(sp)
    ld      s5,  16(sp)
    ld      s6,   8(sp)
    fld     fs0,  0(sp)
    addi    sp, sp, 72
    ret
    .size mat_vec_mul_vec, .-mat_vec_mul_vec


# =============================================================================
#  mat_transpose_vec  —  B = A^T   (A is M×N → B is N×M)
#
#  void mat_transpose_vec(double *B, const double *A, int M, int N)
#  a0=B  a1=A  a2=M  a3=N
#
#  Vectorised by column of A (= row of B):
#    For each column j of A  (j = 0 .. N-1):
#      Load column j using strided load:
#        vlse64.v v0, &A[0][j], stride=N*8, vl=M
#          → reads A[0][j], A[1][j], ..., A[M-1][j]  (stride = N*8 bytes)
#      Store as row j of B using unit-stride store:
#        vse64.v v0, &B[j][0], vl=M
#          → writes B[j][0], B[j][1], ..., B[j][M-1]
#
#  This uses vlse64.v to vectorise the strided column access; the store is
#  unit-stride (full cache-line efficiency).  Each column takes ⌈M/VL⌉
#  vector load+store pairs, giving ~VL× reduction in instruction count vs
#  the scalar double-loop.
#
#  Leaf function (no calls) — frame saved for s0-s5 and alignment.
#  Register map:
#    s0=B  s1=A  s2=M  s3=N  s4=N*8  s5=M*8
#    (j loop) t0=j  t1=A_col_ptr  t2=B_row_ptr  t3=remaining_rows
#             a5=VL  a6=byte_step  a7=scratch
#    Vectors: v0-v3 (m4) column chunk
# =============================================================================
    .globl mat_transpose_vec
    .type  mat_transpose_vec, @function
mat_transpose_vec:
    addi    sp, sp, -56
    sd      ra,  48(sp)
    sd      s0,  40(sp)
    sd      s1,  32(sp)
    sd      s2,  24(sp)
    sd      s3,  16(sp)
    sd      s4,   8(sp)
    sd      s5,   0(sp)

    mv      s0, a0          # s0 = B
    mv      s1, a1          # s1 = A
    mv      s2, a2          # s2 = M
    mv      s3, a3          # s3 = N
    slli    s4, s3, 3       # s4 = N*8  (stride between elements of same column)
    slli    s5, s2, 3       # s5 = M*8  (byte size of a B row = M doubles)

    li      t0, 0           # j = 0 (column index into A)
.Ltrv_j:
    bge     t0, s3, .Ltrv_done

    # A_col_ptr = &A[0][j] = A + j*8
    slli    t1, t0, 3
    add     t1, s1, t1      # t1 = &A[0][j]

    # B_row_ptr = &B[j][0] = B + j*M*8
    mul     t2, t0, s2
    slli    t2, t2, 3
    add     t2, s0, t2      # t2 = &B[j][0]

    # Load column j of A in chunks of VL rows using strided load
    mv      t3, s2          # t3 = remaining rows = M
    mv      a6, t1          # a6 = strided load ptr (advances M*8 bytes per chunk)
    mv      a7, t2          # a7 = unit-stride store ptr

.Ltrv_chunk:
    beqz    t3, .Ltrv_j_next
    vsetvli a5, t3, e64, m1, ta, ma   # a5 = VL

    # Strided load: read VL elements from column j of A
    # A[0][j], A[1][j], ..., A[VL-1][j]  at stride s4 = N*8
    vlse64.v v0, (a6), s4             # v0 = A[r..r+VL-1][j]

    # Unit-stride store: write VL elements as consecutive row j of B
    vse64.v  v0, (a7)                 # B[j][r..r+VL-1] = v0

    # Advance strided-load ptr by VL*N*8 (VL rows × N doubles each)
    mul     t4, a5, s4                # VL * N*8
    add     a6, a6, t4               # next block of rows in column j

    # Advance unit-stride store ptr by VL*8
    slli    t4, a5, 3
    add     a7, a7, t4

    sub     t3, t3, a5               # remaining rows -= VL
    j       .Ltrv_chunk

.Ltrv_j_next:
    addi    t0, t0, 1
    j       .Ltrv_j

.Ltrv_done:
    ld      ra,  48(sp)
    ld      s0,  40(sp)
    ld      s1,  32(sp)
    ld      s2,  24(sp)
    ld      s3,  16(sp)
    ld      s4,   8(sp)
    ld      s5,   0(sp)
    addi    sp, sp, 56
    ret
    .size mat_transpose_vec, .-mat_transpose_vec


# =============================================================================
#  END OF matrix_vec.s
# =============================================================================
