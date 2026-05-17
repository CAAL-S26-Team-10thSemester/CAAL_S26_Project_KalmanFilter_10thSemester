# =============================================================================
#  ekf_utils_vector.s  —  RISC-V RVV Vectorised Assembly: Joseph-form Update
#  Kalman Filter Milestone-4
# =============================================================================
#
#  Contains the vectorised Joseph-form covariance update used by both
#  lkf_vector.s and ekf_vector.s.  Identical algorithm to the scalar
#  mat_joseph_update in matrix_asm.s, but delegates all matrix operations
#  to the RVV-vectorised kernels in matrix_vec.s.
#
#  Functions exported:
#    mat_joseph_update_vec(Pout, P, K, H, R, n, m, scratch)
#
#  Dependencies (from matrix_vec.s):
#    mat_mul_vec, mat_sub_vec, mat_add_vec, mat_transpose_vec
#  Dependencies (from matrix_asm.s):
#    mat_eye  (scalar; init only — not performance-critical)
#
#  ABI: RISC-V LP64D + V  (rv64gcv, lp64d)
# =============================================================================

    .extern mat_mul_vec
    .extern mat_sub_vec
    .extern mat_add_vec
    .extern mat_transpose_vec
    .extern mat_eye

    .section .text

# =============================================================================
#  mat_joseph_update_vec  —  P = (I - K*H) * P * (I - K*H)^T  +  K * R * K^T
#
#  void mat_joseph_update_vec(double *Pout,       // a0
#                             const double *P,    // a1  (n×n)
#                             const double *K,    // a2  (n×m)
#                             const double *H,    // a3  (m×n)
#                             const double *R,    // a4  (m×m)
#                             int n,              // a5
#                             int m,              // a6
#                             double *scratch)    // a7
#
#  scratch layout (same as scalar mat_joseph_update):
#    [0        .. n*n-1  ]  IKH   (n×n)
#    [n*n      .. 2*n*n-1]  tmp1  (n×n)
#    [2*n*n    .. 3*n*n-1]  tmp2  (n×n)
#    [3*n*n    .. 4*n*n-1]  KT    (m×n, allocated as n×n worst-case)
#  Caller MUST provide >= 4*n*n doubles of scratch.
#
#  Register allocation:
#    s0  = Pout          s1  = P            s2  = K
#    s3  = H             s4  = R            s5  = n
#    s6  = m             s7  = scratch base
#    s8  = &IKH          s9  = &tmp1        s10 = &tmp2
#    s11 = &KT
# =============================================================================
    .globl mat_joseph_update_vec
    .type  mat_joseph_update_vec, @function
mat_joseph_update_vec:
    addi    sp, sp, -104
    sd      ra,   0(sp)
    sd      s0,   8(sp)
    sd      s1,  16(sp)
    sd      s2,  24(sp)
    sd      s3,  32(sp)
    sd      s4,  40(sp)
    sd      s5,  48(sp)
    sd      s6,  56(sp)
    sd      s7,  64(sp)
    sd      s8,  72(sp)
    sd      s9,  80(sp)
    sd      s10, 88(sp)
    sd      s11, 96(sp)

    # Preserve arguments in callee-saved registers
    mv      s0, a0          # Pout
    mv      s1, a1          # P
    mv      s2, a2          # K
    mv      s3, a3          # H
    mv      s4, a4          # R
    mv      s5, a5          # n
    mv      s6, a6          # m
    mv      s7, a7          # scratch

    # Compute scratch sub-region pointers
    mul     t0, s5, s5
    slli    t0, t0, 3       # nn8 = n*n*8  (byte size of an n×n matrix)

    mv      s8,  s7         # IKH  = scratch[0        .. n*n-1  ]
    add     s9,  s7, t0     # tmp1 = scratch[n*n      .. 2*n*n-1]
    add     s10, s9, t0     # tmp2 = scratch[2*n*n    .. 3*n*n-1]
    add     s11, s10, t0    # KT   = scratch[3*n*n    .. 4*n*n-1]

    # -----------------------------------------------------------------
    # Step 1:  IKH = I_n - K @ H
    # -----------------------------------------------------------------

    # 1a: IKH = I_n  (scalar mat_eye — not perf-critical, runs once)
    mv      a0, s8
    mv      a1, s5
    call    mat_eye

    # 1b: tmp1 = K @ H   (n×m) × (m×n) → (n×n)
    mv      a0, s9          # dst = tmp1
    mv      a1, s2          # A   = K
    mv      a2, s3          # B   = H
    mv      a3, s5          # M   = n
    mv      a4, s6          # K   = m
    mv      a5, s5          # N   = n
    call    mat_mul_vec

    # 1c: IKH = IKH - tmp1   (I - K@H, element-wise n×n)
    mv      a0, s8          # dst = IKH
    mv      a1, s8          # A   = IKH (= I_n)
    mv      a2, s9          # B   = tmp1 (= K@H)
    mv      a3, s5          # M   = n
    mv      a4, s5          # N   = n
    call    mat_sub_vec

    # -----------------------------------------------------------------
    # Step 2:  tmp1 = IKH @ P    (n×n) × (n×n) → (n×n)
    # -----------------------------------------------------------------
    mv      a0, s9          # dst = tmp1
    mv      a1, s8          # A   = IKH
    mv      a2, s1          # B   = P
    mv      a3, s5          # M   = n
    mv      a4, s5          # K   = n
    mv      a5, s5          # N   = n
    call    mat_mul_vec

    # -----------------------------------------------------------------
    # Step 3:  Pout = (IKH @ P) @ IKH^T
    # -----------------------------------------------------------------

    # 3a: tmp2 = IKH^T   (n×n → n×n)
    mv      a0, s10         # dst = tmp2
    mv      a1, s8          # src = IKH
    mv      a2, s5          # M   = n
    mv      a3, s5          # N   = n
    call    mat_transpose_vec

    # 3b: Pout = tmp1 @ tmp2   (n×n) × (n×n) → (n×n)
    mv      a0, s0          # dst = Pout
    mv      a1, s9          # A   = tmp1 (= IKH@P)
    mv      a2, s10         # B   = tmp2 (= IKH^T)
    mv      a3, s5          # M   = n
    mv      a4, s5          # K   = n
    mv      a5, s5          # N   = n
    call    mat_mul_vec

    # -----------------------------------------------------------------
    # Step 4:  tmp1 = K @ R      (n×m) × (m×m) → (n×m)
    # -----------------------------------------------------------------
    mv      a0, s9          # dst = tmp1
    mv      a1, s2          # A   = K
    mv      a2, s4          # B   = R
    mv      a3, s5          # M   = n
    mv      a4, s6          # K   = m
    mv      a5, s6          # N   = m
    call    mat_mul_vec

    # -----------------------------------------------------------------
    # Step 5:  KT = K^T          (n×m → m×n)
    # -----------------------------------------------------------------
    mv      a0, s11         # dst = KT
    mv      a1, s2          # src = K
    mv      a2, s5          # M   = n
    mv      a3, s6          # N   = m
    call    mat_transpose_vec

    # -----------------------------------------------------------------
    # Step 6:  tmp2 = (K @ R) @ K^T   (n×m) × (m×n) → (n×n)
    # -----------------------------------------------------------------
    mv      a0, s10         # dst = tmp2
    mv      a1, s9          # A   = tmp1 (= K@R)
    mv      a2, s11         # B   = KT
    mv      a3, s5          # M   = n
    mv      a4, s6          # K   = m
    mv      a5, s5          # N   = n
    call    mat_mul_vec

    # -----------------------------------------------------------------
    # Step 7:  Pout = Pout + tmp2   (element-wise n×n)
    # -----------------------------------------------------------------
    mv      a0, s0          # dst = Pout
    mv      a1, s0          # A   = Pout
    mv      a2, s10         # B   = tmp2 (= K@R@K^T)
    mv      a3, s5          # M   = n
    mv      a4, s5          # N   = n
    call    mat_add_vec

    # -----------------------------------------------------------------
    # Epilogue — restore callee-saved registers and return
    # -----------------------------------------------------------------
    ld      ra,   0(sp)
    ld      s0,   8(sp)
    ld      s1,  16(sp)
    ld      s2,  24(sp)
    ld      s3,  32(sp)
    ld      s4,  40(sp)
    ld      s5,  48(sp)
    ld      s6,  56(sp)
    ld      s7,  64(sp)
    ld      s8,  72(sp)
    ld      s9,  80(sp)
    ld      s10, 88(sp)
    ld      s11, 96(sp)
    addi    sp, sp, 104
    ret
    .size mat_joseph_update_vec, .-mat_joseph_update_vec


# =============================================================================
#  END OF ekf_utils_vector.s
# =============================================================================
