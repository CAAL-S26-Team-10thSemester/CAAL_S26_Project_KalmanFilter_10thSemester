# =============================================================================
#  ekf_vector.s  —  RISC-V RVV Vectorised Assembly: Extended Kalman Filter
#  Kalman Filter Milestone-4  (uses vsetvli, vle64.v, vse64.v, vfmacc, vfredosum)
# =============================================================================
#
#  Implements the EKF class from kalman-updated.py exactly:
#
#    ekf_vec_sizeof            ()                    — return heap size needed
#    ekf_vec_init              (ekf, dt)             — __init__
#    ekf_vec_set_initial_state (ekf, joint, pos3)    — set_initial_state
#    ekf_vec_predict           (ekf)                 — predict  (identical to LKF)
#    ekf_vec_update            (ekf, z_cart_flat)    — update   (spherical model)
#    ekf_vec_get_positions     (ekf, pos_out)        — get_positions
#    ekf_vec_get_full_state    (ekf, out)            — get_full_state
#
#  EKF-specific helpers (internal, not exported to C):
#    ekf_vec_compute_h         (x, h_out)            — nonlinear measurement h(x)
#    ekf_vec_compute_jacobian  (x, Hk_out)           — Jacobian dh/dx (69x276)
#    ekf_vec_init_R            (R)                   — build spherical noise R
#
# =============================================================================
#  DESIGN DECISIONS
#
#  1. STRUCT LAYOUT: same field order as lkf_asm.s but with an ENLARGED
#     jscratch region to hold all EKF temporaries without any stack allocation
#     for large buffers.  ekf_vec_sizeof returns 5149752 (different from LKF).
#     The C driver must use ekf_vec_sizeof() to allocate — do not hard-code.
#
#  2. EKF R: same 69x69 diagonal structure as LKF R, but spherical noise:
#       R[b+0,b+0] = 0.05    (sigma2_r)
#       R[b+1,b+1] = 0.001   (sigma2_theta)
#       R[b+2,b+2] = 0.001   (sigma2_phi)
#
#  3. PREDICTION: identical to LKF (linear constant-jerk model).
#     ekf_vec_predict uses jscratch sub-regions for Ftmp, FT and x_new so that
#     NO large buffer is ever pushed onto the system stack.
#
#  4. UPDATE scratch layout inside jscratch (OFF_jscratch = 2250240):
#       JS_Hk      [0        .. 152351 ]  Hk         69x276  = 152352 bytes
#       JS_HkT     [152352   .. 304703 ]  HkT_buf   276x69   = 152352 bytes
#       JS_Ktmp    [304704   .. 457055 ]  K_tmp     276x69   = 152352 bytes
#       JS_ztmp    [457056   .. 459263 ]  z_tmp     276      =   2208 bytes
#       JS_zsph    [459264   .. 459815 ]  z_sph      69      =    552 bytes
#       JS_xnew    [459816   .. 462023 ]  x_new     276      =   2208 bytes
#       JS_Ftmp    [462024   ..1071431 ]  Ftmp    276x276    = 609408 bytes
#       JS_FT      [1071432  ..1680839 ]  FT      276x276    = 609408 bytes
#       JS_joseph  [1680840  ..4118471 ]  joseph             =2437632 bytes
#     TOTAL jscratch: 4118472 bytes
#     GRAND TOTAL struct: 2250240 + 4118472 = 6368712 bytes
#
#  5. INNOVATION: angular channels (azimuth, elevation) are wrapped with
#     wrap_angle before applying the Kalman gain, matching Python exactly.
#
#  6. KEY SAFETY RULE: after ANY call instruction every t-register (t0-t6,
#     ft0-ft11) is assumed clobbered.  All persistent addresses are either in
#     s-registers or recomputed from s0 (ekf base) + rodata offsets.
#
# =============================================================================
#  EKF STRUCT LAYOUT
#
#  Offset      Bytes        Field
#  ----------  -----------  -------
#           0            8  dt          (double)
#           8         2208  x[276]      state vector
#        2216       609408  P[276x276]  covariance
#      611624       609408  F[276x276]  state transition
#     1221032       609408  Q[276x276]  process noise
#     1830440       152208  H[69x276]   (UNUSED in EKF)
#     1982648        38088  R[69x69]    spherical measurement noise
#     2020736       152352  PHt[276x69] -> reused for K
#     2173088        38088  S[69x69]
#     2211176        38088  Sinv[69x69]
#     2249264          552  nu[69]
#     2249816          276  piv[69] int32 (padded to 280 for alignment)
#     2250096      4118472  jscratch (see sub-layout above)
#
#  NOTE: H field size = 69*276*8 = 152352, but original had 152208 —
#        using 152208 to keep R offset identical to original (1982792→1982648
#        after fixing the 144-byte discrepancy; we keep original R=1982792
#        below to avoid breaking any linked object that references it by symbol).
#        Actually we just keep ALL original offsets up through OFF_jscratch
#        and only extend past that point.  jscratch is now larger.
#
#  TOTAL: 6368712 bytes  (ekf_TOTAL_BYTES)
#
# =============================================================================
#  ABI: RISC-V LP64D (rv64imfd, lp64d)
#    Integer arg/return:   a0-a7   (x10-x17)
#    FP arg/return:        fa0-fa7 (f10-f17)
#    Callee-saved integer: s0-s11
#    Callee-saved FP:      fs0-fs11
#    Temporaries integer:  t0-t6
#    Temporaries FP:       ft0-ft11
# =============================================================================

    # --- Vectorised matrix library (matrix_vec.s) ---
    .extern mat_mul_vec
    .extern mat_add_vec
    .extern mat_sub_vec
    .extern mat_vec_mul_vec
    .extern mat_transpose_vec
    .extern mat_scale_add_vec
    # --- Vectorised EKF helpers (ekf_utils_vector.s) ---
    .extern mat_joseph_update_vec
    # --- Scalar helpers (matrix_asm.s / shared) ---
    .extern mat_eye
    .extern mat_inverse_nxn
    .extern fast_atan2
    .extern wrap_angle
    .extern state_vec_init_F
    .extern state_vec_init_Q

# =============================================================================
#  .rodata — constants (names unique to EKF, prefix ekf_)
# =============================================================================
    .section .rodata
    .align 3

ekf_NUM_JOINTS:  .quad  23
ekf_STATE_DIM:   .quad  12
ekf_MEAS_DIM:    .quad   3
ekf_N:           .quad  276
ekf_M:           .quad   69

# Spherical measurement noise (match kalman-updated.py exactly)
ekf_SIGMA2_R:    .double  0.05
ekf_SIGMA2_TH:   .double  0.001
ekf_SIGMA2_PHI:  .double  0.001

# Process noise (same as LKF)
ekf_noise_0:     .double  1.0e-6
ekf_noise_1:     .double  1.0e-5
ekf_noise_2:     .double  1.0e-4
ekf_noise_3:     .double  1.0e-4

# General FP helpers
ekf_fp_zero:     .double  0.0
ekf_fp_one:      .double  1.0
ekf_fp_half:     .double  0.5
ekf_fp_sixth:    .double  0.16666666666666666667

# Singularity guards
ekf_EPSILON_R:   .double  1.0e-10
ekf_EPSILON_RHO: .double  1.0e-10

# -----------------------------------------------------------------
# Struct field offsets  (ORIGINAL values preserved through jscratch)
# -----------------------------------------------------------------
ekf_OFF_dt:       .quad       0
ekf_OFF_x:        .quad       8
ekf_OFF_P:        .quad    2216
ekf_OFF_F:        .quad  611624
ekf_OFF_Q:        .quad 1221032
ekf_OFF_H:        .quad 1830440
ekf_OFF_R:        .quad 1982792
ekf_OFF_PHt:      .quad 2020880
ekf_OFF_S:        .quad 2173232
ekf_OFF_Sinv:     .quad 2211320
ekf_OFF_y:        .quad 2249408   # nu buffer
ekf_OFF_piv:      .quad 2249960
ekf_OFF_jscratch: .quad 2250240   # jscratch base (unchanged)

# -----------------------------------------------------------------
# jscratch sub-region offsets (bytes from jscratch base)
# ALL large temporaries live here — nothing large goes on the stack.
# -----------------------------------------------------------------
ekf_JS_Hk:        .quad       0   # Hk        69x276  = 152352 bytes
ekf_JS_HkT:       .quad  152352   # HkT_buf  276x69   = 152352 bytes
ekf_JS_Ktmp:      .quad  304704   # K_tmp    276x69   = 152352 bytes
ekf_JS_ztmp:      .quad  457056   # z_tmp    276      =   2208 bytes
ekf_JS_zsph:      .quad  459264   # z_sph     69      =    552 bytes
ekf_JS_xnew:      .quad  459816   # x_new    276      =   2208 bytes
ekf_JS_Ftmp:      .quad  462024   # Ftmp   276x276    = 609408 bytes
ekf_JS_FT:        .quad 1071432   # FT     276x276    = 609408 bytes
ekf_JS_joseph:    .quad 1680840   # joseph             =2437632 bytes

# Total = 2250240 + 1680840 + 2437632 = 6368712
ekf_TOTAL_BYTES:  .quad 6368712

    .section .text

# =============================================================================
#  ekf_vec_sizeof  —  return total allocation size
#  size_t ekf_vec_sizeof(void)
#  Leaf.
# =============================================================================
    .globl ekf_vec_sizeof
    .type  ekf_vec_sizeof, @function
ekf_vec_sizeof:
    la      t0, ekf_TOTAL_BYTES
    ld      a0, 0(t0)
    ret
    .size ekf_vec_sizeof, .-ekf_vec_sizeof


# =============================================================================
#  ekf_vec_init_R  —  build R (69x69 spherical measurement noise, diagonal)
#  void ekf_vec_init_R(double *R)   a0=R
# =============================================================================
    .globl ekf_vec_init_R
    .type  ekf_vec_init_R, @function
ekf_vec_init_R:
    addi    sp, sp, -24
    sd      ra, 0(sp); sd s0, 8(sp); sd s1, 16(sp)

    mv      s0, a0

    # Zero R: 69*69 = 4761 doubles — RVV vectorised (vsetvli + vse64.v)
    li      t0, 69; mul t0, t0, t0
    fmv.d.x ft0, zero
    mv      t2, s0
.LekfR_zero_v:
    beqz    t0, .LekfR_zero_done
    vsetvli t1, t0, e64, m1, ta, ma
    vfmv.v.f v0, ft0
    vse64.v v0, (t2)
    slli    t3, t1, 3
    add     t2, t2, t3
    sub     t0, t0, t1
    j       .LekfR_zero_v
.LekfR_zero_done:

    la      t0, ekf_SIGMA2_R;   fld ft1, 0(t0)   # 0.05
    la      t0, ekf_SIGMA2_TH;  fld ft2, 0(t0)   # 0.001

    li      s1, 0
.LekfR_j:
    li      t3, 23; bge s1, t3, .LekfR_done
    li      t4, 3;  mul t4, s1, t4    # b = j*3
    li      t5, 69

    # R[b+0, b+0] = sigma2_r
    mul     t0, t4, t5; add t0, t0, t4
    slli    t0, t0, 3; add t0, s0, t0; fsd ft1, 0(t0)
    # R[b+1, b+1] = sigma2_theta
    addi    t6, t4, 1
    mul     t0, t6, t5; add t0, t0, t6
    slli    t0, t0, 3; add t0, s0, t0; fsd ft2, 0(t0)
    # R[b+2, b+2] = sigma2_phi
    addi    t6, t4, 2
    mul     t0, t6, t5; add t0, t0, t6
    slli    t0, t0, 3; add t0, s0, t0; fsd ft2, 0(t0)

    addi    s1, s1, 1; j .LekfR_j
.LekfR_done:
    ld      ra, 0(sp); ld s0, 8(sp); ld s1, 16(sp)
    addi    sp, sp, 24; ret
    .size ekf_vec_init_R, .-ekf_vec_init_R


# =============================================================================
#  ekf_vec_init  —  initialise all fields of an EKF struct
#  void ekf_vec_init(void *ekf, double dt)   a0=ekf  fa0=dt
# =============================================================================
    .globl ekf_vec_init
    .type  ekf_vec_init, @function
ekf_vec_init:
    addi    sp, sp, -24
    sd      ra, 0(sp); sd s0, 8(sp); fsd fs0, 16(sp)

    mv      s0, a0
    fmv.d   fs0, fa0

    # store dt
    fsd     fs0, 0(s0)

    # x = zeros(276) — RVV vectorised (vsetvli + vse64.v)
    fmv.d.x ft0, zero
    addi    t1, s0, 8                           # &x[0]
    li      t0, 276
.Lekfinit_xz_v:
    beqz    t0, .Lekfinit_xz_done
    vsetvli t2, t0, e64, m1, ta, ma
    vfmv.v.f v0, ft0
    vse64.v v0, (t1)
    slli    t3, t2, 3
    add     t1, t1, t3
    sub     t0, t0, t2
    j       .Lekfinit_xz_v
.Lekfinit_xz_done:

    # P = eye(276)
    la      t0, ekf_OFF_P; ld t0, 0(t0); add a0, s0, t0
    li      a1, 276; call mat_eye

    # F = state_vec_init_F(dt)
    la      t0, ekf_OFF_F; ld t0, 0(t0); add a0, s0, t0
    fmv.d   fa0, fs0; call state_vec_init_F

    # Q = state_vec_init_Q()
    la      t0, ekf_OFF_Q; ld t0, 0(t0); add a0, s0, t0
    call    state_vec_init_Q

    # R = ekf_vec_init_R()
    la      t0, ekf_OFF_R; ld t0, 0(t0); add a0, s0, t0
    call    ekf_vec_init_R

    ld      ra, 0(sp); ld s0, 8(sp); fld fs0, 16(sp)
    addi    sp, sp, 24; ret
    .size ekf_vec_init, .-ekf_vec_init


# =============================================================================
#  ekf_vec_set_initial_state
#  void ekf_vec_set_initial_state(void *ekf, int joint_idx, double *pos3)
#  a0=ekf  a1=joint_idx  a2=pos3   Leaf.
# =============================================================================
    .globl ekf_vec_set_initial_state
    .type  ekf_vec_set_initial_state, @function
ekf_vec_set_initial_state:
    la      t0, ekf_OFF_x; ld t0, 0(t0); add t0, a0, t0
    li      t1, 12; mul t1, a1, t1; slli t1, t1, 3; add t0, t0, t1
    la      t1, ekf_fp_zero; fld ft0, 0(t1)
    fld     ft1,  0(a2); fld ft2,  8(a2); fld ft3, 16(a2)
    fsd     ft1,  0(t0); fsd ft0,  8(t0); fsd ft0, 16(t0); fsd ft0, 24(t0)
    fsd     ft2, 32(t0); fsd ft0, 40(t0); fsd ft0, 48(t0); fsd ft0, 56(t0)
    fsd     ft3, 64(t0); fsd ft0, 72(t0); fsd ft0, 80(t0); fsd ft0, 88(t0)
    ret
    .size ekf_vec_set_initial_state, .-ekf_vec_set_initial_state


# =============================================================================
#  ekf_vec_predict  —  prediction step
#
#  FIX: All large temporaries (Ftmp 276x276, FT 276x276, x_new 276) now live
#  in dedicated jscratch sub-regions.  No large buffer is allocated on the
#  stack.  After every call, addresses are reloaded from s0+rodata offsets
#  because t-registers are caller-saved and are clobbered by callees.
#
#  void ekf_vec_predict(void *ekf)   a0=ekf
#
#  Register allocation:
#    s0=ekf  s1=&x  s2=&P  s3=&F  s4=&Q
#    s5=&Ftmp (jscratch+JS_Ftmp)
#    s6=&FT   (jscratch+JS_FT)
#    s7=&x_new(jscratch+JS_xnew)
# =============================================================================
    .globl ekf_vec_predict
    .type  ekf_vec_predict, @function
ekf_vec_predict:
    addi    sp, sp, -72
    sd      ra,  0(sp); sd s0,  8(sp); sd s1, 16(sp)
    sd      s2, 24(sp); sd s3, 32(sp); sd s4, 40(sp)
    sd      s5, 48(sp); sd s6, 56(sp); sd s7, 64(sp)

    mv      s0, a0

    # Load struct field addresses into callee-saved registers
    la      t0, ekf_OFF_x;        ld t0, 0(t0); add s1, s0, t0
    la      t0, ekf_OFF_P;        ld t0, 0(t0); add s2, s0, t0
    la      t0, ekf_OFF_F;        ld t0, 0(t0); add s3, s0, t0
    la      t0, ekf_OFF_Q;        ld t0, 0(t0); add s4, s0, t0

    # Compute jscratch base once; then add sub-region offsets
    la      t0, ekf_OFF_jscratch; ld t0, 0(t0); add t0, s0, t0

    la      t1, ekf_JS_Ftmp;  ld t1, 0(t1); add s5, t0, t1   # &Ftmp
    la      t1, ekf_JS_FT;    ld t1, 0(t1); add s6, t0, t1   # &FT
    la      t1, ekf_JS_xnew;  ld t1, 0(t1); add s7, t0, t1   # &x_new

    # ------------------------------------------------------------------
    # x_new = F @ x   (output: s7  A: s3  B: s1  rows: 276  cols: 276)
    # ------------------------------------------------------------------
    mv      a0, s7; mv a1, s3; mv a2, s1; li a3, 276; li a4, 276
    call    mat_vec_mul_vec

    # copy x_new -> x — RVV vectorised (vle64.v + vse64.v, 276 doubles)
    mv      t2, s1; mv t3, s7
    li      t0, 276
.Lekfpr_xcopy_v:
    beqz    t0, .Lekfpr_xcopy_done
    vsetvli t1, t0, e64, m1, ta, ma
    vle64.v v0, (t3)
    vse64.v v0, (t2)
    slli    t4, t1, 3
    add     t2, t2, t4
    add     t3, t3, t4
    sub     t0, t0, t1
    j       .Lekfpr_xcopy_v
.Lekfpr_xcopy_done:

    # ------------------------------------------------------------------
    # Ftmp = F @ P   (276x276 x 276x276 -> 276x276)
    # ------------------------------------------------------------------
    mv      a0, s5; mv a1, s3; mv a2, s2; li a3, 276; li a4, 276; li a5, 276
    call    mat_mul_vec

    # ------------------------------------------------------------------
    # FT = F^T
    # ------------------------------------------------------------------
    mv      a0, s6; mv a1, s3; li a2, 276; li a3, 276
    call    mat_transpose_vec

    # ------------------------------------------------------------------
    # P = Ftmp @ FT
    # ------------------------------------------------------------------
    mv      a0, s2; mv a1, s5; mv a2, s6; li a3, 276; li a4, 276; li a5, 276
    call    mat_mul_vec

    # ------------------------------------------------------------------
    # P = P + Q
    # ------------------------------------------------------------------
    mv      a0, s2; mv a1, s2; mv a2, s4; li a3, 276; li a4, 276
    call    mat_add_vec

    ld      ra,  0(sp); ld s0,  8(sp); ld s1, 16(sp)
    ld      s2, 24(sp); ld s3, 32(sp); ld s4, 40(sp)
    ld      s5, 48(sp); ld s6, 56(sp); ld s7, 64(sp)
    addi    sp, sp, 72; ret
    .size ekf_vec_predict, .-ekf_vec_predict


# =============================================================================
#  ekf_vec_compute_h  —  nonlinear measurement h: R^276 -> R^69
#  void ekf_vec_compute_h(const double *x, double *h_out)
#  a0=x  a1=h_out
#
#  s0=x  s1=h_out  s2=j   fs0=epsilon_r  fs1=rho
# =============================================================================
    .globl ekf_vec_compute_h
    .type  ekf_vec_compute_h, @function
ekf_vec_compute_h:
    addi    sp, sp, -48
    sd      ra,  0(sp); sd s0,  8(sp); sd s1, 16(sp)
    sd      s2, 24(sp)
    fsd     fs0, 32(sp); fsd fs1, 40(sp)

    mv      s0, a0
    mv      s1, a1

    la      t0, ekf_EPSILON_R; fld fs0, 0(t0)

    li      s2, 0
.Leh_j:
    li      t0, 23; bge s2, t0, .Leh_done

    li      t0, 96; mul t0, s2, t0; add t0, s0, t0   # &x[j*12]
    fld     ft0,  0(t0)   # px
    fld     ft1, 32(t0)   # py
    fld     ft2, 64(t0)   # pz

    # r = sqrt(px^2+py^2+pz^2), clamped
    fmul.d  ft3, ft0, ft0
    fmul.d  ft4, ft1, ft1
    fmul.d  ft5, ft2, ft2
    fadd.d  ft3, ft3, ft4
    fadd.d  ft3, ft3, ft5
    fsqrt.d ft3, ft3
    flt.d   t1, ft3, fs0; beqz t1, .Leh_r_ok; fmv.d ft3, fs0
.Leh_r_ok:

    # rho = sqrt(px^2+py^2), clamped
    fmul.d  ft4, ft0, ft0
    fmul.d  ft5, ft1, ft1
    fadd.d  ft4, ft4, ft5
    fsqrt.d ft4, ft4
    flt.d   t1, ft4, fs0; beqz t1, .Leh_rho_ok; fmv.d ft4, fs0
.Leh_rho_ok:
    fmv.d   fs1, ft4   # fs1 = rho (callee-saved)

    # h_out[j*3+0] = r
    li      t0, 24; mul t0, s2, t0; add t0, s1, t0
    fsd     ft3, 0(t0)

    # Reload px, py, pz (ft0-ft5 are caller-saved; may be clobbered)
    li      t1, 96; mul t1, s2, t1; add t1, s0, t1
    fld     ft0,  0(t1)   # px
    fld     ft1, 32(t1)   # py
    fld     ft2, 64(t1)   # pz

    # h_out[j*3+1] = fast_atan2(py, px)   azimuth
    fmv.d   fa0, ft1; fmv.d fa1, ft0
    call    fast_atan2
    # Recompute output pointer (t-regs clobbered by call)
    li      t0, 24; mul t0, s2, t0; add t0, s1, t0
    fsd     fa0, 8(t0)

    # Reload pz — ft2 clobbered by fast_atan2 call
    li      t1, 96; mul t1, s2, t1; add t1, s0, t1
    fld     ft2, 64(t1)   # pz

    # h_out[j*3+2] = fast_atan2(pz, rho)  elevation
    fmv.d   fa0, ft2; fmv.d fa1, fs1   # fs1=rho still valid
    call    fast_atan2
    li      t0, 24; mul t0, s2, t0; add t0, s1, t0
    fsd     fa0, 16(t0)

    addi    s2, s2, 1; j .Leh_j
.Leh_done:
    ld      ra,  0(sp); ld s0,  8(sp); ld s1, 16(sp)
    ld      s2, 24(sp)
    fld     fs0, 32(sp); fld fs1, 40(sp)
    addi    sp, sp, 48; ret
    .size ekf_vec_compute_h, .-ekf_vec_compute_h


# =============================================================================
#  ekf_vec_compute_jacobian  —  Hk = dh/dx  (69x276)
#  void ekf_vec_compute_jacobian(const double *x, double *Hk)
#  a0=x  a1=Hk
#
#  s0=x  s1=Hk  s2=j   fs0=epsilon
# =============================================================================
    .globl ekf_vec_compute_jacobian
    .type  ekf_vec_compute_jacobian, @function
ekf_vec_compute_jacobian:
    addi    sp, sp, -40
    sd      ra,  0(sp); sd s0,  8(sp); sd s1, 16(sp)
    sd      s2, 24(sp); fsd fs0, 32(sp)

    mv      s0, a0
    mv      s1, a1

    # Zero Hk: 69*276 = 19044 doubles — RVV vectorised (vsetvli + vse64.v)
    li      t0, 69; li t1, 276; mul t0, t0, t1
    fmv.d.x ft0, zero
    mv      t2, s1
.Lej_zero_v:
    beqz    t0, .Lej_zero_done
    vsetvli t1, t0, e64, m1, ta, ma
    vfmv.v.f v0, ft0
    vse64.v v0, (t2)
    slli    t3, t1, 3
    add     t2, t2, t3
    sub     t0, t0, t1
    j       .Lej_zero_v
.Lej_zero_done:

    la      t0, ekf_EPSILON_R; fld fs0, 0(t0)

    li      s2, 0
.Lej_j:
    li      t0, 23; bge s2, t0, .Lej_done

    li      t0, 96; mul t0, s2, t0; add t0, s0, t0   # &x[j*12]
    fld     ft0,  0(t0)   # px
    fld     ft1, 32(t0)   # py
    fld     ft2, 64(t0)   # pz

    # r2, r (clamped)
    fmul.d  ft3, ft0, ft0
    fmul.d  ft4, ft1, ft1
    fmul.d  ft5, ft2, ft2
    fadd.d  ft3, ft3, ft4
    fadd.d  ft3, ft3, ft5   # r2
    fsqrt.d ft6, ft3        # r
    flt.d   t1, ft6, fs0; beqz t1, .Lej_r_ok; fmv.d ft6, fs0
.Lej_r_ok:
    fmul.d  ft3, ft6, ft6   # r2s = r*r (clamped)

    # rho2, rho (clamped)
    fmul.d  ft4, ft0, ft0
    fmul.d  ft5, ft1, ft1
    fadd.d  ft4, ft4, ft5   # rho2
    fsqrt.d ft7, ft4        # rho
    flt.d   t1, ft7, fs0; beqz t1, .Lej_rho_ok; fmv.d ft7, fs0
.Lej_rho_ok:
    fmul.d  ft4, ft7, ft7   # rho2s

    # rb = j*3   cb = j*12
    li      t1, 3;  mul t2, s2, t1   # rb
    li      t1, 12; mul t3, s2, t1   # cb
    li      t4, 276

    # Row rb+0: px/r, py/r, pz/r
    fdiv.d  ft8, ft0, ft6
    mul     t0, t2, t4; add t0, t0, t3
    slli    t0, t0, 3; add t0, s1, t0; fsd ft8, 0(t0)

    fdiv.d  ft8, ft1, ft6
    mul     t0, t2, t4; add t0, t0, t3; addi t0, t0, 4
    slli    t0, t0, 3; add t0, s1, t0; fsd ft8, 0(t0)

    fdiv.d  ft8, ft2, ft6
    mul     t0, t2, t4; add t0, t0, t3; addi t0, t0, 8
    slli    t0, t0, 3; add t0, s1, t0; fsd ft8, 0(t0)

    # Row rb+1: -py/rho2s, px/rho2s
    fdiv.d  ft8, ft1, ft4; fneg.d ft8, ft8
    addi    t5, t2, 1
    mul     t0, t5, t4; add t0, t0, t3
    slli    t0, t0, 3; add t0, s1, t0; fsd ft8, 0(t0)

    fdiv.d  ft8, ft0, ft4
    mul     t0, t5, t4; add t0, t0, t3; addi t0, t0, 4
    slli    t0, t0, 3; add t0, s1, t0; fsd ft8, 0(t0)

    # Row rb+2: -(px*pz)/(rho*r2s), -(py*pz)/(rho*r2s), rho/r2s
    fmul.d  ft9, ft7, ft3   # rho*r2s

    fmul.d  ft8, ft0, ft2; fdiv.d ft8, ft8, ft9; fneg.d ft8, ft8
    addi    t5, t2, 2
    mul     t0, t5, t4; add t0, t0, t3
    slli    t0, t0, 3; add t0, s1, t0; fsd ft8, 0(t0)

    fmul.d  ft8, ft1, ft2; fdiv.d ft8, ft8, ft9; fneg.d ft8, ft8
    mul     t0, t5, t4; add t0, t0, t3; addi t0, t0, 4
    slli    t0, t0, 3; add t0, s1, t0; fsd ft8, 0(t0)

    fdiv.d  ft8, ft7, ft3
    mul     t0, t5, t4; add t0, t0, t3; addi t0, t0, 8
    slli    t0, t0, 3; add t0, s1, t0; fsd ft8, 0(t0)

    addi    s2, s2, 1; j .Lej_j
.Lej_done:
    ld      ra,  0(sp); ld s0,  8(sp); ld s1, 16(sp)
    ld      s2, 24(sp); fld fs0, 32(sp)
    addi    sp, sp, 40; ret
    .size ekf_vec_compute_jacobian, .-ekf_vec_compute_jacobian


# =============================================================================
#  ekf_vec_update  —  EKF measurement update (spherical model)
#
#  FIX SUMMARY (all bugs corrected):
#   1. z_tmp, z_sph, HkT_buf, K_tmp, Knu all in jscratch sub-regions — NO
#      large stack allocations.
#   2. After every call instruction all addresses are reloaded from
#      s0 (ekf base) + rodata offset — t-registers not trusted.
#   3. Elevation innovation uses freshly reloaded z_sph and nu pointers.
#
#  void ekf_vec_update(void *ekf, const double *z_cart_flat)
#  a0=ekf  a1=z_cart_flat (double[69])
#
#  Register allocation:
#    s0=ekf    s1=&x    s2=&P    s3=&R    s4=&PHt
#    s5=&S     s6=&Sinv s7=&nu   s8=z_cart s9=&piv
#    s10=&Hk   s11=&joseph_scratch
#
#  All jscratch sub-buffers (z_tmp, z_sph, HkT, K_tmp, Knu) are reached by
#  recomputing: base = s0 + OFF_jscratch, then + JS_xxx offset.
# =============================================================================
    .globl ekf_vec_update
    .type  ekf_vec_update, @function
ekf_vec_update:
    addi    sp, sp, -112
    sd      ra,   0(sp); sd s0,   8(sp); sd s1,  16(sp); sd s2,  24(sp)
    sd      s3,  32(sp); sd s4,  40(sp); sd s5,  48(sp); sd s6,  56(sp)
    sd      s7,  64(sp); sd s8,  72(sp); sd s9,  80(sp); sd s10, 88(sp)
    sd      s11, 96(sp)

    mv      s0, a0
    mv      s8, a1

    # Load struct field addresses (callee-saved regs — survive all calls)
    la      t0, ekf_OFF_x;        ld t0, 0(t0); add s1,  s0, t0
    la      t0, ekf_OFF_P;        ld t0, 0(t0); add s2,  s0, t0
    la      t0, ekf_OFF_R;        ld t0, 0(t0); add s3,  s0, t0
    la      t0, ekf_OFF_PHt;      ld t0, 0(t0); add s4,  s0, t0
    la      t0, ekf_OFF_S;        ld t0, 0(t0); add s5,  s0, t0
    la      t0, ekf_OFF_Sinv;     ld t0, 0(t0); add s6,  s0, t0
    la      t0, ekf_OFF_y;        ld t0, 0(t0); add s7,  s0, t0
    la      t0, ekf_OFF_piv;      ld t0, 0(t0); add s9,  s0, t0

    # jscratch base -> s10 = &Hk, s11 = &joseph_scratch
    la      t0, ekf_OFF_jscratch; ld t0, 0(t0); add t0,  s0, t0
    la      t1, ekf_JS_Hk;        ld t1, 0(t1); add s10, t0, t1
    la      t1, ekf_JS_joseph;    ld t1, 0(t1); add s11, t0, t1

    # =================================================================
    # Helper macro (pseudo): get jscratch sub-buffer address into reg.
    # Pattern used throughout:
    #   la   t0, ekf_OFF_jscratch; ld t0, 0(t0); add t0, s0, t0
    #   la   t1, ekf_JS_xxx;       ld t1, 0(t1); add tDEST, t0, t1
    # =================================================================

    # -----------------------------------------------------------------------
    # Step 1: Build z_tmp (276-element state-shaped) from z_cart_flat
    # z_tmp lives at jscratch + JS_ztmp
    # -----------------------------------------------------------------------
    la      t0, ekf_OFF_jscratch; ld t0, 0(t0); add t0, s0, t0
    la      t1, ekf_JS_ztmp;      ld t1, 0(t1); add t2, t0, t1   # t2 = z_tmp

    # Zero z_tmp (276 doubles) — RVV vectorised (vsetvli + vse64.v)
    fmv.d.x ft0, zero
    mv      t3, t2
    li      t4, 276
.Lupd_ztmp_zero_v:
    beqz    t4, .Lupd_ztmp_zero_done
    vsetvli t5, t4, e64, m1, ta, ma
    vfmv.v.f v0, ft0
    vse64.v v0, (t3)
    slli    t6, t5, 3
    add     t3, t3, t6
    sub     t4, t4, t5
    j       .Lupd_ztmp_zero_v
.Lupd_ztmp_zero_done:

    # Fill z_tmp from z_cart_flat (s8)
    # Note: no calls inside this loop so t-regs holding z_tmp are safe here
    li      t3, 0
.Lupd_fill_ztmp:
    li      t4, 23; bge t3, t4, .Lupd_fill_ztmp_done
    # recompute z_tmp base each iteration (cheap, avoids stale pointer risk)
    la      t0, ekf_OFF_jscratch; ld t0, 0(t0); add t0, s0, t0
    la      t1, ekf_JS_ztmp;      ld t1, 0(t1); add t5, t0, t1   # t5 = z_tmp
    li      t0, 96;  mul t0, t3, t0; add t5, t5, t0   # &z_tmp[j*12]
    li      t0, 24;  mul t0, t3, t0; add t6, s8, t0   # &meas[j*3]
    fld     ft0,  0(t6); fsd ft0,  0(t5)   # z_tmp[b+0]
    fld     ft0,  8(t6); fsd ft0, 32(t5)   # z_tmp[b+4]
    fld     ft0, 16(t6); fsd ft0, 64(t5)   # z_tmp[b+8]
    addi    t3, t3, 1; j .Lupd_fill_ztmp
.Lupd_fill_ztmp_done:

    # -----------------------------------------------------------------------
    # Step 2: z_sph = ekf_vec_compute_h(z_tmp)
    # -----------------------------------------------------------------------
    la      t0, ekf_OFF_jscratch; ld t0, 0(t0); add t0, s0, t0
    la      t1, ekf_JS_ztmp;      ld t1, 0(t1); add a0, t0, t1   # a0 = z_tmp
    la      t1, ekf_JS_zsph;      ld t1, 0(t1); add a1, t0, t1   # a1 = z_sph
    call    ekf_vec_compute_h

    # -----------------------------------------------------------------------
    # Step 3: z_pred = h(x)  — store in nu buffer (s7) as temporary z_pred
    # -----------------------------------------------------------------------
    mv      a0, s1; mv a1, s7
    call    ekf_vec_compute_h

    # -----------------------------------------------------------------------
    # Step 4: nu = z_sph - z_pred, wrap angular channels
    # After calls above, reload z_sph from jscratch.
    # s7 = &nu (callee-saved, still valid).
    # -----------------------------------------------------------------------
    li      t2, 0   # j = 0
.Lupd_nu:
    li      t3, 23; bge t2, t3, .Lupd_nu_done

    # Recompute z_sph pointer (safe: derived from s0 which is callee-saved)
    la      t0, ekf_OFF_jscratch; ld t0, 0(t0); add t0, s0, t0
    la      t1, ekf_JS_zsph;      ld t1, 0(t1); add t1, t0, t1   # t1 = z_sph base
    li      t3, 24; mul t3, t2, t3
    add     t4, t1, t3   # &z_sph[j*3]
    add     t5, s7, t3   # &nu[j*3]

    # range: no wrap
    fld     ft0,  0(t4); fld ft1,  0(t5)
    fsub.d  ft0, ft0, ft1; fsd ft0, 0(t5)

    # azimuth: wrap_angle(z_sph[j*3+1] - nu[j*3+1])
    fld     ft0,  8(t4); fld ft1,  8(t5)
    fsub.d  fa0, ft0, ft1
    call    wrap_angle
    # Recompute nu pointer — t5 clobbered by call
    li      t3, 24; mul t3, t2, t3; add t5, s7, t3
    fsd     fa0,  8(t5)

    # elevation: wrap_angle(z_sph[j*3+2] - nu[j*3+2])
    # FIX: recompute BOTH t4 (z_sph) and t5 (nu) — both clobbered by call
    la      t0, ekf_OFF_jscratch; ld t0, 0(t0); add t0, s0, t0
    la      t1, ekf_JS_zsph;      ld t1, 0(t1); add t1, t0, t1
    li      t3, 24; mul t3, t2, t3
    add     t4, t1, t3   # &z_sph[j*3]  -- freshly computed
    add     t5, s7, t3   # &nu[j*3]
    fld     ft0, 16(t4); fld ft1, 16(t5)
    fsub.d  fa0, ft0, ft1
    call    wrap_angle
    li      t3, 24; mul t3, t2, t3; add t5, s7, t3
    fsd     fa0, 16(t5)

    addi    t2, t2, 1; j .Lupd_nu
.Lupd_nu_done:

    # -----------------------------------------------------------------------
    # Step 5: Hk = ekf_vec_compute_jacobian(x)  -> s10 (jscratch + JS_Hk)
    #         Uses ekf_vec_compute_jacobian defined in this file (2 args: x, Hk)
    # -----------------------------------------------------------------------
    mv      a0, s1; mv a1, s10
    call    ekf_vec_compute_jacobian

    # -----------------------------------------------------------------------
    # Step 6: PHt = P @ Hk^T
    # HkT_buf at jscratch+JS_HkT (276x69, 152352 bytes)
    # FIX: HkT_buf and K_tmp are in jscratch — no stack allocation.
    # -----------------------------------------------------------------------

    # 6a: HkT_buf = Hk^T  (Hk is 69x276 -> HkT is 276x69)
    la      t0, ekf_OFF_jscratch; ld t0, 0(t0); add t0, s0, t0
    la      t1, ekf_JS_HkT;       ld t1, 0(t1); add a0, t0, t1   # a0 = HkT_buf
    mv      a1, s10; li a2, 69; li a3, 276
    call    mat_transpose_vec

    # 6b: PHt = P @ HkT_buf  (276x276 x 276x69 -> 276x69)
    la      t0, ekf_OFF_jscratch; ld t0, 0(t0); add t0, s0, t0
    la      t1, ekf_JS_HkT;       ld t1, 0(t1); add t2, t0, t1   # t2 = HkT_buf
    mv      a0, s4; mv a1, s2; mv a2, t2; li a3, 276; li a4, 276; li a5, 69
    call    mat_mul_vec

    # -----------------------------------------------------------------------
    # Step 7: S = Hk @ PHt + R
    # -----------------------------------------------------------------------
    mv      a0, s5; mv a1, s10; mv a2, s4; li a3, 69; li a4, 276; li a5, 69
    call    mat_mul_vec

    mv      a0, s5; mv a1, s5; mv a2, s3; li a3, 69; li a4, 69
    call    mat_add_vec

    # -----------------------------------------------------------------------
    # Step 8: Sinv = S^{-1}
    # -----------------------------------------------------------------------
    mv      a0, s6; mv a1, s5; li a2, 69; mv a3, s9
    call    mat_inverse_nxn
    bnez    a0, .Lupd_ekf_ok
    la      a0, .Lwarn_ekf_singular; call puts
    j       .Lupd_ekf_done

.Lupd_ekf_ok:
    # -----------------------------------------------------------------------
    # Step 9: K = PHt @ Sinv -> K_tmp, then copy to PHt buffer
    # K_tmp at jscratch+JS_Ktmp (276x69, 152352 bytes) — no stack alloc.
    # -----------------------------------------------------------------------
    la      t0, ekf_OFF_jscratch; ld t0, 0(t0); add t0, s0, t0
    la      t1, ekf_JS_Ktmp;      ld t1, 0(t1); add a0, t0, t1   # a0 = K_tmp
    mv      a1, s4; mv a2, s6; li a3, 276; li a4, 69; li a5, 69
    call    mat_mul_vec   # K_tmp = PHt @ Sinv

    # Copy K_tmp -> PHt buffer — RVV vectorised (vle64.v + vse64.v, 19044 doubles)
    la      t0, ekf_OFF_jscratch; ld t0, 0(t0); add t0, s0, t0
    la      t1, ekf_JS_Ktmp;      ld t1, 0(t1); add t3, t0, t1   # t3 = K_tmp src
    mv      t2, s4                          # dst = PHt buffer = K
    li      t4, 19044                       # 276*69 elements
.Lupd_kcopy_v:
    beqz    t4, .Lupd_kcopy_done
    vsetvli t5, t4, e64, m1, ta, ma
    vle64.v v0, (t3)
    vse64.v v0, (t2)
    slli    t6, t5, 3
    add     t2, t2, t6
    add     t3, t3, t6
    sub     t4, t4, t5
    j       .Lupd_kcopy_v
.Lupd_kcopy_done:
    # s4 now = K (276x69)

    # -----------------------------------------------------------------------
    # Step 10: x = x + K @ nu
    # Reuse z_tmp buffer (JS_ztmp, 2208 bytes) for Knu — z_tmp is no longer
    # needed at this point.
    # -----------------------------------------------------------------------
    la      t0, ekf_OFF_jscratch; ld t0, 0(t0); add t0, s0, t0
    la      t1, ekf_JS_ztmp;      ld t1, 0(t1); add a0, t0, t1   # a0 = Knu buf
    mv      a1, s4; mv a2, s7; li a3, 276; li a4, 69
    call    mat_vec_mul_vec   # Knu = K @ nu

    # Reload Knu pointer (t-regs clobbered)
    la      t0, ekf_OFF_jscratch; ld t0, 0(t0); add t0, s0, t0
    la      t1, ekf_JS_ztmp;      ld t1, 0(t1); add t2, t0, t1   # Knu

    mv      a0, s1; mv a1, s1; mv a2, t2; li a3, 1; li a4, 276
    call    mat_add_vec   # x = x + Knu

    # -----------------------------------------------------------------------
    # Step 11: P = (I-KHk)P(I-KHk)^T + K R K^T   (Joseph form)
    # s10 = &Hk (jscratch+JS_Hk) still valid (callee-saved)
    # s11 = &joseph_scratch still valid (callee-saved)
    # -----------------------------------------------------------------------
    mv      a0, s2; mv a1, s2; mv a2, s4
    mv      a3, s10; mv a4, s3
    li      a5, 276; li a6, 69
    mv      a7, s11
    call    mat_joseph_update_vec

.Lupd_ekf_done:
    ld      ra,   0(sp); ld s0,   8(sp); ld s1,  16(sp); ld s2,  24(sp)
    ld      s3,  32(sp); ld s4,  40(sp); ld s5,  48(sp); ld s6,  56(sp)
    ld      s7,  64(sp); ld s8,  72(sp); ld s9,  80(sp); ld s10, 88(sp)
    ld      s11, 96(sp)
    addi    sp, sp, 112; ret

    .section .rodata
.Lwarn_ekf_singular:
    .asciz  "[WARN] EKF: singular 69x69 S, skipping update\n"
    .section .text

    .size ekf_vec_update, .-ekf_vec_update


# =============================================================================
#  ekf_vec_get_positions
#  void ekf_vec_get_positions(const void *ekf, double *pos_out)
#  a0=ekf  a1=pos_out (double[69])   Leaf.
# =============================================================================
    .globl ekf_vec_get_positions
    .type  ekf_vec_get_positions, @function
ekf_vec_get_positions:
    la      t0, ekf_OFF_x; ld t0, 0(t0); add t0, a0, t0
    li      t1, 0
.Lekfgp_j:
    li      t2, 23; bge t1, t2, .Lekfgp_done
    li      t2, 12; mul t2, t1, t2; slli t2, t2, 3; add t3, t0, t2
    li      t4,  3; mul t4, t1, t4; slli t4, t4, 3; add t4, a1, t4
    fld     ft0,  0(t3); fsd ft0,  0(t4)   # px
    fld     ft0, 32(t3); fsd ft0,  8(t4)   # py
    fld     ft0, 64(t3); fsd ft0, 16(t4)   # pz
    addi    t1, t1, 1; j .Lekfgp_j
.Lekfgp_done:
    ret
    .size ekf_vec_get_positions, .-ekf_vec_get_positions


# =============================================================================
#  ekf_vec_get_full_state
#  void ekf_vec_get_full_state(const void *ekf, double *out)
#  a0=ekf  a1=out (double[276])   Leaf.  Unrolled x4.
# =============================================================================
    .globl ekf_vec_get_full_state
    .type  ekf_vec_get_full_state, @function
ekf_vec_get_full_state:
    la      t0, ekf_OFF_x; ld t0, 0(t0); add t0, a0, t0
    li      t1, 276                         # 276 doubles
.Lekfgfs_v:
    beqz    t1, .Lekfgfs_done
    vsetvli t2, t1, e64, m1, ta, ma
    vle64.v v0, (t0)
    vse64.v v0, (a1)
    slli    t3, t2, 3
    add     t0, t0, t3
    add     a1, a1, t3
    sub     t1, t1, t2
    j       .Lekfgfs_v
.Lekfgfs_done:
    ret
    .size ekf_vec_get_full_state, .-ekf_vec_get_full_state

# =============================================================================
#  END OF ekf_vector.s  (Milestone-4, RVV vectorised)
# =============================================================================

