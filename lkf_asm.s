# =============================================================================
#  lkf_asm.s  —  RISC-V Scalar Assembly: Linear Kalman Filter
#  Kalman Filter Milestone-3
# =============================================================================
#
#  Implements the LKF class from kalman-updated.py exactly:
#
#    lkf_init          (dt)                    — __init__
#    lkf_set_initial_state (lkf, joint, pos)   — set_initial_state
#    lkf_predict       (lkf)                   — predict
#    lkf_update        (lkf, z_flat)           — update
#    lkf_get_positions (lkf, pos_out)          — get_positions
#    lkf_get_full_state(lkf, out)              — get_full_state
#
#  Also exposes the matrix-init helpers used only at startup:
#    state_init_F      (F, dt)
#    state_init_Q      (Q)
#    meas_init_H       (H)
#    meas_init_R       (R)
#
#  All matrix operations delegate to matrix_asm.s via .extern / call.
#
# =============================================================================
#  LKF STATE STRUCT  (C layout, all fields 8-byte aligned)
#  The struct is heap-allocated by the caller (C driver or lkf runner).
#
#  Offset  Size    Field
#  ------  ------  -------
#       0     8    dt          (double)
#       8  2208    x[276]      (double[TOTAL_STATE_DIM])   state vector
#    2216  609408  P[276×276]  (double[276*276])            covariance
#  611624  609408  F[276×276]  (double[276*276])            state transition
# 1221032  609408  Q[276×276]  (double[276*276])            process noise
# 1830440  152208  H[69×276]   (double[69*276])             measurement
# 1982648   38088  R[69×69]    (double[69*69])              meas noise
# 2020736  (end)
#
#  Scratch buffers appended after struct (caller allocates total):
# 2020736  152208  PHt[276×69]  temp for P@H^T
# 2172944   38088  S[69×69]     innovation covariance
# 2211032   38088  Sinv[69×69]  S inverse
# 2249120     552  y[69]        innovation vector
# 2249672  609408  Pjoseph_scratch[4*276*276]  Joseph update scratch
# (total ≈ 5.0 MB per LKF instance)
#
#  Constants (matching kalman-updated.py exactly):
# =============================================================================

    .section .rodata
    .align 3

# --- Dimensions ---------------------------------------------------------------
lkf_NUM_JOINTS:    .quad  23
lkf_STATE_DIM:     .quad  12
lkf_MEAS_DIM:      .quad   3
lkf_N:             .quad  276    # TOTAL_STATE_DIM
lkf_M:             .quad   69    # TOTAL_MEAS_DIM

# --- Noise constants (match kalman-updated.py exactly) -----------------------
lkf_EST_R_PX:      .double  0.29472279
lkf_EST_R_PY:      .double  0.09632091
lkf_EST_R_PZ:      .double  0.00204269

# --- Noise map: noise_map = {0:1e-6, 1:1e-5, 2:1e-4, 3:1e-4} ---------------
lkf_noise_0:       .double  1e-6
lkf_noise_1:       .double  1e-5
lkf_noise_2:       .double  1e-4
lkf_noise_3:       .double  1e-4

# --- FP helpers ---------------------------------------------------------------
lkf_fp_zero:       .double  0.0
lkf_fp_one:        .double  1.0
lkf_fp_half:       .double  0.5
lkf_fp_sixth:      .double  0.16666666666666666667   # 1/6

# --- Struct byte offsets (must match the layout table above) -----------------
#  dt   = 0
#  x    = 8                          (276 * 8 = 2208 bytes)
#  P    = 8  + 2208  = 2216          (276*276*8 = 609408 bytes)
#  F    = 2216 + 609408 = 611624
#  Q    = 611624 + 609408 = 1221032
#  H    = 1221032 + 609408 = 1830440  (69*276*8 = 152208 bytes)
#  R    = 1830440 + 152208 = 1982648  (69*69*8 = 38088 bytes)
# Scratch (appended, caller allocates):
#  PHt  = 1982648 + 38088 = 2020736  (276*69*8 = 152208 bytes)
#  S    = 2020736 + 152208 = 2172944 (69*69*8 = 38088 bytes)
#  Sinv = 2172944 + 38088 = 2211032
#  y    = 2211032 + 38088 = 2249120  (69*8 = 552 bytes)
#  piv  = 2249120 + 552   = 2249672  (69*4 = 276 bytes, int32)
#  joseph_scratch = 2249672+280 rounded to 8 = 2249952  (4*276*276*8 bytes)

lkf_OFF_dt:        .quad       0
lkf_OFF_x:         .quad       8
lkf_OFF_P:         .quad    2216
lkf_OFF_F:         .quad  611624
lkf_OFF_Q:         .quad 1221032
lkf_OFF_H:         .quad 1830440
lkf_OFF_R:         .quad 1982792
lkf_OFF_PHt:       .quad 2020880
lkf_OFF_S:         .quad 2173232
lkf_OFF_Sinv:      .quad 2211320
lkf_OFF_y:         .quad 2249408
lkf_OFF_piv:       .quad 2249960
lkf_OFF_jscratch:  .quad 2250240   # 4*276*276*8 = 2437632 bytes
lkf_TOTAL_BYTES:   .quad 4687872   # total allocation needed

    .section .text

# Declare all matrix_asm.s functions used here
    .extern mat_eye
    .extern mat_mul
    .extern mat_add
    .extern mat_sub
    .extern mat_transpose
    .extern mat_vec_mul
    .extern mat_inverse_nxn
    .extern mat_joseph_update

# =============================================================================
#  lkf_sizeof  —  return total byte size needed for one LKF instance
#
#  size_t lkf_sizeof(void)
#  Returns: a0 = 4687584
# =============================================================================
    .globl lkf_sizeof
    .type  lkf_sizeof, @function
lkf_sizeof:
    la      t0, lkf_TOTAL_BYTES
    ld      a0, 0(t0)
    ret
    .size lkf_sizeof, .-lkf_sizeof


# =============================================================================
#  lkf_field  —  internal macro-like helper: load field ptr into a reg
#  Usage (inline in other functions):
#    la t0, lkf_OFF_xxx ; ld t0, 0(t0) ; add <reg>, <lkf>, t0
# =============================================================================

# =============================================================================
#  state_init_F  —  build F (276×276 block-diagonal state transition)
#
#  Python:
#    F = eye(276)
#    dt2 = 0.5 * dt * dt
#    dt3 = (1/6) * dt^3
#    for j in 0..22:
#      base = j*12
#      for axis in 0..2:
#        r = base + axis*4
#        F[r,r+1]=dt; F[r,r+2]=dt2; F[r,r+3]=dt3
#        F[r+1,r+2]=dt; F[r+1,r+3]=dt2
#        F[r+2,r+3]=dt
#
#  void state_init_F(double *F, double dt)
#  a0=F, fa0=dt
#
#  Register map:
#    s0=F  fs0=dt  fs1=dt2  fs2=dt3
#    s1=j(joint)  s2=axis  s3=r  s4=N=276
# =============================================================================
    .globl state_init_F
    .type  state_init_F, @function
state_init_F:
    addi    sp, sp, -56
    sd      ra,  0(sp); sd s0,  8(sp); sd s1, 16(sp)
    sd      s2, 24(sp); sd s3, 32(sp); sd s4, 40(sp)
    fsd     fs0, 48(sp)

    mv      s0, a0              # F base ptr
    fmv.d   fs0, fa0            # dt

    # F = eye(276)
    li      a1, 276
    call    mat_eye             # mat_eye(F, 276) — a0=s0 already set above
    # (a0 was s0 before the call; mat_eye clobbers a0 — restore from s0)

    # dt2 = 0.5 * dt * dt
    la      t0, lkf_fp_half
    fld     ft0, 0(t0)
    fmul.d  fs1, fs0, fs0       # dt*dt
    fmul.d  fs1, ft0, fs1       # 0.5 * dt^2

    # dt3 = (1/6) * dt^3
    la      t0, lkf_fp_sixth
    fld     ft0, 0(t0)
    fmul.d  fs2, fs1, fs0       # 0.5*dt^2 * dt  (= dt^3/2)
    fadd.d  fs2, fs2, fs2       # dt^3
    fmul.d  fs2, ft0, fs2       # (1/6)*dt^3

    li      s4, 276             # N
    li      s1, 0               # j = 0
.LF_j:
    li      t0, 23
    bge     s1, t0, .LF_done

    # base = j * 12
    li      t1, 12
    mul     t2, s1, t1          # base = j*12

    li      s2, 0               # axis = 0
.LF_axis:
    li      t0, 3
    bge     s2, t0, .LF_axis_done

    # r = base + axis*4
    slli    t0, s2, 2
    add     s3, t2, t0          # r = base + axis*4

    # --- F[r, r+1] = dt  (offset = (r*276 + r+1)*8) ---
    # helper: write ft to F[row][col]
    # offset = (row * 276 + col) * 8
    # F[r,r+1]
    mul     t0, s3, s4; add t0, t0, s3; addi t0, t0, 1
    slli    t0, t0, 3; add t0, s0, t0
    fsd     fs0, 0(t0)

    # F[r, r+2] = dt2
    mul     t0, s3, s4; add t0, t0, s3; addi t0, t0, 2
    slli    t0, t0, 3; add t0, s0, t0
    fsd     fs1, 0(t0)

    # F[r, r+3] = dt3
    mul     t0, s3, s4; add t0, t0, s3; addi t0, t0, 3
    slli    t0, t0, 3; add t0, s0, t0
    fsd     fs2, 0(t0)

    # F[r+1, r+2] = dt
    addi    t1, s3, 1           # row = r+1
    mul     t0, t1, s4; add t0, t0, s3; addi t0, t0, 2
    slli    t0, t0, 3; add t0, s0, t0
    fsd     fs0, 0(t0)

    # F[r+1, r+3] = dt2
    mul     t0, t1, s4; add t0, t0, s3; addi t0, t0, 3
    slli    t0, t0, 3; add t0, s0, t0
    fsd     fs1, 0(t0)

    # F[r+2, r+3] = dt
    addi    t1, s3, 2           # row = r+2
    mul     t0, t1, s4; add t0, t0, s3; addi t0, t0, 3
    slli    t0, t0, 3; add t0, s0, t0
    fsd     fs0, 0(t0)

    addi    s2, s2, 1; j .LF_axis
.LF_axis_done:
    addi    s1, s1, 1; j .LF_j
.LF_done:
    ld      ra,  0(sp); ld s0,  8(sp); ld s1, 16(sp)
    ld      s2, 24(sp); ld s3, 32(sp); ld s4, 40(sp)
    fld     fs0, 48(sp)
    addi    sp, sp, 56; ret
    .size state_init_F, .-state_init_F


# =============================================================================
#  state_init_Q  —  build Q (276×276 block-diagonal process noise)
#
#  Python:
#    Q = zeros(276,276)
#    noise_map = {0:1e-6, 1:1e-5, 2:1e-4, 3:1e-4}
#    for j in 0..22:
#      base = j*12
#      for i in 0..11:
#        Q[base+i, base+i] = noise_map[i % 4]
#
#  void state_init_Q(double *Q)
#  a0=Q
# =============================================================================
    .globl state_init_Q
    .type  state_init_Q, @function
state_init_Q:
    addi    sp, sp, -32
    sd      ra, 0(sp); sd s0, 8(sp); sd s1, 16(sp); sd s2, 24(sp)

    mv      s0, a0              # Q base

    # zero Q: 276*276*8 bytes
    li      t0, 276; mul t0, t0, t0; slli t0, t0, 3   # byte count
    add     t1, s0, t0          # end ptr
    la      t2, lkf_fp_zero; fld ft0, 0(t2)
    mv      t2, s0
.LQ_zero:
    bge     t2, t1, .LQ_zero_done
    fsd     ft0, 0(t2); addi t2, t2, 8; j .LQ_zero
.LQ_zero_done:

    # Load noise values
    la      t0, lkf_noise_0; fld ft0, 0(t0)   # 1e-6
    la      t0, lkf_noise_1; fld ft1, 0(t0)   # 1e-5
    la      t0, lkf_noise_2; fld ft2, 0(t0)   # 1e-4 (also noise_3)

    li      s1, 0               # j = 0
.LQ_j:
    li      t3, 23
    bge     s1, t3, .LQ_done
    li      t4, 12; mul t4, s1, t4   # base = j*12

    li      s2, 0               # i = 0
.LQ_i:
    li      t3, 12
    bge     s2, t3, .LQ_i_done

    # Q[base+i, base+i] = noise_map[i%4]
    add     t5, t4, s2          # row = col = base+i
    li      t6, 276
    mul     t0, t5, t6; add t0, t0, t5
    slli    t0, t0, 3; add t0, s0, t0    # &Q[base+i][base+i]

    # select noise based on i%4
    andi    t3, s2, 3           # i % 4
    beqz    t3, .LQ_n0
    li      t6, 1; beq t3, t6, .LQ_n1
    fsd     ft2, 0(t0)          # i%4 == 2 or 3: 1e-4
    j       .LQ_next
.LQ_n0: fsd ft0, 0(t0); j .LQ_next
.LQ_n1: fsd ft1, 0(t0)
.LQ_next:
    addi    s2, s2, 1; j .LQ_i
.LQ_i_done:
    addi    s1, s1, 1; j .LQ_j
.LQ_done:
    ld      ra, 0(sp); ld s0, 8(sp); ld s1, 16(sp); ld s2, 24(sp)
    addi    sp, sp, 32; ret
    .size state_init_Q, .-state_init_Q


# =============================================================================
#  meas_init_H  —  build H (69×276 block-diagonal measurement matrix)
#
#  Python:
#    H = zeros(69, 276)
#    for j in 0..22:
#      rb = j*3;  cb = j*12
#      H[rb+0, cb+0] = 1.0   # px
#      H[rb+1, cb+4] = 1.0   # py
#      H[rb+2, cb+8] = 1.0   # pz
#
#  void meas_init_H(double *H)
#  a0=H
# =============================================================================
    .globl meas_init_H
    .type  meas_init_H, @function
meas_init_H:
    addi    sp, sp, -24
    sd      ra, 0(sp); sd s0, 8(sp); sd s1, 16(sp)

    mv      s0, a0

    # zero H: 69*276*8 bytes
    li      t0, 69; li t1, 276; mul t0, t0, t1; slli t0, t0, 3
    add     t1, s0, t0
    la      t2, lkf_fp_zero; fld ft0, 0(t2)
    mv      t2, s0
.LH_zero:
    bge     t2, t1, .LH_zero_done
    fsd     ft0, 0(t2); addi t2, t2, 8; j .LH_zero
.LH_zero_done:

    la      t0, lkf_fp_one; fld ft1, 0(t0)
    li      s1, 0           # j = 0
.LH_j:
    li      t0, 23
    bge     s1, t0, .LH_done

    li      t1, 3;  mul t2, s1, t1   # rb = j*3
    li      t1, 12; mul t3, s1, t1   # cb = j*12

    # H[rb+0, cb+0] = 1.0:  offset = (rb*276 + cb)*8
    li      t4, 276
    mul     t0, t2, t4; add t0, t0, t3
    slli    t0, t0, 3; add t0, s0, t0; fsd ft1, 0(t0)

    # H[rb+1, cb+4] = 1.0
    addi    t5, t2, 1
    mul     t0, t5, t4; add t0, t0, t3; addi t0, t0, 4
    slli    t0, t0, 3; add t0, s0, t0; fsd ft1, 0(t0)

    # H[rb+2, cb+8] = 1.0
    addi    t5, t2, 2
    mul     t0, t5, t4; add t0, t0, t3; addi t0, t0, 8
    slli    t0, t0, 3; add t0, s0, t0; fsd ft1, 0(t0)

    addi    s1, s1, 1; j .LH_j
.LH_done:
    ld      ra, 0(sp); ld s0, 8(sp); ld s1, 16(sp)
    addi    sp, sp, 24; ret
    .size meas_init_H, .-meas_init_H


# =============================================================================
#  meas_init_R  —  build R (69×69 block-diagonal measurement noise)
#
#  Python:
#    R = zeros(69,69)
#    for j in 0..22:
#      b = j*3
#      R[b+0,b+0] = EST_R_PX (0.29472279)
#      R[b+1,b+1] = EST_R_PY (0.09632091)
#      R[b+2,b+2] = EST_R_PZ (0.00204269)
#
#  void meas_init_R(double *R)
#  a0=R
# =============================================================================
    .globl meas_init_R
    .type  meas_init_R, @function
meas_init_R:
    addi    sp, sp, -24
    sd      ra, 0(sp); sd s0, 8(sp); sd s1, 16(sp)

    mv      s0, a0

    # zero R: 69*69*8 bytes
    li      t0, 69; mul t0, t0, t0; slli t0, t0, 3
    add     t1, s0, t0
    la      t2, lkf_fp_zero; fld ft0, 0(t2)
    mv      t2, s0
.LR_zero:
    bge     t2, t1, .LR_zero_done
    fsd     ft0, 0(t2); addi t2, t2, 8; j .LR_zero
.LR_zero_done:

    la      t0, lkf_EST_R_PX; fld ft0, 0(t0)
    la      t0, lkf_EST_R_PY; fld ft1, 0(t0)
    la      t0, lkf_EST_R_PZ; fld ft2, 0(t0)

    li      s1, 0
.LR_j:
    li      t3, 23
    bge     s1, t3, .LR_done
    li      t4, 3; mul t4, s1, t4    # b = j*3

    li      t5, 69
    # R[b+0,b+0]
    mul     t0, t4, t5; add t0, t0, t4
    slli    t0, t0, 3; add t0, s0, t0; fsd ft0, 0(t0)
    # R[b+1,b+1]
    addi    t6, t4, 1
    mul     t0, t6, t5; add t0, t0, t6
    slli    t0, t0, 3; add t0, s0, t0; fsd ft1, 0(t0)
    # R[b+2,b+2]
    addi    t6, t4, 2
    mul     t0, t6, t5; add t0, t0, t6
    slli    t0, t0, 3; add t0, s0, t0; fsd ft2, 0(t0)

    addi    s1, s1, 1; j .LR_j
.LR_done:
    ld      ra, 0(sp); ld s0, 8(sp); ld s1, 16(sp)
    addi    sp, sp, 24; ret
    .size meas_init_R, .-meas_init_R


# =============================================================================
#  lkf_init  —  initialise all fields of an LKF struct
#
#  Python __init__:
#    self.dt = dt
#    self.x  = zeros(276)
#    self.P  = eye(276)
#    self.F  = state_init_F(dt)
#    self.Q  = state_init_Q()
#    self.H  = meas_init_H()
#    self.R  = meas_init_R()
#
#  void lkf_init(void *lkf, double dt)
#  a0=lkf ptr, fa0=dt
#
#  Register map:  s0=lkf
# =============================================================================
    .globl lkf_init
    .type  lkf_init, @function
lkf_init:
    addi    sp, sp, -24
    sd      ra, 0(sp); sd s0, 8(sp)
    fsd     fs0, 16(sp)

    mv      s0, a0
    fmv.d   fs0, fa0            # save dt

    # store dt at offset 0
    fsd     fs0, 0(s0)

    # x = zeros(276): offset 8, 276*8=2208 bytes
    la      t0, lkf_fp_zero; fld ft0, 0(t0)
    addi    t1, s0, 8           # &x[0]
    li      t2, 276; slli t2, t2, 3; add t2, t1, t2
.Linit_xz:
    bge     t1, t2, .Linit_xz_done
    fsd     ft0, 0(t1); addi t1, t1, 8; j .Linit_xz
.Linit_xz_done:

    # P = eye(276): offset 2216
    la      t0, lkf_OFF_P; ld t0, 0(t0)
    add     a0, s0, t0          # &P
    li      a1, 276
    call    mat_eye

    # F = state_init_F(dt): offset 611624
    la      t0, lkf_OFF_F; ld t0, 0(t0)
    add     a0, s0, t0          # &F
    fmv.d   fa0, fs0            # dt
    call    state_init_F

    # Q = state_init_Q(): offset 1221032
    la      t0, lkf_OFF_Q; ld t0, 0(t0)
    add     a0, s0, t0
    call    state_init_Q

    # H = meas_init_H(): offset 1830440
    la      t0, lkf_OFF_H; ld t0, 0(t0)
    add     a0, s0, t0
    call    meas_init_H

    # R = meas_init_R(): offset 1982648
    la      t0, lkf_OFF_R; ld t0, 0(t0)
    add     a0, s0, t0
    call    meas_init_R

    ld      ra, 0(sp); ld s0, 8(sp); fld fs0, 16(sp)
    addi    sp, sp, 24; ret
    .size lkf_init, .-lkf_init


# =============================================================================
#  lkf_set_initial_state  —  set sub-state for one joint from position vector
#
#  Python:
#    b = joint_idx * 12
#    x[b+0]=pos[0]; x[b+1]=0; x[b+2]=0; x[b+3]=0
#    x[b+4]=pos[1]; x[b+5]=0; x[b+6]=0; x[b+7]=0
#    x[b+8]=pos[2]; x[b+9]=0; x[b+10]=0;x[b+11]=0
#
#  void lkf_set_initial_state(void *lkf, int joint_idx, double *pos3)
#  a0=lkf, a1=joint_idx, a2=pos3 (ptr to double[3])
# =============================================================================
    .globl lkf_set_initial_state
    .type  lkf_set_initial_state, @function
lkf_set_initial_state:
    # compute &x[b]:  lkf + OFF_x + joint*12*8
    la      t0, lkf_OFF_x; ld t0, 0(t0)
    add     t0, a0, t0          # &x[0]
    li      t1, 12; mul t1, a1, t1; slli t1, t1, 3
    add     t0, t0, t1          # &x[b]

    la      t1, lkf_fp_zero; fld ft0, 0(t1)

    # load pos[0], pos[1], pos[2]
    fld     ft1, 0(a2)          # pos[0]
    fld     ft2, 8(a2)          # pos[1]
    fld     ft3,16(a2)          # pos[2]

    fsd     ft1,  0(t0)         # x[b+0]  = pos[0]
    fsd     ft0,  8(t0)         # x[b+1]  = 0
    fsd     ft0, 16(t0)         # x[b+2]  = 0
    fsd     ft0, 24(t0)         # x[b+3]  = 0
    fsd     ft2, 32(t0)         # x[b+4]  = pos[1]
    fsd     ft0, 40(t0)         # x[b+5]  = 0
    fsd     ft0, 48(t0)         # x[b+6]  = 0
    fsd     ft0, 56(t0)         # x[b+7]  = 0
    fsd     ft3, 64(t0)         # x[b+8]  = pos[2]
    fsd     ft0, 72(t0)         # x[b+9]  = 0
    fsd     ft0, 80(t0)         # x[b+10] = 0
    fsd     ft0, 88(t0)         # x[b+11] = 0
    ret
    .size lkf_set_initial_state, .-lkf_set_initial_state


# =============================================================================
#  lkf_predict  —  prediction step
#
#  Python:
#    self.x = F @ x                     (state_predict_x)
#    self.P = F @ P @ F.T + Q           (state_predict_P)
#
#  Implementation:
#    x_new = mat_mul(F, x, 276, 276, 1)  [mat_vec_mul]
#    Ftmp  = mat_mul(F, P, 276, 276, 276)
#    P_new = mat_mul(Ftmp, F^T, 276, 276, 276) + Q
#
#  Scratch used:  PHt field reused as Ftmp (276×276 = 609408 bytes)
#                 Note: PHt is 276×69 which is SMALLER than 276×276.
#                 We use the joseph_scratch area (4*276*276) as Ftmp.
#
#  void lkf_predict(void *lkf)
#  a0=lkf
#
#  Register map:  s0=lkf  s1=&x  s2=&P  s3=&F  s4=&Q  s5=&jscratch(Ftmp)  s6=&FT
# =============================================================================
    .globl lkf_predict
    .type  lkf_predict, @function
lkf_predict:
    addi    sp, sp, -64
    sd      ra,  0(sp); sd s0,  8(sp); sd s1, 16(sp)
    sd      s2, 24(sp); sd s3, 32(sp); sd s4, 40(sp)
    sd      s5, 48(sp); sd s6, 56(sp)

    mv      s0, a0

    # load field pointers
    la      t0, lkf_OFF_x;  ld t0, 0(t0); add s1, s0, t0   # &x
    la      t0, lkf_OFF_P;  ld t0, 0(t0); add s2, s0, t0   # &P
    la      t0, lkf_OFF_F;  ld t0, 0(t0); add s3, s0, t0   # &F
    la      t0, lkf_OFF_Q;  ld t0, 0(t0); add s4, s0, t0   # &Q
    # Ftmp = jscratch[0..276*276-1], FT = jscratch[276*276..2*276*276-1]
    la      t0, lkf_OFF_jscratch; ld t0, 0(t0); add s5, s0, t0  # Ftmp
    li      t1, 276; mul t1, t1, t1; slli t1, t1, 3
    add     s6, s5, t1          # FT = Ftmp + 276*276*8

    # ---- x_new = F @ x  (mat_vec_mul: F(276x276), x(276)) ------------------
    # We compute in-place: use y field or a stack buf.
    # Use a stack-allocated 276*8=2208 byte buf for x_new
    li      t0, 276; slli t0, t0, 3   # 2208
    sub     sp, sp, t0
    mv      t1, sp                    # x_new on stack

    mv      a0, t1; mv a1, s3; mv a2, s1
    li      a3, 276; li a4, 276
    call    mat_vec_mul         # x_new = F @ x

    # copy x_new -> x
    mv      t2, s1; mv t3, t1
    li      t4, 276; slli t4, t4, 3; add t4, t2, t4
.Lpr_xcopy:
    bge     t2, t4, .Lpr_xcopy_done
    fld     ft0, 0(t3); fsd ft0, 0(t2)
    addi    t2, t2, 8; addi t3, t3, 8; j .Lpr_xcopy
.Lpr_xcopy_done:
    # free x_new stack buf
    li      t0, 276; slli t0, t0, 3; add sp, sp, t0

    # ---- P_new = F @ P @ F^T + Q --------------------------------------------
    # Step 1: Ftmp = F @ P   (276×276)
    mv      a0, s5; mv a1, s3; mv a2, s2
    li      a3, 276; li a4, 276; li a5, 276
    call    mat_mul

    # Step 2: FT = F^T
    mv      a0, s6; mv a1, s3; li a2, 276; li a3, 276
    call    mat_transpose

    # Step 3: P = Ftmp @ FT  (write directly into P)
    mv      a0, s2; mv a1, s5; mv a2, s6
    li      a3, 276; li a4, 276; li a5, 276
    call    mat_mul

    # Step 4: P = P + Q
    mv      a0, s2; mv a1, s2; mv a2, s4
    li      a3, 276; li a4, 276
    call    mat_add

    ld      ra,  0(sp); ld s0,  8(sp); ld s1, 16(sp)
    ld      s2, 24(sp); ld s3, 32(sp); ld s4, 40(sp)
    ld      s5, 48(sp); ld s6, 56(sp)
    addi    sp, sp, 64; ret
    .size lkf_predict, .-lkf_predict


# =============================================================================
#  lkf_update  —  measurement update step
#
#  Python:
#    z = flat measurement vector (69,)         [built from measurements]
#    y  = z - H @ x                            (innovation, 69)
#    PHt = P @ H.T                             (276×69)
#    S   = H @ PHt + R                         (69×69)
#    Sinv, ok = mat_inverse_nxn(S)
#    if not ok: warn and return
#    K   = PHt @ Sinv                          (276×69)
#    x   = x + K @ y
#    P   = mat_joseph_update(P, K, H, R)
#
#  void lkf_update(void *lkf, const double *z_flat)
#  a0=lkf, a1=z_flat  (pre-built 69-element Cartesian measurement vector)
#
#  Note: z_flat is already the packed (NUM_JOINTS*3,) Cartesian array,
#        matching self.z built in the Python update() method.
#
#  Scratch layout inside the struct (reused across calls):
#    PHt[276×69]  @ OFF_PHt
#    S[69×69]     @ OFF_S
#    Sinv[69×69]  @ OFF_Sinv
#    y[69]        @ OFF_y
#    piv[69]      @ OFF_piv   (int32)
#    jscratch[4*276*276] @ OFF_jscratch  (Joseph form)
#
#  Register map:
#    s0=lkf  s1=&x  s2=&P  s3=&H  s4=&R
#    s5=&PHt s6=&S  s7=&Sinv  s8=&y  s9=z_flat  s10=&piv  s11=&jscratch
# =============================================================================
    .globl lkf_update
    .type  lkf_update, @function
lkf_update:
    addi    sp, sp, -104
    sd      ra,  0(sp); sd s0,  8(sp); sd s1, 16(sp); sd s2, 24(sp)
    sd      s3, 32(sp); sd s4, 40(sp); sd s5, 48(sp); sd s6, 56(sp)
    sd      s7, 64(sp); sd s8, 72(sp); sd s9, 80(sp); sd s10, 88(sp)
    sd      s11, 96(sp)

    mv      s0, a0
    mv      s9, a1              # z_flat (69 doubles, already packed)

    # load all field pointers
    la      t0, lkf_OFF_x;        ld t0, 0(t0); add s1, s0, t0
    la      t0, lkf_OFF_P;        ld t0, 0(t0); add s2, s0, t0
    la      t0, lkf_OFF_H;        ld t0, 0(t0); add s3, s0, t0
    la      t0, lkf_OFF_R;        ld t0, 0(t0); add s4, s0, t0
    la      t0, lkf_OFF_PHt;      ld t0, 0(t0); add s5, s0, t0
    la      t0, lkf_OFF_S;        ld t0, 0(t0); add s6, s0, t0
    la      t0, lkf_OFF_Sinv;     ld t0, 0(t0); add s7, s0, t0
    la      t0, lkf_OFF_y;        ld t0, 0(t0); add s8, s0, t0
    la      t0, lkf_OFF_piv;      ld t0, 0(t0); add s10, s0, t0
    la      t0, lkf_OFF_jscratch; ld t0, 0(t0); add s11, s0, t0

    # ---- Step 1: y = z - H @ x  (innovation) --------------------------------
    # z_pred = H @ x  (69×276 × 276 → 69)  stored in y first
    mv      a0, s8; mv a1, s3; mv a2, s1
    li      a3, 69; li a4, 276
    call    mat_vec_mul         # y = H @ x  (z_pred)

    # y = z_flat - y  (y = z - H@x)
    mv      a0, s8; mv a1, s9; mv a2, s8
    li      a3, 1; li a4, 69
    call    mat_sub             # y = z - z_pred  (1×69 treated as flat vector)

    # ---- Step 2: PHt = P @ H^T  (276×276 × 276×69 → 276×69) ----------------
    # We need H^T (276×69). H is 69×276.
    # Optimisation: H is block-diagonal with only 3 nonzero entries per 3×12 block.
    # For generality (and correctness vs Python), compute H^T explicitly.
    # Use S scratch as temp for HT (276×69 = 152208 bytes; S is 38088 bytes).
    # S is too small. Use jscratch first 276*69*8=152208 bytes for HT.
    mv      a0, s11; mv a1, s3; li a2, 69; li a3, 276
    call    mat_transpose       # jscratch[0..152207] = H^T (276×69)

    # PHt = P @ H^T  (276×276 × 276×69 → 276×69)
    mv      a0, s5; mv a1, s2; mv a2, s11
    li      a3, 276; li a4, 276; li a5, 69
    call    mat_mul

    # ---- Step 3: S = H @ PHt + R  (69×276 × 276×69 → 69×69) ---------------
    mv      a0, s6; mv a1, s3; mv a2, s5
    li      a3, 69; li a4, 276; li a5, 69
    call    mat_mul             # S = H @ PHt

    mv      a0, s6; mv a1, s6; mv a2, s4
    li      a3, 69; li a4, 69
    call    mat_add             # S = S + R

    # ---- Step 4: Sinv = S^{-1}  (69×69) ------------------------------------
    mv      a0, s7; mv a1, s6; li a2, 69; mv a3, s10
    call    mat_inverse_nxn     # returns 1=ok, 0=singular in a0

    # Check singularity
    bnez    a0, .Lupd_ok
    # Singular: print warning and skip update
    la      a0, .Lwarn_singular
    call    puts
    j       .Lupd_done

.Lupd_ok:
    # ---- Step 5: K = PHt @ Sinv  (276×69 × 69×69 → 276×69) ----------------
    # Reuse jscratch for K (276×69 = 152208 bytes)
    mv      a0, s11; mv a1, s5; mv a2, s7
    li      a3, 276; li a4, 69; li a5, 69
    call    mat_mul             # jscratch = K

    # ---- Step 6: x = x + K @ y  (276×69 × 69 → 276) -----------------------
    # K@y into a stack buffer (276*8=2208 bytes)
    li      t0, 276; slli t0, t0, 3
    sub     sp, sp, t0
    mv      t1, sp              # Ky on stack

    mv      a0, t1; mv a1, s11; mv a2, s8
    li      a3, 276; li a4, 69
    call    mat_vec_mul         # Ky = K @ y

    # x = x + Ky
    mv      a0, s1; mv a1, s1; mv a2, t1
    li      a3, 1; li a4, 276
    call    mat_add

    li      t0, 276; slli t0, t0, 3; add sp, sp, t0   # free Ky

    # ---- Step 7: P = (I-KH)P(I-KH)^T + KRK^T  (Joseph form) ---------------
    # mat_joseph_update(Pout, P, K, H, R, n, m, scratch)
    # K = jscratch (276×69), scratch starts after K in jscratch
    # jscratch layout: [K(276×69=152208B)] [joseph_scratch(4*276*276*8=2437632B)]
    # joseph needs 4*n*n doubles = 4*276*276 for scratch, start after K
    li      t0, 276; li t1, 69; mul t0, t0, t1; slli t0, t0, 3
    add     t1, s11, t0         # &jscratch[276*69] = joseph scratch start

    mv      a0, s2              # Pout = P (in-place)
    mv      a1, s2              # P
    mv      a2, s11             # K
    mv      a3, s3              # H
    mv      a4, s4              # R
    li      a5, 276             # n
    li      a6, 69              # m
    mv      a7, t1              # scratch
    call    mat_joseph_update

.Lupd_done:
    ld      ra,  0(sp); ld s0,  8(sp); ld s1, 16(sp); ld s2, 24(sp)
    ld      s3, 32(sp); ld s4, 40(sp); ld s5, 48(sp); ld s6, 56(sp)
    ld      s7, 64(sp); ld s8, 72(sp); ld s9, 80(sp); ld s10, 88(sp)
    ld      s11, 96(sp)
    addi    sp, sp, 104; ret

.Lwarn_singular:
    .asciz  "[WARN] LKF: singular 69x69 S, skipping update\n"

    .size lkf_update, .-lkf_update


# =============================================================================
#  lkf_get_positions  —  extract (NUM_JOINTS, 3) position array
#
#  Python:
#    for j in 0..22:
#      b = j * 12
#      pos[j] = [x[b], x[b+4], x[b+8]]
#
#  void lkf_get_positions(const void *lkf, double *pos_out)
#  a0=lkf, a1=pos_out  (double[23*3] = double[69])
# =============================================================================
    .globl lkf_get_positions
    .type  lkf_get_positions, @function
lkf_get_positions:
    la      t0, lkf_OFF_x; ld t0, 0(t0); add t0, a0, t0  # &x[0]
    li      t1, 0               # j = 0
.Lgp_j:
    li      t2, 23
    bge     t1, t2, .Lgp_done
    # b = j*12;  &x[b] = t0 + b*8 = t0 + j*96
    li      t2, 12; mul t2, t1, t2; slli t2, t2, 3
    add     t3, t0, t2          # &x[b]
    # pos[j*3+0] = x[b+0]
    li      t4, 3; mul t4, t1, t4; slli t4, t4, 3; add t4, a1, t4  # &pos[j*3]
    fld     ft0,  0(t3); fsd ft0, 0(t4)         # px
    fld     ft0, 32(t3); fsd ft0, 8(t4)         # py = x[b+4]  (offset 4*8=32)
    fld     ft0, 64(t3); fsd ft0,16(t4)         # pz = x[b+8]  (offset 8*8=64)
    addi    t1, t1, 1; j .Lgp_j
.Lgp_done:
    ret
    .size lkf_get_positions, .-lkf_get_positions


# =============================================================================
#  lkf_get_full_state  —  copy x[276] to output buffer
#
#  void lkf_get_full_state(const void *lkf, double *out)
#  a0=lkf, a1=out (double[276])
# =============================================================================
    .globl lkf_get_full_state
    .type  lkf_get_full_state, @function
lkf_get_full_state:
    la      t0, lkf_OFF_x; ld t0, 0(t0); add t0, a0, t0  # &x[0]
    li      t1, 276; slli t1, t1, 3; add t2, t0, t1       # end ptr
.Lgfs_loop:
    bge     t0, t2, .Lgfs_done
    fld     ft0, 0(t0); fsd ft0, 0(a1)
    addi    t0, t0, 8; addi a1, a1, 8; j .Lgfs_loop
.Lgfs_done:
    ret
    .size lkf_get_full_state, .-lkf_get_full_state

# =============================================================================
#  END OF lkf_asm.s
# =============================================================================
