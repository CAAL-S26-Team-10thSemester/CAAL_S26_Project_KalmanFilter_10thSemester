# -*- coding: utf-8 -*-
"""
Kalman Filter Milestone 2 — VSCode / local Python (single-file, no C files)
=============================================================================
Run from the directory containing both CSV files:
  - 3D Full Body Humain Gait Walking Dataset (Noisy Values).csv
  - 3D Full Body Humain Gait Walking Dataset (True Values).csv

Everything (LKF, EKF, plotting, animation) lives in this one file.
No separate C files, helper scripts, or Colab magic commands are used.

Install dependencies once:
    pip install numpy pandas matplotlib pillow
    (optional for MP4 export: install ffmpeg system-wide)
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 — registers 3-D projection
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# ==============================================================================
# USER SETTINGS — edit as needed
# ==============================================================================

NOISY_CSV  = "3D Full Body Humain Gait Walking Dataset (Noisy Values).csv"
TRUE_CSV   = "3D Full Body Humain Gait Walking Dataset (True Values).csv"
LKF_OUT    = "lkf_results.csv"
EKF_OUT    = "ekf_results.csv"
DT         = 1.0 / 30.0   # seconds per frame (30 FPS)
JOINT_IDX  = 0             # joint analysed in 2-D plots (0 = pelvis, 0-22)
MAX_FRAMES = 3040          # frame cap for animation
FRAME_STEP = 4             # animate every Nth frame

# -- Validate inputs ----------------------------------------------------------
for _p in [NOISY_CSV, TRUE_CSV]:
    if not os.path.exists(_p):
        sys.exit(f"ERROR: Cannot find '{_p}'.\nPlace both CSVs in: {os.getcwd()}")

print(f"\u2705 Noisy : {NOISY_CSV}")
print(f"\u2705 True  : {TRUE_CSV}")

# ==============================================================================
# CONSTANTS
# ==============================================================================

NUM_JOINTS      = 23
STATE_DIM       = 12    # per joint: [px,vx,ax,jx, py,vy,ay,jy, pz,vz,az,jz]
MEAS_DIM        = 3     # per joint: [px, py, pz]
TOTAL_STATE_DIM = NUM_JOINTS * STATE_DIM   # 276  <-- single large state vector
TOTAL_MEAS_DIM  = NUM_JOINTS * MEAS_DIM   #  69

EPSILON_R   = 1e-10
EPSILON_RHO = 1e-10

# Cartesian measurement noise (LKF R matrix)
EST_R_PX = 0.29472279
EST_R_PY = 0.09632091
EST_R_PZ = 0.00204269

JOINT_NAMES = [
    "pelvis",        "L5",            "L3",           "T12",
    "T8",            "neck",          "head",
    "shoulderRight", "upperArmRight", "forearmRight",  "handRight",
    "shoulderLeft",  "upperArmLeft",  "forearmLeft",   "handLeft",
    "upperLegRight", "lowerLegRight", "footRight",     "toeRight",
    "upperLegLeft",  "lowerLegLeft",  "footLeft",      "toeLeft",
]

BONES = [
    (0, 1),  (1, 2),  (2, 3),  (3, 4),  (4, 5),  (5, 6),
    (4, 7),  (7, 8),  (8, 9),  (9, 10),
    (4, 11), (11, 12), (12, 13), (13, 14),
    (0, 15), (15, 16), (16, 17), (17, 18),
    (0, 19), (19, 20), (20, 21), (21, 22),
]

# ==============================================================================
# MATRIX UTILITIES  (pure NumPy — replaces C matrix.c in this Python version)
# ==============================================================================

def mat_eye(n):
    return np.eye(n)

def mat_mul(A, B):
    return A @ B

def mat_add(A, B):
    return A + B

def mat_sub(A, B):
    return A - B

def mat_transpose(A):
    return A.T

def mat_inverse_nxn(A):
    """
    General N x N matrix inverse.
    Returns (inv, ok) where ok=False when the matrix is singular.
    Corresponds to the LU-decomposition mat_inverse_nxn in the C build.
    Used here to invert the 69x69 innovation covariance S.
    """
    try:
        return np.linalg.inv(A), True
    except np.linalg.LinAlgError:
        return np.zeros_like(A), False

def mat_joseph_update(P, K, H, R):
    """
    Joseph-form covariance update (numerically stable PSD guarantee):
        P = (I - K*H) * P * (I - K*H)^T  +  K * R * K^T
    Shapes: K (n x m), H (m x n), P (n x n), R (m x m).
    """
    n   = P.shape[0]
    IKH = np.eye(n) - K @ H
    return IKH @ P @ IKH.T + K @ R @ K.T

# ==============================================================================
# STATE TRANSITION & PROCESS NOISE  (block-diagonal, 276 x 276)
# ==============================================================================

def state_init_F(dt):
    """
    Build the global block-diagonal F (276 x 276).

    23 identical 12 x 12 kinematic blocks sit on the main diagonal.
    Each 12 x 12 block has three 4 x 4 Taylor-expansion sub-blocks (X, Y, Z):

        [ 1   dt   dt2/2   dt3/6 ]
        [ 0    1     dt    dt2/2 ]
        [ 0    0      1      dt  ]
        [ 0    0      0       1  ]

    All joints are kinematically independent -- zero off-diagonal coupling.
    """
    N   = TOTAL_STATE_DIM
    F   = np.eye(N)
    dt2 = 0.5 * dt * dt
    dt3 = (1.0 / 6.0) * dt ** 3
    for j in range(NUM_JOINTS):
        base = j * STATE_DIM
        for axis in range(3):
            r = base + axis * 4
            F[r,     r + 1] = dt
            F[r,     r + 2] = dt2
            F[r,     r + 3] = dt3
            F[r + 1, r + 2] = dt
            F[r + 1, r + 3] = dt2
            F[r + 2, r + 3] = dt
    return F


def state_init_Q():
    """
    Build the global block-diagonal Q (276 x 276).
    23 identical 12 x 12 diagonal blocks; noise tuned so
    position variance << velocity << acceleration ~= jerk.
    """
    N         = TOTAL_STATE_DIM
    Q         = np.zeros((N, N))
    noise_map = {0: 1e-6, 1: 1e-5, 2: 1e-4, 3: 1e-4}
    for j in range(NUM_JOINTS):
        base = j * STATE_DIM
        for i in range(STATE_DIM):
            Q[base + i, base + i] = noise_map[i % 4]
    return Q


def state_predict_x(x, F):
    return F @ x


def state_predict_P(P, F, Q):
    return F @ P @ F.T + Q

# ==============================================================================
# MEASUREMENT MATRICES  (LKF -- Cartesian, 69 x 276)
# ==============================================================================

def meas_init_H():
    """
    Build the global block-diagonal H (69 x 276).
    For joint j the 3 x 12 block sits at rows [j*3..j*3+2],
    cols [j*12..j*12+11] and picks out px(0), py(4), pz(8).
    """
    H = np.zeros((TOTAL_MEAS_DIM, TOTAL_STATE_DIM))
    for j in range(NUM_JOINTS):
        rb = j * MEAS_DIM
        cb = j * STATE_DIM
        H[rb + 0, cb + 0] = 1.0   # px
        H[rb + 1, cb + 4] = 1.0   # py
        H[rb + 2, cb + 8] = 1.0   # pz
    return H


def meas_init_R():
    """
    Build the global block-diagonal R (69 x 69).
    23 identical 3 x 3 diagonal blocks with empirical Cartesian noise.
    """
    m = TOTAL_MEAS_DIM
    R = np.zeros((m, m))
    for j in range(NUM_JOINTS):
        b = j * MEAS_DIM
        R[b + 0, b + 0] = EST_R_PX
        R[b + 1, b + 1] = EST_R_PY
        R[b + 2, b + 2] = EST_R_PZ
    return R

# ==============================================================================
# LINEAR KALMAN FILTER  (single 276-D full-body state vector)
# ==============================================================================

class LKF:
    """
    Linear Kalman Filter operating on a SINGLE 276-D full-body state vector.

    State layout per joint j (offset j*12):
        [px, vx, ax, jx,  py, vy, ay, jy,  pz, vz, az, jz]

    Global matrices:
        F  : 276 x 276  (block-diagonal state transition)
        Q  : 276 x 276  (block-diagonal process noise)
        H  :  69 x 276  (block-diagonal Cartesian position extractor)
        R  :  69 x  69  (block-diagonal Cartesian measurement noise)
        S  :  69 x  69  (innovation covariance, inverted via numpy)
    """

    def __init__(self, dt: float):
        self.dt = dt
        self.x  = np.zeros(TOTAL_STATE_DIM)
        self.P  = np.eye(TOTAL_STATE_DIM)
        self.F  = state_init_F(dt)
        self.Q  = state_init_Q()
        self.H  = meas_init_H()
        self.R  = meas_init_R()

    def set_initial_state(self, joint_idx: int, pos: np.ndarray):
        """Initialise joint joint_idx's sub-state in the global x vector."""
        b = joint_idx * STATE_DIM
        self.x[b + 0]  = pos[0]; self.x[b + 1]  = 0.0
        self.x[b + 2]  = 0.0;    self.x[b + 3]  = 0.0
        self.x[b + 4]  = pos[1]; self.x[b + 5]  = 0.0
        self.x[b + 6]  = 0.0;    self.x[b + 7]  = 0.0
        self.x[b + 8]  = pos[2]; self.x[b + 9]  = 0.0
        self.x[b + 10] = 0.0;    self.x[b + 11] = 0.0

    def predict(self):
        """Prediction step on the full 276-D state."""
        self.x = state_predict_x(self.x, self.F)
        self.P = state_predict_P(self.P, self.F, self.Q)

    def update(self, measurements: np.ndarray):
        """
        Single update on the 276-D state.
        measurements: flat array (NUM_JOINTS * 3,) in Cartesian.

            y  = z - H*x        (69-D innovation)
            S  = H*P*H^T + R    (69 x 69, inverted via numpy)
            K  = P*H^T*S^{-1}   (276 x 69 Kalman gain)
            x  = x + K*y
            P  = (I-KH)*P*(I-KH)^T + K*R*K^T  (Joseph form)
        """
        z = np.zeros(TOTAL_MEAS_DIM)
        for j in range(NUM_JOINTS):
            z[j * MEAS_DIM:(j + 1) * MEAS_DIM] = measurements[j * 3:(j + 1) * 3]

        z_pred   = self.H @ self.x
        y        = z - z_pred
        PHt      = self.P @ self.H.T
        S        = self.H @ PHt + self.R
        Sinv, ok = mat_inverse_nxn(S)
        if not ok:
            print("[WARN] LKF: singular 69x69 S, skipping update")
            return
        K        = PHt @ Sinv
        self.x   = self.x + K @ y
        self.P   = mat_joseph_update(self.P, K, self.H, self.R)

    def get_positions(self) -> np.ndarray:
        pos = np.zeros((NUM_JOINTS, 3))
        for j in range(NUM_JOINTS):
            b = j * STATE_DIM
            pos[j] = [self.x[b], self.x[b + 4], self.x[b + 8]]
        return pos

    def get_full_state(self) -> np.ndarray:
        return self.x.copy()

# ==============================================================================
# EKF HELPERS  (spherical measurement model + manual atan2)
# ==============================================================================

def fast_atan2(y, x):
    """
    Polynomial approximation of atan2 (Scheinerman-Lyons formula).
    Max error ~0.021 deg. Matches the C atan_utils.c implementation exactly.
    No math.atan2 is used.
    """
    if x == 0.0 and y == 0.0:
        return 0.0
    PI  = 3.14159265358979323846
    PI2 = 1.57079632679489661923
    ax  = abs(x)
    ay  = abs(y)
    if ax >= ay:
        z     = ay / ax
        angle = (PI / 4.0) * z - z * (z - 1.0) * (0.2447 + 0.0663 * z)
    else:
        z     = ax / ay
        angle = PI2 - ((PI / 4.0) * z - z * (z - 1.0) * (0.2447 + 0.0663 * z))
    if x < 0:
        angle = (PI - angle) if y >= 0 else (angle - PI)
    elif y < 0:
        angle = -angle
    return angle


def wrap_angle(a: float) -> float:
    PI = 3.14159265358979323846
    while a >  PI: a -= 2.0 * PI
    while a < -PI: a += 2.0 * PI
    return a


def ekf_compute_h(x: np.ndarray) -> np.ndarray:
    """
    Global nonlinear measurement function h: R^276 -> R^69.
    For each joint j: [r_j, theta_j, phi_j]
    where r = ||p||, theta = atan2(py, px), phi = atan2(pz, rho).
    fast_atan2 is used throughout -- no math.atan2.
    """
    h_out = np.zeros(TOTAL_MEAS_DIM)
    for j in range(NUM_JOINTS):
        b   = j * STATE_DIM
        px  = x[b]; py = x[b + 4]; pz = x[b + 8]
        r   = max(np.sqrt(px*px + py*py + pz*pz), EPSILON_R)
        rho = np.sqrt(px*px + py*py)
        hb  = j * MEAS_DIM
        h_out[hb + 0] = r
        h_out[hb + 1] = fast_atan2(py, px)    # azimuth
        h_out[hb + 2] = fast_atan2(pz, rho)   # elevation
    return h_out


def ekf_compute_jacobian(x: np.ndarray) -> np.ndarray:
    """
    Global Jacobian Hk = dh/dx (69 x 276, block-diagonal).
    For joint j the 3 x 12 block sits at rows [j*3..j*3+2],
    cols [j*12..j*12+11].  Only 9 of 36 entries are non-zero.

    Non-zero partial derivatives per joint:
        dr/dpx  =  px/r           dr/dpy  =  py/r           dr/dpz  =  pz/r
        dth/dpx = -py/rho2        dth/dpy =  px/rho2        dth/dpz =  0
        dph/dpx = -px*pz/(rho*r2) dph/dpy = -py*pz/(rho*r2) dph/dpz =  rho/r2
    """
    Hk = np.zeros((TOTAL_MEAS_DIM, TOTAL_STATE_DIM))
    for j in range(NUM_JOINTS):
        b    = j * STATE_DIM
        px   = x[b]; py = x[b + 4]; pz = x[b + 8]
        r2   = px*px + py*py + pz*pz
        r    = max(np.sqrt(r2),   EPSILON_R)
        rho2 = px*px + py*py
        rho  = max(np.sqrt(rho2), EPSILON_RHO)
        r2s  = r * r;  rho2s = rho * rho
        rb   = j * MEAS_DIM
        cb   = j * STATE_DIM
        # Row 0: dr/dx
        Hk[rb + 0, cb + 0] =  px / r
        Hk[rb + 0, cb + 4] =  py / r
        Hk[rb + 0, cb + 8] =  pz / r
        # Row 1: dtheta/dx
        Hk[rb + 1, cb + 0] = -py / rho2s
        Hk[rb + 1, cb + 4] =  px / rho2s
        # Row 2: dphi/dx
        Hk[rb + 2, cb + 0] = -(px * pz) / (rho * r2s)
        Hk[rb + 2, cb + 4] = -(py * pz) / (rho * r2s)
        Hk[rb + 2, cb + 8] =  rho / r2s
    return Hk


def ekf_init_R():
    """
    Build global spherical measurement noise R (69 x 69 diagonal).
    Each 3 x 3 block holds (sigma2_r, sigma2_theta, sigma2_phi).
    """
    m = TOTAL_MEAS_DIM
    R = np.zeros((m, m))
    for j in range(NUM_JOINTS):
        b = j * MEAS_DIM
        R[b + 0, b + 0] = 0.05    # sigma2_r
        R[b + 1, b + 1] = 0.001   # sigma2_theta
        R[b + 2, b + 2] = 0.001   # sigma2_phi
    return R

# ==============================================================================
# EXTENDED KALMAN FILTER  (spherical measurement model, 276-D state)
# ==============================================================================

class EKF:
    """
    Extended Kalman Filter operating on a SINGLE 276-D full-body state vector.

    Measurement model: spherical (r, theta, phi) per joint -> 69-D z vector.
    Prediction is identical to LKF (linear constant-jerk kinematics).
    Update uses the time-varying Jacobian Hk = dh/dx|_{x_hat} (69 x 276)
    and inverts the 69 x 69 innovation covariance via numpy.
    """

    def __init__(self, dt: float):
        self.dt = dt
        self.x  = np.zeros(TOTAL_STATE_DIM)
        self.P  = np.eye(TOTAL_STATE_DIM)
        self.F  = state_init_F(dt)
        self.Q  = state_init_Q()
        self.R  = ekf_init_R()

    def set_initial_state(self, joint_idx: int, pos: np.ndarray):
        b = joint_idx * STATE_DIM
        self.x[b + 0]  = pos[0]; self.x[b + 1]  = 0.0
        self.x[b + 2]  = 0.0;    self.x[b + 3]  = 0.0
        self.x[b + 4]  = pos[1]; self.x[b + 5]  = 0.0
        self.x[b + 6]  = 0.0;    self.x[b + 7]  = 0.0
        self.x[b + 8]  = pos[2]; self.x[b + 9]  = 0.0
        self.x[b + 10] = 0.0;    self.x[b + 11] = 0.0

    def predict(self):
        self.x = state_predict_x(self.x, self.F)
        self.P = state_predict_P(self.P, self.F, self.Q)

    def update(self, measurements: np.ndarray):
        """
        EKF update on the full 276-D state.
        measurements: flat array (NUM_JOINTS * 3,) in Cartesian.

        Steps:
          1. Convert measured Cartesian positions -> spherical z_sph (69-D)
          2. Predicted measurement: z_pred = h(x_hat) (69-D)
          3. Innovation nu = z_sph - z_pred (wrap angular channels)
          4. Global Jacobian Hk (69 x 276)
          5. Sk = Hk*P*Hk^T + R (69 x 69), invert via numpy
          6. Kk = P*Hk^T*Sk^{-1} (276 x 69)
          7. x = x + Kk*nu;  P via Joseph form
        """
        z_tmp = np.zeros(TOTAL_STATE_DIM)
        for j in range(NUM_JOINTS):
            b = j * STATE_DIM
            z_tmp[b + 0] = measurements[j * 3 + 0]
            z_tmp[b + 4] = measurements[j * 3 + 1]
            z_tmp[b + 8] = measurements[j * 3 + 2]

        z_sph  = ekf_compute_h(z_tmp)
        z_pred = ekf_compute_h(self.x)

        nu = np.zeros(TOTAL_MEAS_DIM)
        for j in range(NUM_JOINTS):
            b = j * MEAS_DIM
            nu[b + 0] = z_sph[b + 0] - z_pred[b + 0]
            nu[b + 1] = wrap_angle(z_sph[b + 1] - z_pred[b + 1])
            nu[b + 2] = wrap_angle(z_sph[b + 2] - z_pred[b + 2])

        Hk          = ekf_compute_jacobian(self.x)
        PHkt        = self.P @ Hk.T
        Sk          = Hk @ PHkt + self.R
        Skinv, ok   = mat_inverse_nxn(Sk)
        if not ok:
            print("[WARN] EKF: singular 69x69 Sk, skipping update")
            return
        Kk     = PHkt @ Skinv
        self.x = self.x + Kk @ nu
        self.P = mat_joseph_update(self.P, Kk, Hk, self.R)

    def get_positions(self) -> np.ndarray:
        pos = np.zeros((NUM_JOINTS, 3))
        for j in range(NUM_JOINTS):
            b = j * STATE_DIM
            pos[j] = [self.x[b], self.x[b + 4], self.x[b + 8]]
        return pos

    def get_full_state(self) -> np.ndarray:
        return self.x.copy()

# ==============================================================================
# DATA I/O
# ==============================================================================

def load_raw_csv(path: str, n_joints: int = NUM_JOINTS) -> np.ndarray:
    """Load a gait CSV. Returns ndarray (N_frames, n_joints, 3)."""
    df = pd.read_csv(path, header=None, dtype=str)
    try:
        float(df.iloc[0, 0])
    except (ValueError, TypeError):
        df = df.iloc[1:].reset_index(drop=True)
    cols = n_joints * 3
    data = df.iloc[:, :cols].astype(float).values
    assert data.shape[1] == cols, f"Expected {cols} columns, got {data.shape[1]}"
    return data.reshape(-1, n_joints, 3)


def run_filter(filter_obj, data: np.ndarray, label: str) -> np.ndarray:
    """
    Run predict/update loop over all frames.
    Returns (N_frames, TOTAL_STATE_DIM=276) results array.
    """
    N       = data.shape[0]
    results = np.zeros((N, TOTAL_STATE_DIM))
    for j in range(NUM_JOINTS):
        filter_obj.set_initial_state(j, data[0, j, :])
    print(f"[INFO] Running {label} on {N} frames (276-D global state)...")
    for frame in range(N):
        if frame > 0:
            filter_obj.predict()
        filter_obj.update(data[frame].flatten())
        results[frame] = filter_obj.get_full_state()
        if (frame + 1) % 200 == 0 or frame == N - 1:
            print(f"\r  {label}: {frame + 1}/{N}", end="", flush=True)
    print()
    return results


def save_filter_csv(path: str, results: np.ndarray):
    """Save (N, TOTAL_STATE_DIM) results to CSV with a header row."""
    cols = ["frame"]
    for j in range(NUM_JOINTS):
        for ax in ["x", "y", "z"]:
            cols += [f"j{j}_p{ax}", f"j{j}_v{ax}", f"j{j}_a{ax}", f"j{j}_j{ax}"]
    df = pd.DataFrame(results, columns=cols[1:])
    df.insert(0, "frame", np.arange(len(results)))
    df.to_csv(path, index=False)
    print(f"\u2705 Saved: {path}")


def load_filter_csv(path: str) -> np.ndarray:
    """Load position array (N_frames, NUM_JOINTS, 3) from a results CSV."""
    df  = pd.read_csv(path)
    N   = len(df)
    pos = np.zeros((N, NUM_JOINTS, 3))
    for j in range(NUM_JOINTS):
        b = 1 + j * STATE_DIM
        pos[:, j, 0] = df.iloc[:, b + 0].values
        pos[:, j, 1] = df.iloc[:, b + 4].values
        pos[:, j, 2] = df.iloc[:, b + 8].values
    return pos

# ==============================================================================
# MAIN -- run both filters
# ==============================================================================

print("\n-- Loading CSV data --------------------------------------------------")
noisy_arr = load_raw_csv(NOISY_CSV)
true_arr  = load_raw_csv(TRUE_CSV)
print(f"   Noisy: {noisy_arr.shape}  |  True: {true_arr.shape}")

lkf     = LKF(DT)
lkf_res = run_filter(lkf, noisy_arr, "LKF")
save_filter_csv(LKF_OUT, lkf_res)

ekf     = EKF(DT)
ekf_res = run_filter(ekf, noisy_arr, "EKF")
save_filter_csv(EKF_OUT, ekf_res)

# Sample results -- joint 0 (pelvis)
x = lkf.get_full_state()
print("\n[LKF] Joint 0 (pelvis), final frame:")
print(f"  Position:     ({x[0]:.4f}, {x[4]:.4f}, {x[8]:.4f}) m")
print(f"  Velocity:     ({x[1]:.4f}, {x[5]:.4f}, {x[9]:.4f}) m/s")
print(f"  Acceleration: ({x[2]:.4f}, {x[6]:.4f}, {x[10]:.4f}) m/s2")
print(f"  Jerk:         ({x[3]:.4f}, {x[7]:.4f}, {x[11]:.4f}) m/s3")

x     = ekf.get_full_state()
h_out = ekf_compute_h(x)
print("\n[EKF] Joint 0 (pelvis), final frame:")
print(f"  Position: ({x[0]:.4f}, {x[4]:.4f}, {x[8]:.4f}) m")
print(f"  h(x): r={h_out[0]:.4f} m  theta={h_out[1]:.4f} rad  phi={h_out[2]:.4f} rad")

# ==============================================================================
# PLOTTING
# ==============================================================================

lkf_pos_arr = load_filter_csv(LKF_OUT)
ekf_pos_arr = load_filter_csv(EKF_OUT)

N          = min(lkf_res.shape[0], ekf_res.shape[0],
                 noisy_arr.shape[0], true_arr.shape[0])
t          = np.arange(N) * DT
j          = JOINT_IDX
JOINT_NAME = JOINT_NAMES[j]


def get_axis(results: np.ndarray, joint: int, ax_idx: int):
    """Extract (pos, vel, acc, jerk) for a given joint/axis."""
    b = joint * STATE_DIM + ax_idx * 4
    return results[:N, b], results[:N, b+1], results[:N, b+2], results[:N, b+3]


lkf_px, lkf_vx, lkf_ax_, lkf_jx = get_axis(lkf_res, j, 0)
lkf_py, lkf_vy, lkf_ay_, lkf_jy = get_axis(lkf_res, j, 1)
lkf_pz, lkf_vz, lkf_az_, lkf_jz = get_axis(lkf_res, j, 2)

ekf_px, ekf_vx, ekf_ax_, ekf_jx = get_axis(ekf_res, j, 0)
ekf_py, ekf_vy, ekf_ay_, ekf_jy = get_axis(ekf_res, j, 1)
ekf_pz, ekf_vz, ekf_az_, ekf_jz = get_axis(ekf_res, j, 2)

AXES_LABELS = ["X", "Y", "Z"]
print(f"\n-- Plotting {N} frames | Joint {j}: '{JOINT_NAME}' ----------------")

# -- Plot A: LKF State Time-Series (4 x 3 grid) --------------------------------
STATE_DATA_LKF = {
    "Position (m)":        ([lkf_px, lkf_py, lkf_pz],         "tab:blue"),
    "Velocity (m/s)":      ([lkf_vx, lkf_vy, lkf_vz],         "tab:orange"),
    "Acceleration (m/s2)": ([lkf_ax_, lkf_ay_, lkf_az_],       "tab:green"),
    "Jerk (m/s3)":         ([lkf_jx, lkf_jy, lkf_jz],         "tab:red"),
}
fig, axes = plt.subplots(4, 3, figsize=(16, 14), sharex=True)
fig.suptitle(f"LKF State Estimates -- Joint {j}: {JOINT_NAME}",
             fontsize=15, fontweight="bold", y=1.01)
for row_idx, (ylabel, (series, color)) in enumerate(STATE_DATA_LKF.items()):
    for col_idx, (sig, axis_lbl) in enumerate(zip(series, AXES_LABELS)):
        ax = axes[row_idx][col_idx]
        ax.plot(t, sig, color=color, linewidth=1.2)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(True, alpha=0.3)
        if row_idx == 0:
            ax.set_title(f"{axis_lbl}-axis", fontsize=10, fontweight="bold")
        if row_idx == 3:
            ax.set_xlabel("Time (s)", fontsize=8)
plt.tight_layout()
plt.savefig("lkf_state_timeseries.png", dpi=150, bbox_inches="tight")
plt.show()
print("\u2705 Saved: lkf_state_timeseries.png")

# -- Plot B: True vs Noisy vs LKF vs EKF Position Comparison ------------------
true_pos  = true_arr[:N, j, :]
noisy_pos = noisy_arr[:N, j, :]
lkf_pos   = np.column_stack([lkf_px, lkf_py, lkf_pz])
ekf_pos   = np.column_stack([ekf_px, ekf_py, ekf_pz])

fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
fig.suptitle(f"Position Comparison -- Joint {j}: {JOINT_NAME}",
             fontsize=14, fontweight="bold")
for i, lbl in enumerate(["X", "Y", "Z"]):
    ax = axes[i]
    ax.plot(t, true_pos[:, i],  color="black",     lw=1.5,            label="True",         zorder=4)
    ax.plot(t, noisy_pos[:, i], color="salmon",    lw=0.8, alpha=0.6, label="Noisy",        zorder=1)
    ax.plot(t, lkf_pos[:, i],   color="tab:blue",  lw=1.5, ls="--",   label="LKF estimate", zorder=3)
    ax.plot(t, ekf_pos[:, i],   color="tab:green", lw=1.5, ls=":",    label="EKF estimate", zorder=2)
    ax.set_ylabel(f"{lbl} position (m)", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
axes[-1].set_xlabel("Time (s)", fontsize=10)
plt.tight_layout()
plt.savefig("ekf_position_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("\u2705 Saved: ekf_position_comparison.png")

# -- Plot C: LKF vs EKF RMSE across all 23 joints -----------------------------
rmse_noisy = np.zeros(NUM_JOINTS)
rmse_lkf   = np.zeros(NUM_JOINTS)
rmse_ekf   = np.zeros(NUM_JOINTS)

for jj in range(NUM_JOINTS):
    b      = jj * STATE_DIM
    lkf_j  = np.column_stack([lkf_res[:N, b], lkf_res[:N, b+4], lkf_res[:N, b+8]])
    ekf_j  = np.column_stack([ekf_res[:N, b], ekf_res[:N, b+4], ekf_res[:N, b+8]])
    true_j  = true_arr[:N, jj, :]
    noisy_j = noisy_arr[:N, jj, :]
    rmse_noisy[jj] = np.sqrt(np.mean((noisy_j - true_j) ** 2))
    rmse_lkf[jj]   = np.sqrt(np.mean((lkf_j   - true_j) ** 2))
    rmse_ekf[jj]   = np.sqrt(np.mean((ekf_j   - true_j) ** 2))

x_idx = np.arange(NUM_JOINTS)
width = 0.25
fig, ax = plt.subplots(figsize=(18, 6))
ax.bar(x_idx - width, rmse_noisy, width, color="salmon",    label="Noisy RMSE",
       alpha=0.85, edgecolor="black", linewidth=0.5)
ax.bar(x_idx,         rmse_lkf,   width, color="tab:blue",  label="LKF RMSE",
       alpha=0.85, edgecolor="black", linewidth=0.5)
ax.bar(x_idx + width, rmse_ekf,   width, color="tab:green", label="EKF RMSE",
       alpha=0.85, edgecolor="black", linewidth=0.5)
ax.set_xticks(x_idx)
ax.set_xticklabels(JOINT_NAMES, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Position RMSE (m)", fontsize=11)
ax.set_title("LKF vs EKF -- Position RMSE across all 23 joints",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)

lkf_imp = (rmse_noisy.mean() - rmse_lkf.mean()) / rmse_noisy.mean() * 100
ekf_imp = (rmse_noisy.mean() - rmse_ekf.mean()) / rmse_noisy.mean() * 100
ax.text(0.98, 0.96,
        f"LKF improvement: {lkf_imp:.1f}%\nEKF improvement: {ekf_imp:.1f}%",
        transform=ax.transAxes, ha="right", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray"))
plt.tight_layout()
plt.savefig("lkf_ekf_rmse_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"\n\U0001f4ca RMSE Summary:")
print(f"   Avg Noisy RMSE : {rmse_noisy.mean():.4f} m")
print(f"   Avg LKF   RMSE : {rmse_lkf.mean():.4f} m  ({lkf_imp:.1f}% improvement)")
print(f"   Avg EKF   RMSE : {rmse_ekf.mean():.4f} m  ({ekf_imp:.1f}% improvement)")
print("\u2705 Saved: lkf_ekf_rmse_comparison.png")

# -- Plot D: Innovation residuals (z - H*x_hat) --------------------------------
innovation = noisy_pos - lkf_pos
fig, axes = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
fig.suptitle(f"Innovation Residuals (z - H*x_hat) -- Joint {j}: {JOINT_NAME}",
             fontsize=13, fontweight="bold")
for i, lbl in enumerate(["X", "Y", "Z"]):
    ax = axes[i]
    ax.plot(t, innovation[:, i], color="tab:purple", lw=0.9, alpha=0.8)
    ax.axhline(0, color="black", lw=1.2, zorder=3)
    sigma = innovation[:, i].std()
    ax.axhline( 2 * sigma, color="red", lw=1, ls="--", label="+-2 sigma")
    ax.axhline(-2 * sigma, color="red", lw=1, ls="--")
    ax.set_ylabel(f"Residual {lbl} (m)", fontsize=9)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
axes[-1].set_xlabel("Time (s)", fontsize=10)
plt.tight_layout()
plt.savefig("lkf_innovations.png", dpi=150, bbox_inches="tight")
plt.show()
print("\u2705 Saved: lkf_innovations.png")

# -- Plot E: LKF vs EKF -- velocity / acceleration / jerk ---------------------
fig, axes = plt.subplots(3, 3, figsize=(16, 10), sharex=True)
fig.suptitle(f"LKF vs EKF -- Velocity / Acceleration / Jerk\nJoint {j}: {JOINT_NAME}",
             fontsize=13, fontweight="bold")
rows_data = [
    ("Velocity (m/s)",      [lkf_vx,  lkf_vy,  lkf_vz],  [ekf_vx,  ekf_vy,  ekf_vz]),
    ("Acceleration (m/s2)", [lkf_ax_, lkf_ay_, lkf_az_], [ekf_ax_, ekf_ay_, ekf_az_]),
    ("Jerk (m/s3)",         [lkf_jx,  lkf_jy,  lkf_jz],  [ekf_jx,  ekf_jy,  ekf_jz]),
]
for row_idx, (ylabel, lkf_series, ekf_series) in enumerate(rows_data):
    for col_idx, axis_lbl in enumerate(AXES_LABELS):
        ax = axes[row_idx][col_idx]
        ax.plot(t, lkf_series[col_idx], color="tab:blue",  lw=1.2, label="LKF")
        ax.plot(t, ekf_series[col_idx], color="tab:green", lw=1.2, ls="--", label="EKF")
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(True, alpha=0.3)
        if row_idx == 0:
            ax.set_title(f"{axis_lbl}-axis", fontsize=10, fontweight="bold")
        if row_idx == 2:
            ax.set_xlabel("Time (s)", fontsize=8)
        if col_idx == 2:
            ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig("lkf_ekf_state_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("\u2705 Saved: lkf_ekf_state_comparison.png")

# ==============================================================================
# 3D WALKING ANIMATION
# ==============================================================================

frames_idx = list(range(0, min(N, MAX_FRAMES), FRAME_STEP))
n_anim     = len(frames_idx)
print(f"\n-- Rendering animation ({n_anim} frames) -- please wait... --------")

all_pos = np.concatenate([noisy_arr[:N], lkf_pos_arr[:N], ekf_pos_arr[:N]], axis=0)
pad     = 0.3
x_min, x_max = all_pos[:, :, 0].min() - pad, all_pos[:, :, 0].max() + pad
y_min, y_max = all_pos[:, :, 1].min() - pad, all_pos[:, :, 1].max() + pad
z_min, z_max = all_pos[:, :, 2].min() - pad, all_pos[:, :, 2].max() + pad


def draw_skeleton(ax, positions, joint_color, bone_color):
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
               c=joint_color, s=20, depthshade=True, zorder=5)
    for (ii, kk) in BONES:
        ax.plot([positions[ii, 0], positions[kk, 0]],
                [positions[ii, 1], positions[kk, 1]],
                [positions[ii, 2], positions[kk, 2]],
                color=bone_color, lw=1.8)


def setup_ax3d(ax, title):
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_xlabel("X (m)", fontsize=7)
    ax.set_ylabel("Y (m)", fontsize=7)
    ax.set_zlabel("Z (m)", fontsize=7)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.tick_params(labelsize=6)
    ax.view_init(elev=15, azim=-60)


fig3d = plt.figure(figsize=(18, 7))
fig3d.patch.set_facecolor("#1a1a2e")
ax1 = fig3d.add_subplot(131, projection="3d")
ax2 = fig3d.add_subplot(132, projection="3d")
ax3 = fig3d.add_subplot(133, projection="3d")

for _ax in [ax1, ax2, ax3]:
    _ax.set_facecolor("#1a1a2e")
    _ax.xaxis.pane.fill = False
    _ax.yaxis.pane.fill = False
    _ax.zaxis.pane.fill = False

fig3d.suptitle("3D Full-Body Walking Animation -- Measured vs LKF vs EKF",
               fontsize=13, fontweight="bold", color="white", y=1.01)
time_text = fig3d.text(0.5, 0.97, "", ha="center", fontsize=10, color="white")


def init_anim():
    ax1.cla(); ax2.cla(); ax3.cla()
    setup_ax3d(ax1, "Measured (Noisy)")
    setup_ax3d(ax2, "LKF Estimate")
    setup_ax3d(ax3, "EKF Estimate")
    return []


def update_anim(frame_num):
    fi = frames_idx[frame_num]
    ax1.cla(); ax2.cla(); ax3.cla()
    setup_ax3d(ax1, "Measured (Noisy)")
    setup_ax3d(ax2, "LKF Estimate")
    setup_ax3d(ax3, "EKF Estimate")
    draw_skeleton(ax1, noisy_arr[fi],   joint_color="salmon",  bone_color="#ff6b6b")
    draw_skeleton(ax2, lkf_pos_arr[fi], joint_color="#74b9ff", bone_color="#0984e3")
    draw_skeleton(ax3, ekf_pos_arr[fi], joint_color="#55efc4", bone_color="#00b894")
    time_text.set_text(f"Time: {fi * DT:.2f}s  |  Frame: {fi}/{N - 1}")
    return []


anim_obj = animation.FuncAnimation(
    fig3d, update_anim, frames=n_anim,
    init_func=init_anim, interval=1000 / 30, blit=False
)
plt.tight_layout()

try:
    Writer = animation.FFMpegWriter(fps=30, bitrate=1800,
                                    extra_args=["-vcodec", "libx264"])
    anim_obj.save("walking_animation.mp4", writer=Writer, dpi=120,
                  savefig_kwargs={"facecolor": "#1a1a2e"})
    print("\u2705 Saved: walking_animation.mp4")
except Exception as e:
    print(f"[WARN] MP4 export failed (ffmpeg not installed?): {e}")
    print("       Saving as GIF instead...")
    anim_obj.save("walking_animation.gif", writer="pillow", fps=15, dpi=80)
    print("\u2705 Saved: walking_animation.gif")

plt.show()

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n\U0001f389 All done!")
print("Output CSV : lkf_results.csv | ekf_results.csv")
print("Plots      : lkf_state_timeseries.png | ekf_position_comparison.png |")
print("             lkf_ekf_rmse_comparison.png | lkf_innovations.png |")
print("             lkf_ekf_state_comparison.png")
print("Animation  : walking_animation.mp4 (or .gif)")