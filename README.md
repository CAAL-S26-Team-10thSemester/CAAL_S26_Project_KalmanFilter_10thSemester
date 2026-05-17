# CAAL S26 Project – Kalman Filter

**Team Name:** Group 10
**Semester:** 10th Semester

**Team Members:**
1. Muhammad Usman – ERP: 25177
2. Ajeet Kumar – ERP: 30559
3. Sudharth Kumar – ERP: 26925
4. Faris Ejaz – ERP: 24470

**Course:** CAAL – Spring 2026
**Project:** Kalman Filter Implementation (Linear & Extended)

---

## Milestone 4: RISC-V Vector Extension (RVV 1.0)

This repository contains our **Milestone 4** implementation, which upgrades the scalar matrix operations and Kalman Filters to use the **RISC-V Vector Extension (RVV 1.0)**. 

### Directory Structure
- `./` (Root): Contains the Milestone 3 scalar implementations (`lkf_asm.s`, `ekf_asm.s`, `matrix_asm.s`) used as a numerical and performance baseline.
- `./milestone4/`: Contains the fully vectorised **Milestone 4** implementations (`lkf_vector.s`, `ekf_vector.s`, `matrix_vec.s`, `ekf_utils_vector.s`).
- `*.csv`: Human gait dataset files used for input measurements.

### Key Features
- Complete replacement of scalar inner loops with vector instructions (`vle64.v`, `vse64.v`, `vfmacc.vv`, `vfredosum.vs`, etc.).
- Strided loads (`vlse64.v`) for efficient matrix transpositions.
- **1.88× instruction count reduction** compared to the scalar baseline.
- Preserved perfect numerical parity (0 tolerance violations at $10^{-9}$ threshold).

---

## Setup Instructions

### Recommended: Docker + VS Code Dev Containers

This repository is set up to run inside a Docker container with QEMU user-mode support (with vector extensions and plugins enabled).

#### 1) Install the required tools
1. Install **Docker Desktop**.
2. Install **Visual Studio Code**.
3. In VS Code, install these extensions:
   * **Dev Containers**
   * **Docker**

#### 2) Clone the repository
Create a folder on your machine, open a terminal in that folder, and clone the project:

```bash
git clone <repo-url>
cd CAAL_S26_Project_KalmanFilter_10thSemester
git checkout milestone-4
```

#### 3) Rebuild the container
Open the cloned folder in VS Code.
* Press **Ctrl + Shift + P**
* Run **Dev Containers: Rebuild Container**

Wait for the container to finish building and reopening.

---

## Compiling and Running (Makefile)

The `Makefile` has been updated to handle both the M3 baseline and the M4 vector targets. Run these commands inside the container terminal:

### Verification Targets

Verify the shared scalar matrix library:
```bash
make verify
```

Verify the vectorised matrix library (Milestone 4):
```bash
make verify_matrix_vec
```

### Kalman Filter Targets

Run the baseline scalar LKF and EKF (Milestone 3):
```bash
make lkf
make ekf
make all     # Runs verify, lkf, ekf
```

Run the vectorised LKF and EKF (Milestone 4):
```bash
make lkf_vector
make ekf_vector
make all_vec # Runs both vector targets
```

### Performance Analysis

Compare execution speeds and accuracy between the scalar (M3) and vector (M4) implementations:
```bash
make perf
```

Profile the instruction count reduction using QEMU plugins:
```bash
make insn_count
```

Clean up all generated object files and binaries:
```bash
make clean
```
