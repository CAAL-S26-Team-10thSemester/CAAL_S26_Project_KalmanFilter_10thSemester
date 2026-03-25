# CAAL S26 Project – Kalman Filter

**Team Name:** 10th Semester

**Team Members:**

1. Muhammad Usman – ERP: 25177
2. Sudharth Kumar – ERP: 26925
3. Faris Ejaz – ERP: 24470
4. Ajeet Kumar – ERP: 30559

**Course:** CAAL – Spring 2026
**Project:** Kalman Filter Implementation

---

## Setup Instructions

### Recommended: Docker + VS Code Dev Containers

This repository is set up to run inside a Docker container with QEMU user-mode support and QEMU plugins enabled.

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
```

If you are using the milestone branch, make sure you switch to it first:

```bash
git checkout milestone-3
```

#### 3) Open the project in VS Code

Open the cloned folder in VS Code.

#### 4) Rebuild the container

In VS Code:

* Press **Ctrl + Shift + P**
* Run **Dev Containers: Rebuild Container**

Wait for the container to finish building and reopening.

---

## Verify the setup

Inside the container terminal, run:

```bash
make hello
make run-hello
```

Expected output:

```text
Hello from RISC-V
```

You can also verify plugin support:

```bash
make plugin-check
make run-plugin
```

Expected output includes:

```text
✅ Running with plugin: /usr/local/lib/qemu/plugins/libinsn.so
Hello from RISC-V
cpu 0 insns: ...
total insns: ...
```

To test other plugins, run:

```bash
make run-plugin PLUGIN=libbb.so
make run-plugin PLUGIN=libmem.so
make run-plugin PLUGIN=libsyscall.so
```

---

## Additional example targets

```bash
make vector
make run-vector
make print_c
make run-print_c
```



---

## Notes

* The project uses a QEMU build with plugin support enabled.

* Built plugin shared libraries are installed into:

  ```bash
  /usr/local/lib/qemu/plugins
  ```

* The default plugin used by the Makefile is:

  ```bash
  libinsn.so
  ```
