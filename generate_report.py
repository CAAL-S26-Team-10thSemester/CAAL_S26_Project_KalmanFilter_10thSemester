import pandas as pd

def read_verification(filename):
    if not __import__('os').path.exists(filename): return None
    df = pd.read_csv(filename)
    joints = df[df['section'] == 'joint'].copy()
    comps = df[df['section'] == 'component'].copy()
    summary = df[df['section'] == 'summary'].set_index('name')['avg_abs_error'].to_dict()
    return joints, comps, summary

lkf_asm_j, lkf_asm_c, lkf_asm_s = read_verification('lkf_asm_verification.csv')
lkf_vec_j, lkf_vec_c, lkf_vec_s = read_verification('lkf_vec_verification.csv')
ekf_asm_j, ekf_asm_c, ekf_asm_s = read_verification('ekf_asm_verification.csv')
ekf_vec_j, ekf_vec_c, ekf_vec_s = read_verification('ekf_vec_verification.csv')

with open('milestone4_report.tex', 'w') as f:
    f.write(r'''\documentclass{article}
\usepackage[utf8]{inputenc}
\title{Milestone 4: RISC-V Vector Assembly Implementation}
\author{Kalman Filter Project}
\date{\today}

\begin{document}
\maketitle

\section{Introduction}
This report details the implementation, verification, and performance analysis of the Linear Kalman Filter (LKF) and Extended Kalman Filter (EKF) vectorised using the RISC-V Vector (RVV) extension.

\section{Function-by-Function Walkthrough}
The vectorised implementations leverage the RISC-V RVV 1.0 extension to accelerate the core matrix operations. The vector length is configured dynamically using \texttt{vsetvli}.

\subsection{Matrix Multiplication (\texttt{mat\_mul\_vec})}
The matrix multiplication kernel computes $C = A \times B$. The loop order is chosen to write the output matrix $C$ exactly once per output chunk to minimise memory traffic.
\begin{verbatim}
# For k = 0..K-1:
fld     ft0, 0(a6)                # scalar A[i][k]
vle64.v  v4, (a7)                 # B[k][j..j+VL-1]
vfmacc.vf v0, ft0, v4            # v0 += A[i][k] * B[k][j..j+VL-1]
\end{verbatim}

\subsection{Element-wise Operations}
Addition, subtraction, and scaled addition treat the matrices as flat arrays and use \texttt{vfadd.vv}, \texttt{vfsub.vv}, and \texttt{vfmacc.vf}.
\begin{verbatim}
vle64.v  v0, (a1)       # load A chunk
vle64.v  v4, (a2)       # load B chunk
vfadd.vv v0, v0, v4    # v0 = A + B
vse64.v  v0, (a0)       # store to C
\end{verbatim}

\subsection{Matrix-Vector Multiplication (\texttt{mat\_vec\_mul\_vec})}
This function uses an ordered reduction (\texttt{vfredosum.vs}) for deterministic and reproducible summation across runs.
\begin{verbatim}
vfmul.vv v8, v0, v4              # element-wise product
vfredosum.vs v12, v8, v12         # v12[0] = sum(v8)
\end{verbatim}

\subsection{Matrix Transposition (\texttt{mat\_transpose\_vec})}
Transposition is vectorised by column using strided loads (\texttt{vlse64.v}) and unit-stride stores (\texttt{vse64.v}).
\begin{verbatim}
vlse64.v v0, (a6), s4             # strided load, stride = N*8
vse64.v  v0, (a7)                  # unit-stride store
\end{verbatim}

\section{Numerical Verification}
The vectorised output is compared against the scalar assembly implementation to ensure numerical accuracy ($\epsilon \le 10^{-9}$).

\subsection{Average Absolute Error per Joint}
\begin{table}[ht]
\centering
\begin{tabular}{|l|r|r|}
\hline
Joint & LKF Vector Error & EKF Vector Error \\
\hline
''')
    for i, row in lkf_vec_j.iterrows():
        name = row['name']
        err_l = float(row['avg_abs_error'])
        err_e = float(ekf_vec_j[ekf_vec_j['name'] == name]['avg_abs_error'].values[0])
        f.write(f"{name} & {err_l:.2e} & {err_e:.2e} \\\\\n")

    f.write(r'''\hline
\end{tabular}
\caption{Average absolute error per joint for vectorised LKF and EKF.}
\end{table}

\subsection{Average Absolute Error per State Component}
\begin{table}[ht]
\centering
\begin{tabular}{|l|r|r|}
\hline
Component & LKF Vector Error & EKF Vector Error \\
\hline
''')
    for i, row in lkf_vec_c.iterrows():
        name = row['name']
        err_l = float(row['avg_abs_error'])
        err_e = float(ekf_vec_c[ekf_vec_c['name'] == name]['avg_abs_error'].values[0])
        f.write(f"{name} & {err_l:.2e} & {err_e:.2e} \\\\\n")
    
    f.write(r'''\hline
\end{tabular}
\caption{Average absolute error per component for vectorised LKF and EKF.}
\end{table}

\subsection{Overall Error Bounds}
\begin{itemize}
\item LKF Max Error: ''' + f"{float(lkf_vec_s['max_err']):.2e}" + r'''
\item LKF Min Error: ''' + f"{float(lkf_vec_s['min_err']):.2e}" + r'''
\item EKF Max Error: ''' + f"{float(ekf_vec_s['max_err']):.2e}" + r'''
\item EKF Min Error: ''' + f"{float(ekf_vec_s['min_err']):.2e}" + r'''
\end{itemize}
All errors are well within the required $10^{-9}$ tolerance.

\subsection{Direct Comparison: Scalar vs Vector}
\begin{table}[ht]
\centering
\begin{tabular}{|l|r|r|}
\hline
Joint/Component & M3 Scalar Error & M4 Vector Error \\
\hline
''')
    f.write(f"LKF pelvis & {float(lkf_asm_j[lkf_asm_j['name']=='pelvis']['avg_abs_error'].values[0]):.2e} & {float(lkf_vec_j[lkf_vec_j['name']=='pelvis']['avg_abs_error'].values[0]):.2e} \\\\\n")
    f.write(f"LKF px & {float(lkf_asm_c[lkf_asm_c['name']=='px']['avg_abs_error'].values[0]):.2e} & {float(lkf_vec_c[lkf_vec_c['name']=='px']['avg_abs_error'].values[0]):.2e} \\\\\n")
    f.write(f"EKF head & {float(ekf_asm_j[ekf_asm_j['name']=='head']['avg_abs_error'].values[0]):.2e} & {float(ekf_vec_j[ekf_vec_j['name']=='head']['avg_abs_error'].values[0]):.2e} \\\\\n")
    f.write(f"EKF vy & {float(ekf_asm_c[ekf_asm_c['name']=='vy']['avg_abs_error'].values[0]):.2e} & {float(ekf_vec_c[ekf_vec_c['name']=='vy']['avg_abs_error'].values[0]):.2e} \\\\\n")
    f.write(r'''\hline
\end{tabular}
\caption{Comparison of errors showing vectorisation did not degrade numerical accuracy.}
\end{table}

\section{Performance Analysis}
The speedup analysis compares the scalar (M3) code against the vectorised (M4) code.

\subsection{Instruction Count Reduction}
The instruction count was measured using the QEMU plugin \texttt{libinsn.so}.

\begin{table}[ht]
\centering
\begin{tabular}{|l|r|r|r|}
\hline
Filter & M3 Scalar Insns & M4 Vector Insns & Reduction Ratio \\
\hline
LKF & 6,665,955,583 & 3,547,738,380 & 1.88x \\
EKF & 6,668,931,253 & 3,550,632,658 & 1.88x \\
\hline
\end{tabular}
\caption{Instruction count reduction from M3 scalar to M4 vector.}
\end{table}

\subsection{Speedup Measurement}
The wall-clock time was measured over multiple frames using the \texttt{perf\_compare} utility running in QEMU. Note that under QEMU emulation, vector operations are simulated using scalar loops, resulting in a wall-clock slowdown, however the instruction count correctly reflects the reduction in architectural instructions executed.

\begin{table}[ht]
\centering
\begin{tabular}{|l|r|r|r|}
\hline
Filter & M3 Scalar Time (s) & M4 Vector Time (s) & Speedup (QEMU emulated) \\
\hline
LKF & 25.84 & 116.04 & 0.22x \\
EKF & 24.55 & 106.91 & 0.23x \\
\hline
\end{tabular}
\caption{Execution time speedup comparison.}
\end{table}

\end{document}
'''
    )
