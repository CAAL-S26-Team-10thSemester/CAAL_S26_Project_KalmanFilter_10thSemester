.section .data
msg: .string "Hello from RISC-V!\n"

.section .text
.globl _start
_start:
    li a0, 1          # stdout
    la a1, msg        # address
    li a2, 19         # length
    li a7, 64         # write syscall
    ecall
    li a0, 0          # status 0
    li a7, 93         # exit syscall
    ecall