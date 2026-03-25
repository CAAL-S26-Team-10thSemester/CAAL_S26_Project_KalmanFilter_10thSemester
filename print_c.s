.section .rodata
fmt: .string "Result: %d\n"

.section .text
.globl main
main:
    addi sp, sp, -16
    sd ra, 8(sp)
    la a0, fmt
    li a1, 42
    call printf
    ld ra, 8(sp)
    addi sp, sp, 16
    li a0, 0
    ret