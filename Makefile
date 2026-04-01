# ─────────────────────────────────────────────
# Toolchain
# ─────────────────────────────────────────────
AS   = riscv64-linux-gnu-as
LD   = riscv64-linux-gnu-ld
GCC  = riscv64-linux-gnu-gcc
GDB  = gdb-multiarch
QEMU = /usr/local/bin/qemu-riscv64

# ─────────────────────────────────────────────
# Plugin Configuration
# ─────────────────────────────────────────────
PLUGIN ?= libinsn.so

# Preferred installed location
PLUGIN_SYS ?= /usr/local/lib/qemu/plugins/$(PLUGIN)

# Fallback (build tree — what worked during setup)
PLUGIN_BUILD ?= build/tests/plugin/$(PLUGIN)

# Auto-detect plugin path
PLUGIN_PATH := $(shell \
	if [ -f $(PLUGIN_SYS) ]; then echo $(PLUGIN_SYS); \
	elif [ -f $(PLUGIN_BUILD) ]; then echo $(PLUGIN_BUILD); \
	else echo "NOT_FOUND"; fi)

# ─────────────────────────────────────────────
# Targets
# ─────────────────────────────────────────────
.PHONY: hello run-hello debug-hello \
        vector run-vector \
        print_c run-print_c \
        plugin-check run-plugin \
        clean

# ─────────────────────────────────────────────
# Assembly Program
# ─────────────────────────────────────────────
hello: hello.s
	$(AS) -o hello.o hello.s
	$(LD) -o hello hello.o

run-hello: hello
	$(QEMU) ./hello

debug-hello: hello
	$(QEMU) -g 1234 ./hello


# ─────────────────────────────────────────────
# Vector Program
# ─────────────────────────────────────────────
vector: vector.s
	$(AS) -march=rv64gcv -o vector.o vector.s
	$(LD) -o vector vector.o

run-vector: vector
	$(QEMU) -cpu rv64,v=true,vlen=128 ./vector


# ─────────────────────────────────────────────
# C Program (via GCC)
# ─────────────────────────────────────────────
print_c: print_c.s
	$(GCC) -o print_c print_c.s

run-print_c: print_c
	$(QEMU) ./print_c


# ─────────────────────────────────────────────
# Plugin Support
# ─────────────────────────────────────────────
plugin-check:
	@echo "Checking plugin support..."
	@$(QEMU) --help | grep plugin


run-plugin: hello
	@if [ "$(PLUGIN_PATH)" = "NOT_FOUND" ]; then \
		echo "❌ Plugin not found in:"; \
		echo "   - $(PLUGIN_SYS)"; \
		echo "   - $(PLUGIN_BUILD)"; \
		echo ""; \
		echo "👉 Fix:"; \
		echo "   1. Rebuild container (Ctrl+Shift+P → Dev Containers: Rebuild Container)"; \
		echo "   2. Ensure QEMU built with --enable-plugins"; \
		exit 1; \
	fi
	@echo "✅ Using plugin: $(PLUGIN_PATH)"
	@$(QEMU) -d plugin -plugin "$(PLUGIN_PATH)" -D insn.log ./hello


# ─────────────────────────────────────────────
# Cleanup
# ─────────────────────────────────────────────
clean:
	rm -f *.o hello vector print_c insn.log