AS=riscv64-linux-gnu-as
LD=riscv64-linux-gnu-ld
GCC=riscv64-linux-gnu-gcc
GDB=gdb-multiarch

QEMU=/usr/local/bin/qemu-riscv64

PLUGIN ?= libinsn.so
PLUGIN_DIR ?= /usr/local/lib/qemu/plugins
PLUGIN_PATH ?= $(PLUGIN_DIR)/$(PLUGIN)

.PHONY: hello run-hello debug-hello vector run-vector print_c run-print_c plugin-check run-plugin clean

hello: hello.s
	$(AS) -o hello.o hello.s
	$(LD) -o hello hello.o

run-hello: hello
	$(QEMU) ./hello

debug-hello: hello
	$(QEMU) -g 1234 ./hello

vector: vector.s
	$(AS) -march=rv64gcv -o vector.o vector.s
	$(LD) -o vector vector.o

run-vector: vector
	$(QEMU) -cpu rv64,v=true,vlen=128 ./vector

print_c: print_c.s
	$(GCC) -o print_c print_c.s

run-print_c: print_c
	$(QEMU) ./print_c

plugin-check:
	$(QEMU) --help | grep plugin

run-plugin: hello
	@if [ ! -f "$(PLUGIN_PATH)" ]; then \
		echo "❌ Plugin not found at $(PLUGIN_PATH)"; \
		exit 1; \
	fi
	@echo "✅ Running with plugin: $(PLUGIN_PATH)"
	@$(QEMU) -d plugin -plugin "$(PLUGIN_PATH)" -D insn.log ./hello

clean:
	rm -f *.o hello vector print_c insn.log