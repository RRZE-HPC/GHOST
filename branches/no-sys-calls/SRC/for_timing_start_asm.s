.intel_syntax noprefix
.data
.align 16
.text
.globl for_timing_start_asm_
.type for_timing_start_asm_, @function
for_timing_start_asm_ :

..B1.1:
	 push rax 
	 push rbx
	 push rcx
	 push rdx
	 xor rax, rax
	 cpuid
	 rdtsc
	 shl rdx, 32
	 add rax, rdx
         mov [rdi], rax
         pop rdx
	 pop rcx
	 pop rbx
	 pop rax
	 ret

.size for_timing_start_asm_, .-for_timing_start_asm_
