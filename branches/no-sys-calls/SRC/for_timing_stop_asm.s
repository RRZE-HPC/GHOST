.intel_syntax noprefix
.data
.align 16
.text
.globl for_timing_stop_asm_ 
.type for_timing_stop_asm_, @function
for_timing_stop_asm_ :

..B1.1:
	 mov r8, [rdi]

	 push rax 
	 push rbx
	 push rcx
	 push rdx
	 xor rax, rax
	 cpuid
	 rdtsc
	 shl rdx, 32
	 add rax, rdx
         sub rax, r8
         pop rdx
	 pop rcx
	 pop rbx
	 mov [rsi], rax
	 pop rax 

	 
	 ret

.size for_timing_stop_asm_, .-for_timing_stop_asm_
