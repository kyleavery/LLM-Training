What features are present in the following function?

```
push ebx
add esp, 0xfffffefc
mov ebx, eax
push 0x104
lea eax, [esp + 4]
push eax
call sub.kernel32.dll_GetWindowsDirectoryA
mov edx, ebx
mov eax, esp
call fcn.00404da4
add esp, 0x104
pop ebx
ret
```