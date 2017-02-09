Heterogeneous execution
=======================

GHOST is capable of running on different compute architectures at the same time.
Distinction between architectures is done on a per-process base.
Each GHOST process has set a type of #ghost_type which can be either ::GHOST_TYPE_WORK for processes which work themselves or ::GHOST_TYPE_CUDA for processes which drive a CUDA GPU as an accelerator.
The type can either be set by means of the environment variable GHOST_TYPE (like `GHOST_TYPE=CUDA ./a.out`) or via the function ghost_type_set().

Type identification
-------------------

If the type has not been set, GHOST applies some heuristics in order to identify a senseful type automatically.
For example, on an heterogeneous compute node with two CPU sockets and two CUDA GPUs, the first process will be of ::GHOST_TYPE_WORK, covering the host CPU.
If two more processes get launched on this node, they will get assigned the type ::GHOST_TYPE_CUDA and one of the GPUs each.
For management purposes, both of those processes will get an exclusive CPU core on the socket which is closest to the respective GPU.
At this point, all of the resources on the node are used.
A further process on this node will be of type ::GHOST_TYPE_WORK.
In this case, the CPU resources will get divided equally between the two CPU processes.
I.e., each of those processes runs on a single socket.
In many cases, this is a sensible usage model as it lowers the danger of NUMA problems.
So, a rule of thumb for the number of processes to start on a heterogeneous node is the number of GPUs plus the number of CPU sockets.


Data locality
-------------

Sparse or dense matrices may reside in the host and/or device memory.
A #ghost_sparsemat is stored exclusively either on the host or device.
This location is decided at creation time depending on the type.

If a process is of ::GHOST_TYPE_WORK, the data of a #ghost_densemat resides on host memory only.
On the other side, if a process is of ::GHOST_TYPE_CUDA, a #ghost_densemat will be allocated on the device if not specified otherwise.
For easy exchange of densemat data between host and device, densemats will automatically get duplicated to the host/device memory if ghost_densemat_upload() or ghost_densemat_download() get called and ::GHOST_DENSEMAT_NOT_RELOCATE is not set in the densemat traits.

If a densemat, which is the result of any numerical operation, is present on both the host and the device, the computation will be carried out on the device per default. This behavior can be changed by setting the ghost_densemat_traits::compute_at field.

