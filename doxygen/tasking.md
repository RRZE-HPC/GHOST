Tasking
=======

The tasking functionality is a key feature of GHOST. It is designed to be as
general as possible and to allow the use of (a-)synchronous tasks in many
situations.
The implementation is based on a task queue and a thread pool.
There is a single task queue which holds all tasks, regardless of their
properties such as hardware affinity.

Besides, there is a pool of shepherd threads which contains all threads which may ever manage a task.
The number of threads in the pool is equal to the maximum number of concurrently executing tasks.

Affinity control of the GHOST threads is done by means of the
[hwloc](http://www.open-mpi.org/projects/hwloc/) library.

Shepherd thread lifecycle
-------------------------

A pthread is created for each thread of the thread pool in ghost_thpool_create().
Each of the threads enters the specified function as the starting routine where it runs 
in an infinite loop.
In this loop, the threads are waiting on a semaphore which counts the number
of tasks in all queues. As soon as a task gets added to a queue, the first
thread returning from `sem_wait` will have the chance to execute this task.
If, for any reason, the thread cannot execute the task (or any other task in the
queue), the above-mentioned semaphore gets increased by one and the shepherd thread
starts waiting again.
Once the task queues are empty and subject to be killed (i.e., at termination of
the application), a global variable is set to one and the threads
break out of the infinite loop.


Task types
----------

A GHOST task is of type ghost_task_t and should be initialized via ghost_task_create(), given a number of
parameters.

There are several types of tasks which can be distinguished by the value of
their ghost_task_t#flags variable. They can be combined by bitwise OR-ing the flags.

1. **GHOST_TASK_DEFAULT** identifies default tasks.
They are added to a single task queue. However, if none of the threads in this
locality domain takes the task out of the queue it may be executed by a thread
running in a different locality domain.

2. **GHOST_TASK_LD_STRICT** bounds a task to a locality domain. A
strict LD task is guaranteed to be executed only by threads running in the given
locality domain.

3. **GHOST_TASK_PRIO_HIGH** assigns high priority to a task.
It will be added to the head of the given queue(s) and thus they are probably
chosen earlier than the other tasks in the queues.

4. **GHOST_TASK_USE_PARENTS** is the flag to use in case a task is being 
added from within another task. When this flag is set, the child task can use
all cores which are actually reserved for the adding task.


