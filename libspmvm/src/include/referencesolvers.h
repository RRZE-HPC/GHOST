#ifndef _FORTRANFUNCTION_H_
#define _FORTRANFUNCTION_H_

void fortrancrsaxpyc_(int *, int *, mat_data_t *, mat_data_t *, mat_data_t *, int *, int *);
void fortrancrsaxpy_(int *, int *, mat_data_t *, mat_data_t *, mat_data_t *, int *, int *);
void fortrancrsaxpycf_(int *, int *, mat_data_t *, mat_data_t *, mat_data_t *, int *, int *);
void fortrancrsaxpyf_(int *, int *, mat_data_t *, mat_data_t *, mat_data_t *, int *, int *);
void fortrancrsc_(int *, int *, mat_data_t *, mat_data_t *, mat_data_t *, int *, int *);
void fortrancrs_(int *, int *, mat_data_t *, mat_data_t *, mat_data_t *, int *, int *);
void fortrancrscf_(int *, int *, mat_data_t *, mat_data_t *, mat_data_t *, int *, int *);
void fortrancrsf_(int *, int *, mat_data_t *, mat_data_t *, mat_data_t *, int *, int *);

#endif
