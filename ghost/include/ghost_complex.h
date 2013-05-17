#include <complex>

using namespace std;
template <class T> class ghost_complex : public complex<T> 
{
	public:
		ghost_complex() : complex<T>(0.,0.) {}
		ghost_complex(T a) : complex<T>(a) {}
		ghost_complex(complex<T> a) : complex<T>(a) {}
		operator float() const;
};

template <class T>
ghost_complex<T>::operator float() const {
	return (float)this.real;
}
