#include <complex>

template <typename T> 
struct ghost_complex : public std::complex<T> 
{
		ghost_complex() : std::complex<T>(0.,0.) {}
		ghost_complex(T a) : std::complex<T>(a) {}
		ghost_complex(T a, T b) : std::complex<T>(a,b) {}
		ghost_complex(const std::complex<T> &a) : std::complex<T>(a) {}
		ghost_complex(const ghost_complex<float> &a) : std::complex<T>(a) {}
		operator float() const;
	//	operator double() const;
		ghost_complex<T> operator+(const ghost_complex<T>&) const;
		ghost_complex<T> operator*(const ghost_complex<T>&) const;
};

template <typename T>
ghost_complex<T>::operator float() const {
	return (float)(std::real(*this));
}

/*template <typename T>
ghost_complex<T>::operator double() const {
	return (double)(std::real(*this));
}*/

template <typename T>
ghost_complex<T> ghost_complex<T>::operator +(const ghost_complex<T>& c) const {
	return ghost_complex<T>(std::real(*this) + std::real(c), std::imag(*this) + std::imag(c));
}

template <typename T>
ghost_complex<T> ghost_complex<T>::operator *(const ghost_complex<T>& c) const {
	return ghost_complex<T>(
			std::real(*this)*std::real(c) - std::imag(*this)*std::imag(c), 
			std::real(*this)*std::imag(c) + std::imag(*this)*std::real(c));
}
