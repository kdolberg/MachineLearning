#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

#include <functional>
#include "linear_algebra.h"

namespace MachineLearning {
	typedef struct {
		std::function<LinearAlgebra::scalar_t(LinearAlgebra::scalar_t)> function;
		std::function<LinearAlgebra::scalar_t(LinearAlgebra::scalar_t)> derivative;
	} ActivationFunctionStruct;

	class ActivationFunction : ActivationFunctionStruct {
	public:
		using ActivationFunctionStruct::ActivationFunctionStruct;
		ActivationFunction(){}
		ActivationFunction(const ActivationFunctionStruct& afs) {
			this->function = afs.function;
			this->derivative = afs.derivative;
		}
		LinearAlgebra::scalar_t operator()(LinearAlgebra::scalar_t in) const {
			return this->function(in);
		}
		template <typename T>
		T operator()(const T& t) const {
			assert(T::is_matrix_like());
			T ret(t.matrix_size());
			for (LinearAlgebra::mindex_t i = {0,0}; i.row < t.get_num_rows(); ++i.row) {
				for (i.col = 0; i.col < t.get_num_cols(); ++i.col) {
					ret[i] = this->function(t[i]);
				}
			}
			return ret;
		}
		LinearAlgebra::scalar_t ddx(LinearAlgebra::scalar_t in) const {
			return this->derivative(in);
		}
		template <typename T>
		T ddx(const T& t) const {
			assert(T::is_matrix_like());
			T ret(t.matrix_size());
			for (LinearAlgebra::mindex_t i = {0,0}; i.row < t.get_num_rows(); ++i.row) {
				for (i.col = 0; i.col < t.get_num_cols(); ++i.col) {
					ret[i] = this->derivative(t[i]);
				}
			}
			return ret;
		}
	};
	ActivationFunction get_sigmoid();
	ActivationFunction get_leaky_ReLU();
}

#endif //ACTIVATION_FUNCTION_H