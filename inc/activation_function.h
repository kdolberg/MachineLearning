#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

#include <functional>
#include "types.h"

namespace MachineLearning {
	typedef struct {
		std::function<scalar_t(scalar_t)> function;
		std::function<scalar_t(scalar_t)> derivative;
	} ActivationFunctionStruct;

	/**
	 * @brief Class used to define a layer's activation function.
	 */
	class ActivationFunction : ActivationFunctionStruct {
	public:
		using ActivationFunctionStruct::ActivationFunctionStruct;

		ActivationFunction(const ActivationFunctionStruct& afs) {
			this->function = afs.function;
			this->derivative = afs.derivative;
		}
		scalar_t operator()(scalar_t in) const {
			return this->function(in);
		}
		template <LinearAlgebra::MATRIXLIKE T>
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
		scalar_t ddx(scalar_t in) const {
			return this->derivative(in);
		}
		/**
		 * @brief Calculates (d/dt){THIS_FUNCTION(t)}
		 * @param t Any matrix-like object
		 * @return 
		 */
		template <LinearAlgebra::MATRIXLIKE T>
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
	/**
	 * @brief Returns the sigmoid ActivationFunction
	 * @return ____ ActivationFunction
	 */
	ActivationFunction get_sigmoid();
	/**
	 * Returns the leaky ReLU ActivationFunction
	 * @return ____ ActivationFunction
	 */
	ActivationFunction get_leaky_ReLU();
} //MachineLearning

#endif //ACTIVATION_FUNCTION_H