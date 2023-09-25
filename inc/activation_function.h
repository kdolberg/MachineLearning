#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

#include <functional>
#include "types.h"

namespace MachineLearning {
	typedef struct {
		std::function<scalar(scalar)> function;
		std::function<scalar(scalar)> derivative;
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
		scalar operator()(scalar in) const {
			return this->function(in);
		}
		template <LinearAlgebra::MATRIXLIKE T>
		T operator()(const T& t) const {
			assert(T::is_matrix_like());
			T ret(t.matrix_size());
			for (MachineLearning::mindex i = {0,0}; i.row < t.get_num_rows(); ++i.row) {
				for (i.col = 0; i.col < t.get_num_cols(); ++i.col) {
					ret[i] = this->function(t[i]);
				}
			}
			return ret;
		}
		scalar ddx(scalar in) const {
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
			for (MachineLearning::mindex i = {0,0}; i.row < t.get_num_rows(); ++i.row) {
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