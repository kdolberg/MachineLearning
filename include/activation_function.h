#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

#include <functional>
#include "linear_algebra.h"

namespace MachineLearning {
	typedef struct {
		std::function<LinearAlgebra::scalar_t(LinearAlgebra::scalar_t)> function;
		std::function<LinearAlgebra::scalar_t(LinearAlgebra::scalar_t)> derivative;
	} ActivationFunctionStruct;

	/**
	 * @brief Class used to define a layer's activation function.
	 */
	class ActivationFunction : ActivationFunctionStruct {
	public:
		using ActivationFunctionStruct::ActivationFunctionStruct;

		/**
		 * COMMENTED OUT THESE CONSTRUCTORS.
		 * I DON'T THINK THEY ARE USED.
		 * DELETE THIS COMMENT AND THE ONE BELOW ONCE THAT IS CONFIRMED.
		 * */

		// ActivationFunction(){}
		// ActivationFunction(const ActivationFunctionStruct& afs) { // I don't think this constructor is used. Delete if so
		// 	this->function = afs.function;
		// 	this->derivative = afs.derivative;
		// }
		LinearAlgebra::scalar_t operator()(LinearAlgebra::scalar_t in) const {
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
		LinearAlgebra::scalar_t ddx(LinearAlgebra::scalar_t in) const {
			return this->derivative(in);
		}
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
	 * @brief Returns the sigmoid function
	 */
	ActivationFunction get_sigmoid();
	/**
	 * Returns the leaky ReLU function
	 */
	ActivationFunction get_leaky_ReLU();
} //MachineLearning

#endif //ACTIVATION_FUNCTION_H