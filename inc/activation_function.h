#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

#include <functional>
#include <string>
#include "types.h"

namespace MachineLearning {
	class ActivationFunction;

	typedef struct {
		std::function<scalar(scalar)> function;
		std::function<scalar(scalar)> derivative;
	} ActivationFunctionStruct;

	/**
	 * @brief Returns the sigmoid ActivationFunction
	 * @return sigmoid ActivationFunction
	 */
	ActivationFunction get_sigmoid();
	/**
	 * Returns the leaky ReLU ActivationFunction
	 * @return leaky ReLU ActivationFunction
	 */
	ActivationFunction get_leaky_ReLU();
	/**
	 * @brief Class used to define a layer's activation function.
	 */

	class ActivationFunction : ActivationFunctionStruct {
		friend ActivationFunction get_leaky_ReLU();
		friend ActivationFunction get_sigmoid();
		std::string name;
	public:
		using ActivationFunctionStruct::ActivationFunctionStruct;

		ActivationFunction(const ActivationFunctionStruct& afs) {
			this->function = afs.function;
			this->derivative = afs.derivative;
		}
		scalar operator()(scalar in) const;

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
		scalar ddx(scalar in) const;
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
		const std::string& str() const;
	}; // ActivationFunction
} //MachineLearning

std::ostream& operator<<(std::ostream& os, const MachineLearning::ActivationFunction& af);

#endif //ACTIVATION_FUNCTION_H