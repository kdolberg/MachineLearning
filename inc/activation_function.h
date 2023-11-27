#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

#include <functional>
#include <string>
#include "types.h"

namespace MachineLearning {typedef char func_sym;}

#define SIGMOID			((MachineLearning::func_sym)'S')
#define LEAKY_RELU		((MachineLearning::func_sym)'L')
#define UNDEFINED		((MachineLearning::func_sym)'\0')

#define SIGMOID_NAME	("sigmoid")
#define LEAKY_RELU_NAME	("leaky_ReLU")
#define UNDEFINED_NAME 	("undefined")

namespace MachineLearning {
	std::string sym2name(func_sym sym);
	class ActivationFunction;

	typedef struct {
		std::function<scalar(scalar)> function;
		std::function<scalar(scalar)> derivative;
		func_sym sym;
	} ActivationFunctionStruct;

	uint unit(MachineLearning::scalar x);

	ActivationFunction sym2ActFunc(func_sym sym);

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
		friend ActivationFunction sym2ActFunc(func_sym sym);
	public:
		using ActivationFunctionStruct::ActivationFunctionStruct;

		ActivationFunction(const ActivationFunctionStruct& afs) {
			this->function = afs.function;
			this->derivative = afs.derivative;
			this->sym = afs.sym;
		}
		ActivationFunction(func_sym _sym) {
			ActivationFunction tmp = sym2ActFunc(_sym);
			this->function = tmp.function;
			this->derivative = tmp.derivative;
			this->sym = tmp.sym;
		}
		ActivationFunction(const func_sym * sym) : ActivationFunction(*sym) {}
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
		std::string str() const;
		func_sym get_sym() const;
		bool operator==(const MachineLearning::ActivationFunction&) const;
	}; // ActivationFunction
} //MachineLearning

std::ostream& operator<<(std::ostream& os, const MachineLearning::ActivationFunction& af);
bool operator==(const MachineLearning::ActivationFunction&,const MachineLearning::ActivationFunction&);

#endif //ACTIVATION_FUNCTION_H