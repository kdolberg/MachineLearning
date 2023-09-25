#ifndef LAYER_H
#define LAYER_H

#include <list>
#include "types.h"
#include "activation_function.h"
#include "confirm.h"

#define COLUMNS_IN_BASE_MATRIX 1

namespace NetTest {
	class PrivateAPI;
}

namespace MachineLearning {

	class Net;
	/**
	 * @brief Defines the structure of a layer's parameters
	 */
	class LayerParams {
		friend MachineLearning::Net;
		friend NetTest::PrivateAPI;
	public:
		LinearAlgebra::Matrix weights;
		LinearAlgebra::Matrix biases;
		LayerParams() {}
		LayerParams(uint num_inputs,uint num_outputs) : LayerParams() {
			LinearAlgebra::mindex_t weight_matrix_dims = MINDEX(num_outputs,num_inputs);
			this->weights = LinearAlgebra::Matrix(weight_matrix_dims);
			this->biases = LinearAlgebra::Matrix(MINDEX(weight_matrix_dims.row,COLUMNS_IN_BASE_MATRIX));
		}
		LayerParams(const LinearAlgebra::Matrix& weights, const LinearAlgebra::Matrix& biases) {
			CONFIRM(weights.size().row==biases.size().row);
			CONFIRM(biases.size().col==1);
			this->weights = weights;
			this->biases = biases;
		}
	private:
		template <typename T>
		T call_op(const T& in_signal) const {
			assert(T::is_matrix_like());
			T ret = LinearAlgebra::matrix_multiply<T>(&(this->weights),&in_signal);
			LinearAlgebra::columnwise_add(&ret,&(this->biases));
			return ret;
		}
	public:
		void randomize() {
			this->weights.randomize();
			this->biases.randomize();
		}
		const LinearAlgebra::Matrix& get_weights() const {
			return this->weights;
		}
		const LinearAlgebra::Matrix& get_biases() const {
			return this->biases;
		}
		/**
		 * @brief 
		 */
		LinearAlgebra::Matrix operator()(const LinearAlgebra::Matrix& in_signal) const {
			return this->call_op<LinearAlgebra::Matrix>(in_signal);
		}
		LinearAlgebra::HorizontalVector operator()(const LinearAlgebra::HorizontalVector& in_signal) const {
			return this->call_op<LinearAlgebra::HorizontalVector>(in_signal);
		}
		LinearAlgebra::VerticalVector operator()(const LinearAlgebra::VerticalVector& in_signal) const {
			return this->call_op<LinearAlgebra::VerticalVector>(in_signal);
		}
		/**
		 * @brief Returns the number of inputs for this layer
		 * @return The number of inputs for this layer (AKA the number of column in the matrix of weights)
		 */
		uint get_num_inputs() const {
			return this->get_weights().get_num_cols();
		}
		uint get_num_outputs() const {
			assert(this->get_weights().get_num_rows()==this->get_biases().get_num_rows());
			return this->get_weights().get_num_rows();
		}
		MachineLearning::LayerParams& operator+=(const MachineLearning::LayerParams& b) {
			this->weights += b.weights;
			this->biases += b.biases;
			return (*this);
		}
		bool operator==(const LayerParams& ls) const {
			return (this->get_weights()==ls.get_weights()) && (this->get_biases()==ls.get_biases());
		}
	}; //LayerParams

} //MachineLearning

std::ostream& operator<<(std::ostream& os,const MachineLearning::LayerParams& lp);

MachineLearning::LayerParams operator-(const MachineLearning::LayerParams& a, const MachineLearning::LayerParams& b);

MachineLearning::LayerParams operator/(const MachineLearning::LayerParams& lp, LinearAlgebra::uint u);

MachineLearning::LayerParams operator*(const MachineLearning::LayerParams& lp, MachineLearning::scalar u);

#endif //LAYER_H