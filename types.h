#ifndef TYPES_H
#define TYPES_H

#include "linear_algebra.h"
#include "activation_function.h"

#define UNUSED(var) {(void)(var);}

namespace MachineLearning {
	typedef LinearAlgebra::uint uint;

	struct layer_params_struct {
		LinearAlgebra::Matrix weights;
		LinearAlgebra::VerticalVector biases;
	} layer_params_struct;

	/**
	 * @brief Defines the structure of a layer's parameters
	 */
	class LayerParams : public layer_params_struct {
	public:
		using MachineLearning::layer_params_struct::layer_params_struct;
		LayerParams() {}
		LayerParams(uint num_inputs,uint num_outputs) : LayerParams() {
			LinearAlgebra::mindex_t i = MINDEX(num_outputs,num_inputs);
			this->weights = LinearAlgebra::Matrix(i);
			this->biases = LinearAlgebra::VerticalVector(i.row);
		}

		template <typename T>
		T operator()(const T& in_signal) {
			assert(T::is_matrix_like());
			T ret = LinearAlgebra::matrix_multiply<T>(&(this->weights),&in_signal);
			LinearAlgebra::columnwise_add(&ret,&(this->biases));
			return ret;
		}
	};

	typedef struct {
		LinearAlgebra::Matrix x;
		LinearAlgebra::Matrix y;
	} TrainingDatasetSig;

	class Layer {
	public:
		LayerParams parameters;
		ActivationFunction func;
		Layer(uint num_inputs, uint num_outputs, ActivationFunction func_,int dummy) {
			UNUSED(dummy);
			this->parameters.weights.resize(MINDEX(num_outputs,num_inputs));
			this->parameters.biases.resize(MINDEX(num_outputs,1));
			this->func = func_;
		}
		Layer(uint num_inputs, uint num_outputs, ActivationFunction func_, bool randomize) : Layer(num_inputs,num_outputs,func_,0) {
			if(randomize) {
				this->parameters.weights.randomize();
				this->parameters.biases.randomize();
			}
		}
		Layer(uint num_inputs, uint num_outputs, ActivationFunction func_) : Layer(num_inputs,num_outputs,func_,true) {}
		template <typename T>
		T operator()(const T& t) {
			T not_the_real_return = this->parameters(t);
			return not_the_real_return;
		}
	};
} //MachineLearning

#endif //TYPES_H