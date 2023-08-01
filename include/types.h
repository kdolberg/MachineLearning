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
	private:
		template <typename T>
		T call_op(const T& in_signal) const {
			assert(T::is_matrix_like());
			T ret = LinearAlgebra::matrix_multiply<T>(&(this->weights),&in_signal);
			LinearAlgebra::columnwise_add(&ret,&(this->biases));
			return ret;
		}
	public:
		const LinearAlgebra::Matrix& get_weights() const {
			return this->weights;
		}
		LinearAlgebra::Matrix operator()(const LinearAlgebra::Matrix& in_signal) const {
			return this->call_op<LinearAlgebra::Matrix>(in_signal);
		}
		LinearAlgebra::HorizontalVector operator()(const LinearAlgebra::HorizontalVector& in_signal) const {
			return this->call_op<LinearAlgebra::HorizontalVector>(in_signal);
		}
		LinearAlgebra::VerticalVector operator()(const LinearAlgebra::VerticalVector& in_signal) const {
			return this->call_op<LinearAlgebra::VerticalVector>(in_signal);
		}
		uint get_num_inputs() const {
			return this->weights.get_num_cols();
		}
		uint get_num_outputs() const {
			assert(this->weights.get_num_rows()==this->biases.get_num_rows());
			return this->weights.get_num_rows();
		}
	};

	typedef struct {
		LinearAlgebra::Matrix x;
		LinearAlgebra::Matrix y;
	} TrainingDatasetSig;

	class LayerNoCache {
	public:
		LayerParams parameters;
		ActivationFunction func;
	private:
		Layer(uint num_inputs, uint num_outputs, ActivationFunction func_,int dummy) {
			UNUSED(dummy);
			this->parameters.weights.resize(MINDEX(num_outputs,num_inputs));
			this->parameters.biases.resize(MINDEX(num_outputs,1));
			this->func = func_;
		}
	public:
		Layer(uint num_inputs, uint num_outputs, ActivationFunction func_, bool randomize) : Layer(num_inputs,num_outputs,func_,0) {
			if(randomize) {
				this->parameters.weights.randomize();
				this->parameters.biases.randomize();
			}
		}
		Layer(uint num_inputs, uint num_outputs, ActivationFunction func_) : Layer(num_inputs,num_outputs,func_,true) {}
		template <typename T>
		T operator()(const T& t) {
			return this->func(this->parameters(t));
		}
		Matrix calc_pre_activation_function_output(const Matrix& x_data) const {
			return this->parameters(x_data);
		}
		Matrix calc_post_activation_function_output(const Matrix& pre_activation_function_output) const {
			return this->func(pre_activation_function_output);
		}
	};
	LinearAlgebra::Matrix calc_derivatives_to_pass_on(const LinearAlgebra::Matrix& derivatives,const LinearAlgebra::Matrix& weights);
	LinearAlgebra::Matrix calc_partial_derivatives(const LinearAlgebra::Matrix& derivatives,const LinearAlgebra::Matrix& x_data);

	typedef struct {
		LinearAlgebra::Matrix pre_act_func_output;
		LinearAlgebra::Matrix post_act_func_output;
	} LayerForDataCache;

	typedef struct {
		LinearAlgebra::Matrix derivatives;
		LinearAlgebra::Matrix partial_derivatives; 
	} LayerBackDataCache;

	class Layer : public LayerNoCache {
		LayerForDataCache for_data;
		LayerBackDataCache back_data;
	public:
		using LayerNoCache::LayerNoCache;
		const LinearAlgebra::Matrix& update_forprop_data_cache(const LinearAlgebra::Matrix& x_data) {
			this->for_data.pre_act_func_output = i->calc_pre_activation_function_output(x_data);
			this->for_data.post_act_func_output = i->calc_post_activation_function_output(i->pre_act_func_output);
			return this->for_data.post_act_func_output;
		}
		const LinearAlgebra::Matrix& update_backprop_data_cache(const LinearAlgebra::Matrix& from_prev_layer) {
			this->back_data.derivatives = this->func.derivative(this->for_data.pre_act_func_output);
			this->back_data.partial_derivatives = calc_partial_derivatives(this->back_data.derivatives,)
			return calc_derivatives_to_pass_on(this->back_data.derivatives,this->parameters.weights);
		}
	}
} //MachineLearning

#endif //TYPES_H