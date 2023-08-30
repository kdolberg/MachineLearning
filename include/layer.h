#ifndef LAYER_H
#define LAYER_H

#include "types.h"
//blah

namespace MachineLearning {
	/**
	 * @brief Defines the structure of a layer's parameters
	 */
	class LayerParams {
	public:
		LinearAlgebra::Matrix weights;
		LinearAlgebra::Matrix biases;
		LayerParams() {}
		LayerParams(uint num_inputs,uint num_outputs) : LayerParams() {
			LinearAlgebra::mindex_t i = MINDEX(num_outputs,num_inputs);
			this->weights = LinearAlgebra::Matrix(i);
			this->biases = LinearAlgebra::Matrix(MINDEX(i.row,1));
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
			return this->weights.get_num_cols();
		}
		uint get_num_outputs() const {
			assert(this->weights.get_num_rows()==this->biases.get_num_rows());
			return this->weights.get_num_rows();
		}
	}; //LayerParams

	/**
	 * @brief Defines a layer (save for the data caches used in backpropagation and forwardpropagation)
	 */
	class LayerNoCache {
	public:
		LayerParams parameters;
		ActivationFunction func;
		LayerNoCache(uint num_inputs, uint num_outputs, ActivationFunction func_,int dummy) {
			UNUSED(dummy);
			this->parameters.weights.resize(MINDEX(num_outputs,num_inputs));
			this->parameters.biases.resize(MINDEX(num_outputs,1));
			this->func = func_;
		}
		LayerNoCache(uint num_inputs, uint num_outputs, ActivationFunction func_, bool randomize) : LayerNoCache(num_inputs,num_outputs,func_,0) {
			if(randomize) {
				this->parameters.weights.randomize();
				this->parameters.biases.randomize();
			}
		}
		LayerNoCache(uint num_inputs, uint num_outputs, ActivationFunction func_) : LayerNoCache(num_inputs,num_outputs,func_,true) {}
		template <typename T>
		T operator()(const T& t) {
			return this->func(this->parameters(t));
		}
		LinearAlgebra::Matrix calc_pre_activation_function_output(const LinearAlgebra::Matrix& x_data) const {
			return this->parameters(x_data);
		}
		LinearAlgebra::Matrix calc_post_activation_function_output(const LinearAlgebra::Matrix& pre_activation_function_output) const {
			return this->func(pre_activation_function_output);
		}
	}; //LayerNoCache

	LinearAlgebra::Matrix calc_derivatives_to_pass_on(const LinearAlgebra::Matrix& derivatives,const LinearAlgebra::Matrix& weights);
	LinearAlgebra::Matrix calc_partial_derivatives(const LinearAlgebra::Matrix& derivatives,const LinearAlgebra::Matrix& pre_activation_function_output);

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
		LinearAlgebra::Matrix update_forprop_data_cache(const LinearAlgebra::Matrix& x_data) {
			this->for_data.pre_act_func_output = this->calc_pre_activation_function_output(x_data);
			this->for_data.post_act_func_output = this->calc_post_activation_function_output(this->for_data.pre_act_func_output);
			return this->for_data.post_act_func_output;
		}
		LinearAlgebra::Matrix update_backprop_data_cache(const LinearAlgebra::Matrix& from_prev_layer) {
			this->back_data.derivatives = this->func.ddx(this->for_data.pre_act_func_output);
			// this->back_data.partial_derivatives = calc_partial_derivatives(this->back_data.derivatives,)
			return calc_derivatives_to_pass_on(this->back_data.derivatives,this->parameters.weights)*from_prev_layer;
		}
		const LinearAlgebra::Matrix& get_post_act_func_output() const {
			return this->for_data.post_act_func_output;
		}
	}; //Layer
} //MachineLearning
#endif //LAYER_H