#ifndef LAYER_H
#define LAYER_H

#include "types.h"

#define COLUMNS_IN_BASE_MATRIX 1

namespace MachineLearning {
	/**
	 * @brief Defines the structure of a layer's parameters
	 */
	class LayerParams {
	private:
		LinearAlgebra::Matrix weights;
		LinearAlgebra::Matrix biases;
	public:
		LayerParams() {}
		LayerParams(uint num_inputs,uint num_outputs) : LayerParams() {
			LinearAlgebra::mindex_t weight_matrix_dims = MINDEX(num_outputs,num_inputs);
			this->weights = LinearAlgebra::Matrix(weight_matrix_dims);
			this->biases = LinearAlgebra::Matrix(MINDEX(weight_matrix_dims.row,COLUMNS_IN_BASE_MATRIX));
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
	}; //LayerParams

	/**
	 * @brief Defines a layer (save for the data caches used in backpropagation and forwardpropagation)
	 */
	class LayerNoCache : public MachineLearning::LayerParams {
	public:
		using LayerParams::LayerParams;
		using LayerParams::operator();
		ActivationFunction func;
		LayerNoCache(uint num_inputs, uint num_outputs, ActivationFunction func_,int dummy) : LayerParams(num_inputs,num_outputs) {
			UNUSED(dummy);
			this->func = func_;
		}
		LayerNoCache(uint num_inputs, uint num_outputs, ActivationFunction func_, bool randomize) : LayerNoCache(num_inputs,num_outputs,func_,0) {
			if(randomize) {
				this->randomize();
			}
		}
		LayerNoCache(uint num_inputs, uint num_outputs, ActivationFunction func_) : LayerNoCache(num_inputs,num_outputs,func_,true) {}
		template <typename T>
		T operator()(const T& t) {
			return this->func((*this)(t));
		}
		LinearAlgebra::Matrix calc_pre_activation_function_output(const LinearAlgebra::Matrix& x_data) const {
			return (*this)(x_data);
		}
		LinearAlgebra::Matrix calc_post_activation_function_output(const LinearAlgebra::Matrix& pre_activation_function_output) const {
			return this->func(pre_activation_function_output);
		}
	}; //LayerNoCache

	LinearAlgebra::Matrix calc_derivatives_to_pass_on(const LinearAlgebra::Matrix& weights,const LinearAlgebra::Matrix& naive_derivatives,const LinearAlgebra::Matrix& from_prev_layer);
	LayerParams calc_partial_derivatives(const LinearAlgebra::Matrix& derivatives,const LinearAlgebra::Matrix& pre_activation_function_output, const LinearAlgebra::Matrix& from_prev_layer);

	typedef struct {
		LinearAlgebra::Matrix pre_act_func_output;
		LinearAlgebra::Matrix post_act_func_output;
	} LayerForDataCache;

	typedef struct {
		LinearAlgebra::Matrix derivatives;
		LayerParams partial_derivatives; 
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
			this->back_data.partial_derivatives = this->calc_partial_derivatives(from_prev_layer);
			return this->calc_derivatives_to_pass_on(from_prev_layer);
		}
		const LinearAlgebra::Matrix& get_post_act_func_output() const {
			return this->for_data.post_act_func_output;
		}
		LinearAlgebra::Matrix calc_derivatives_to_pass_on(const LinearAlgebra::Matrix& from_prev_layer) {
			return MachineLearning::calc_derivatives_to_pass_on(this->get_weights(),this->back_data.derivatives,from_prev_layer);
		}
		LayerParams calc_partial_derivatives(const LinearAlgebra::Matrix& from_prev_layer) {
			return MachineLearning::calc_partial_derivatives(this->back_data.derivatives,this->for_data.pre_act_func_output,from_prev_layer);
		}
	}; //Layer
} //MachineLearning

// MachineLearning::LayerParams& operator+=(MachineLearning::LayerParams& a, const MachineLearning::LayerParams& b);

MachineLearning::Layer& operator+=(MachineLearning::Layer& a, const MachineLearning::LayerParams& b);

MachineLearning::Layer& operator+=(MachineLearning::Layer& a, const MachineLearning::Layer& b);

std::ostream& operator<<(std::ostream& os,const MachineLearning::LayerParams& lp);

#endif //LAYER_H