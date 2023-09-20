#ifndef LAYER_H
#define LAYER_H

#include <list>
#include "types.h"
#include "confirm.h"

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
		bool operator==(LayerParams& ls) {
			return (this->get_weights()==ls.get_weights()) && (this->get_biases()==ls.get_biases());
		}
	}; //LayerParams

	typedef struct {
		LinearAlgebra::Matrix pre_act_func_output;
		LinearAlgebra::Matrix post_act_func_output;
	} LayerForDataCache;

	typedef struct {
		LinearAlgebra::Matrix derivatives_for_layer_below;
		LinearAlgebra::Matrix naive_derivatives;
		LayerParams partial_derivatives;
	} LayerBackDataCache;

	class LayerStruct {
	public:
		LayerParams params;
		LayerForDataCache fordata;
		LayerBackDataCache backdata;
		ActivationFunction actfunc;
		LayerStruct(uint num_ins,uint num_outs) {
			this->params = MachineLearning::LayerParams(num_ins,num_outs);
			//Default activation function
			this->actfunc = get_leaky_ReLU();
		}
		LayerStruct(const LinearAlgebra::Matrix& weights, const LinearAlgebra::Matrix& biases) {
			this->params = MachineLearning::LayerParams(weights,biases);
			//Default activation function
			this->actfunc = get_leaky_ReLU();
		}
	};				

	template <typename T>
	class PropIter : public T {
	public:
		using T::T;				
		PropIter(const T& i) {
			this->T::operator=(i);
		}
	protected:
		/**
		 * @brief Returns the number of inputs of the layer in question
		 */
		LinearAlgebra::uint get_num_inputs() const {
			return this->params().get_num_inputs();
		}
		/**
		 * @brief Returns the number of outputs of the layer in question
		 */
		LinearAlgebra::uint get_num_outputs() const {
			return this->params().get_num_outputs();
		}
		/**
		 * @brief Returns the LayerParams object in the LayerStruct in question
		 */
		const LayerParams& params() const {
			return (*this)->params;
		}
		/**
		 * @brief Returns the LayerForDataCache
		 */
		const LayerForDataCache& fordata() const {
			return (*this)->fordata;
		}
		/**
		 * @brief Returns the LayerBackDataCache
		 */
		const LayerBackDataCache& backdata() const {
			return (*this)->backdata;
		}
		/**
		 * @brief Returns the ActivationFunction
		 */
		const ActivationFunction& actfunc() const {
			return (*this)->actfunc;
		}
		/**
		 * @brief Returns the LayerParams object in the LayerStruct in question
		 */
		LayerParams& params() {
			return (*this)->params;
		}
		LayerForDataCache& fordata() {
			return (*this)->fordata;
		}
		/**
		 * @brief Returns the LayerBackDataCache
		 */
		LayerBackDataCache& backdata() {
			return (*this)->backdata;
		}
		/**
		 * @brief Returns the ActivationFunction
		 */
		ActivationFunction& actfunc() {
			return (*this)->actfunc;
		}
		const LinearAlgebra::Matrix& get_x_input() const {
			return this->below().get_post_act_func_output();
		}
		const LinearAlgebra::Matrix& get_partial_derivatives() const {
			return (*this)->backdata.partial_derivatives;
		}
		const LinearAlgebra::Matrix& get_derivatives_from_layer_above() const {
			return this->above()->backdata.derivatives_for_layer_below;
		}
		const LinearAlgebra::Matrix& get_pre_act_func_output() const {
			return (*this)->fordata.pre_act_func_output;
		}
		const LinearAlgebra::Matrix& get_post_act_func_output() const {
			return (*this)->fordata.post_act_func_output;
		}
	public:
		const LinearAlgebra::Matrix& get_weights() const {
			return this->params().get_weights();
		}
		const LinearAlgebra::Matrix& get_biases() const {
			return this->params().get_biases();
		}
		PropIter<T> above() const {
			PropIter<T> ret;
			if(std::is_same<T,std::list<LayerStruct>::iterator>::value) {
				ret = std::next(*this);
			} else {
				ret = std::prev(*this);
			}
			return ret;
		}
		PropIter<T> below() const {
			PropIter<T> ret;
			if(std::is_same<T,std::list<LayerStruct>::reverse_iterator>::value) {
				ret = std::next(*this);
			} else {
				ret = std::prev(*this);
			}
			return ret;

		}
	protected:
		virtual void update_data_cache() {} // Change to pure virtual
	};

	class ForPropIter : public MachineLearning::PropIter<std::list<LayerStruct>::iterator> {
	public:
		// Constructors
		using PropIter::PropIter;				

		// Original functions
		/**
		 * @brief Updates the forward propagation data cache
		 * @param Input data AKA post-activation output from the previous layer (unless this is the input layer)
		 */
		void update_data_cache(const LinearAlgebra::Matrix& x_data) {
			this->fordata().pre_act_func_output = (*this)->params(x_data);
			this->fordata().post_act_func_output = (*this)->actfunc(this->get_pre_act_func_output());
		}
		void update_data_cache() {
			this->update_data_cache(this->get_x_input());
		}
	};

	class BackPropIter : public MachineLearning::PropIter<std::list<LayerStruct>::reverse_iterator> {
	protected:
		LinearAlgebra::uint get_num_data_points() const {
			assert(this->get_post_act_func_output().get_num_rows()>0);
			assert(this->get_post_act_func_output().get_num_cols()>0);
			return this->get_post_act_func_output().get_num_cols();
		}
	public:
		using PropIter::PropIter;
		void update_data_cache(const LinearAlgebra::Matrix& derivatives_from_layer_above, const LinearAlgebra::Matrix& x_data) {
			// PRINT_LOC(x_data);
			this->backdata().naive_derivatives = this->actfunc().ddx(this->get_pre_act_func_output());
			this->backdata().derivatives_for_layer_below.resize(MINDEX(this->get_num_inputs(),this->get_num_data_points()));
			LinearAlgebra::Matrix pd_weights(this->get_weights().size());
			LinearAlgebra::Matrix pd_biases(this->get_biases().size());
			// PRINT_LOC(this->backdata().naive_derivatives.size());
			// PRINT_LOC(derivatives_from_layer_above.size());
			assert(this->backdata().naive_derivatives.size()==derivatives_from_layer_above.size());

			for (uint data_index = 0; data_index < this->get_num_data_points(); ++data_index) {
				// PRINT_LOC(data_index);
				for (uint node_index = 0; node_index < this->get_num_outputs(); ++node_index) {
					// PRINT_LOC(node_index);
					// Initialize naive derivative mindex
					LinearAlgebra::mindex_t nd_m = MINDEX(node_index,data_index);
					// PRINT_LOC(nd_m);
					// PRINT_LOC(this->backdata().naive_derivatives.size());
					// PRINT_LOC(derivatives_from_layer_above.size());
					LinearAlgebra::scalar_t f_prime_X_g_prime_tmp = (this->backdata().naive_derivatives[nd_m])*(derivatives_from_layer_above[nd_m]);
					// PRINT_LOC(f_prime_X_g_prime_tmp);
					for (uint weight_index = 0; weight_index < this->get_num_inputs(); ++weight_index) {
						// PRINT_LOC(weight_index);
						// Initialize weights and x_data mindices
						LinearAlgebra::mindex_t w_m = MINDEX(node_index,weight_index);
						LinearAlgebra::mindex_t x_m = MINDEX(weight_index,data_index);

						// Do calculations for partial derivatives
						pd_weights[w_m] = (x_data[x_m])*f_prime_X_g_prime_tmp;

						// Do calculations for derivatives to be passed on to layer below
						this->backdata().derivatives_for_layer_below[w_m] = this->get_weights()[w_m]*f_prime_X_g_prime_tmp;

					} // weight_index
				} // node_index
			} // data_index
			this->backdata().partial_derivatives = (LayerParams){pd_weights,pd_biases};
		}
		void update_data_cache() {
			this->update_data_cache(this->get_derivatives_from_layer_above(),this->get_x_input());
		}
		void update_data_cache_input_layer(const LinearAlgebra::Matrix& x_data) {
			this->update_data_cache(this->get_derivatives_from_layer_above(),x_data);
		}
		void update_data_cache_output_layer(const LinearAlgebra::Matrix& derivatives_from_error_func) {
			this->update_data_cache(derivatives_from_error_func,this->get_x_input());
		}
	};
} //MachineLearning

std::ostream& operator<<(std::ostream& os,const MachineLearning::LayerParams& lp);

#endif //LAYER_H