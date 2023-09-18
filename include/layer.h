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
		LinearAlgebra::Matrix derivatives_for_next_layer;
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
	};

	class ForPropIter : public std::list<LayerStruct>::iterator {
	public:
		using std::list<LayerStruct>::iterator::iterator;
		ForPropIter(const std::list<LayerStruct>::iterator& i) {
			this->std::list<LayerStruct>::iterator::operator=(i);
		}
		const LinearAlgebra::Matrix& get_input() const {
			return std::prev(*this)->fordata.pre_act_func_output;
		}
		const LinearAlgebra::Matrix& get_activation() const {
			return (*this)->fordata.pre_act_func_output;
		}
		void update_forward_data_cache(const LinearAlgebra::Matrix x_data) {
			(*this)->fordata.pre_act_func_output = (*this)->params(x_data);
			(*this)->fordata.post_act_func_output = (*this)->actfunc(this->get_activation());
		}
		void update_forward_data_cache() {
			this->update_forward_data_cache(this->get_input());
		}
	};
	
	class BackPropIter : public std::list<LayerStruct>::reverse_iterator {
	public:
		using std::list<LayerStruct>::reverse_iterator::reverse_iterator;
		const LinearAlgebra::Matrix& get_input() const {
			return std::prev(*this)->backdata.derivatives_for_next_layer;
		}
		const LinearAlgebra::Matrix& get_activation() const {
			return (*this)->fordata.pre_act_func_output;
		}
		LinearAlgebra::Matrix calc_naive_derivatives() const {
			return (*this)->actfunc.ddx((*this)->fordata.pre_act_func_output);
		}

		LinearAlgebra::Matrix calc_derivatives_for_next_layer() const;

		void update_backward_data_cache(const LinearAlgebra::Matrix& derivatives_from_prev_layer) {
			// LinearAlgebra::Matrix& naive_derivatives = this->calc_naive_derivatives();
		}
		void update_backward_data_cache() {
			this->update_backward_data_cache(this->get_input());
		}
	};

	class Net : public std::list<MachineLearning::LayerStruct> {
	public:
		using std::list<MachineLearning::LayerStruct>::list;
		Net() /*: std::list<MachineLearning::LayerParams>::list()*/ {}
		Net(std::vector<uint> def) : Net() {
			for (uint i = 0; i < def.size()-1; ++i) {
				uint num_inputs = def[i];
				uint num_outputs = def[i+1];
				MachineLearning::LayerStruct tmp(num_inputs,num_outputs);
				this->push_back(tmp);
			}
		}
	}; 
} //MachineLearning

std::ostream& operator<<(std::ostream& os,const MachineLearning::LayerParams& lp);

#endif //LAYER_H