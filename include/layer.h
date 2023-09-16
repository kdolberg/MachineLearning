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
		template <LinearAlgebra::MATRIXLIKE W,LinearAlgebra::MATRIXLIKE B>
		LayerParams(const W& weights, const B& biases) {
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
	}; //LayerParams

	typedef struct {
		LinearAlgebra::Matrix pre_act_func_output;
		LinearAlgebra::Matrix post_act_func_output;
	} LayerForDataCache;

	typedef struct {
		LinearAlgebra::Matrix derivatives;
		LayerParams partial_derivatives; 
	} LayerBackDataCache;


	typedef struct {
		std::list<MachineLearning::LayerParams>::const_iterator parameters,parameters_end;
		std::list<MachineLearning::LayerForDataCache>::iterator fordata,fordata_end;
		std::list<MachineLearning::ActivationFunction>::const_iterator act_func,act_func_end;
	} ForPropIterStruct;

	class ForPropIter : protected MachineLearning::ForPropIterStruct {
	public:
		ForPropIter(ForPropIterStruct fpis) {
			this->parameters = fpis.parameters;
			this->parameters_end = fpis.parameters_end;
			this->fordata = fpis.fordata;
			this->fordata_end = fpis.fordata_end;
			this->act_func = fpis.act_func;
			this->act_func_end = fpis.act_func_end;
		}
		ForPropIter(const std::list<MachineLearning::LayerParams>& lp,const std::list<MachineLearning::ActivationFunction>& af) {
			CONFIRM(lp.size()==af.size());
			this->parameters = lp.cbegin();
			this->parameters_end = lp.cend();
			this->act_func = af.cbegin();
			this->act_func_end = af.cend();
		}
		ForPropIter(const std::list<MachineLearning::LayerParams>& lp,std::list<MachineLearning::LayerForDataCache>& fd,const std::list<MachineLearning::ActivationFunction>& af) : ForPropIter(lp,af) {
			this->fordata = fd.begin();
			this->fordata_end = fd.end();
		}
		ForPropIter& operator++() {
			++(this->parameters);
			++(this->fordata);
			++(this->act_func);
			return (*this);
		}
		LinearAlgebra::Matrix update_data_cache(const LinearAlgebra::Matrix& x_data) {
			//Calculate the pre-activation function output
			this->fordata->pre_act_func_output = this->parameters->operator()(x_data);
			//Calculate the post-activation function output
			this->fordata->post_act_func_output = this->act_func->operator()(this->fordata->pre_act_func_output);
			return this->fordata->post_act_func_output;
		}

		bool finished() {
			//Check if the parameters iterator is at the end
			if(this->parameters == this->parameters_end){
				// Assert that both of the other iterators are also at the end
				assert(this->fordata == this->fordata_end);
				assert(this->act_func == this->act_func_end);
				// Return true (which is the return value of all 3 boolean checks)
				return true;
			} else {
				// Assert that both of the other iterators are finished
				assert(this->fordata != this->fordata_end);
				assert(this->act_func != this->act_func_end);
				// Return false (which is the return value of all 3 boolean checks)
				return false;
			}
		}
		ForPropIter next() const {
			ForPropIterStruct ret = {std::next(this->parameters),this->parameters_end,std::next(this->fordata),this->fordata_end,std::next(this->act_func),this->act_func_end};
			return ret;
		}

		LinearAlgebra::Matrix propagate(const LinearAlgebra::Matrix& in_data) {
			if(!(this->finished())) {
				return this->next().propagate(this->update_data_cache(in_data));
			} else {
				return in_data;
			}
		}
	}; // ForPropIter

} //MachineLearning

std::ostream& operator<<(std::ostream& os,const MachineLearning::LayerParams& lp);

#endif //LAYER_H