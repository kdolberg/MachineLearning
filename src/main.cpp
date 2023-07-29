#include <list>
#include <concepts>
#include <iterator>
#include "types.h"
#include "activation_function.h"

namespace MachineLearning {
	/**
	 * @brief Calculates the pre-activation function output data and the post-activation function output data.
	 */
	void forprop(std::list<Layer>::const_iterator i,std::list<Layer>::const_iterator end, ForDataCache& cache, const LinearAlgebra::Matrix& inputdata) {
		cache.pre_act_func_output.push_back(i->parameters(inputdata));
		cache.post_act_func_output.push_back(i->func(cache.pre_act_func_output.back()));
		if(std::next(i,1)!=end) {
			forprop(std::next(i,1),end,cache,cache.post_act_func_output.back());
		}
	}

	/**
	 * @brief Calculates the partial derivatives 
	 */
	void backprop(	std::list<Layer>::const_reverse_iterator i,
					std::list<Layer>::const_reverse_iterator end,
					const ForDataCache& for_cache,
					BackDataCache& back_cache	)
	{
		back_cache.naive_derivatives.push_back(i->derivative());
		if((++i)!=end) {
			backprop(i,end,for_cache,back_cache);
		}
	}
	class MetaLayer : public std::list<Layer>::const_iterator {
		ForDataCache for_data;
		BackDataCache back_data;
	public:
		MetaLayer() {}
		void initialize_data_caches(uint training_dataset_size) {
			//Initialize forward propagation data
			this->for_data.pre_act_func_output 	=	LinearAlgebra::Matrix(MINDEX(	(**this).parameters.get_num_inputs(),	training_dataset_size	));
			this->for_data.post_act_func_output	=	LinearAlgebra::Matrix(MINDEX(	(**this).parameters.get_num_outputs(),	training_dataset_size	));

			//Initialize backward propagation data
			/*Do something with the back_data cache*/
		}
	}; //MetaLayer
	class Net {
	protected:
		std::list<Layer> layers;
	public:
		Net(){}
		Net(std::vector<uint> def) : Net() {
			for (uint i = 0; i < def.size()-1; ++i) {
				uint num_inputs = def[i];
				uint num_outputs = def[i+1];
				Layer tmp(num_inputs,num_outputs,get_leaky_ReLU(),true);
				layers.push_back(tmp);
			}
		}
		std::list<Layer>::const_iterator layer_cbegin() const {
			return this->layers.cbegin();
		}
		std::list<Layer>::const_iterator layer_cend() const {
			return this->layers.cend();
		}
	public:
		std::string str() const {
			std::stringstream ss;
			for (std::list<Layer>::const_iterator i = this->layer_cbegin(); i != this->layer_cend(); ++i) {
				ss << i->parameters.weights << std::endl;
				ss << i->parameters.biases << std::endl;
			}
			return ss.str();
		}
	}; //Net
}

std::ostream& operator<<(std::ostream& os,const MachineLearning::Net n) {
	os << n.str();
	return os;
}

#define IS_MATRIXLIKE(_type) std::is_base_of_v<LinearAlgebra::MatrixLike,_type>

#define PRINT_IF_MATRIXLIKE(_type) std::cout << #_type << ": " << IS_MATRIXLIKE(_type) << std::endl;

int main(int argc, char const *argv[]) {
	std::vector<MachineLearning::uint> def = {5,5,4,1};
	MachineLearning::Net n(def);
	PRINT_IF_MATRIXLIKE(LinearAlgebra::VerticalVector);
	PRINT_IF_MATRIXLIKE(LinearAlgebra::HorizontalVector);
	PRINT_IF_MATRIXLIKE(LinearAlgebra::Matrix);
	PRINT_IF_MATRIXLIKE(LinearAlgebra::mindex_t);
	PRINT_IF_MATRIXLIKE(int);

}