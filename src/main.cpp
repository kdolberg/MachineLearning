#include <list>
#include <concepts>
#include "types.h"
#include "activation_function.h"

namespace MachineLearning {

	class Net {
		/**
		 * @brief Calculates the pre-activation function output data and the post-activation function output data.
		 */
		static void forprop(std::list<Layer>::const_iterator i,std::list<Layer>::const_iterator end, ForDataCache& cache, const LinearAlgebra::Matrix& inputdata) {
			cache.pre_act_func_output.push_back(i->parameters(inputdata));
			cache.post_act_func_output.push_back(i->func(cache.pre_act_func_output.back()));
			if((++i)!=end) {
				forprop(i,end,cache,cache.post_act_func_output.back());
			}
		}

		/**
		 * @brief Calculates the partial derivatives 
		 */
		static void backprop(std::list<Layer>::const_reverse_iterator i, std::list<Layer>::const_reverse_iterator end, const ForDataCache& for_cache, BackDataCache& back_cache) {
			back_cache.naive_derivatives.push_back(i->derivative())
			if((++i)!=end){
				backprop(i,end,for_cache,back_cache);
			}
		}

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
	};
}

// template<typename T>
// concept DerivedFromMatrix_Like = std::is_base_of_v<LinearAlgebra::MatrixLike,T>;

// std::list<int>::const_iterator operator+(std::list<int>::const_iterator iter, int n) {
// 	for (int i = 0; i < n; ++i) {
// 		++iter;
// 	}
// 	return iter;
// }

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