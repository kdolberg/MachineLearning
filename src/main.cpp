#include <list>
#include "types.h"
#include "activation_function.h"

namespace MachineLearning {

	class Net {
		// static void forprop(std::list<Layer>::const_iterator i,std::list<Layer>::const_iterator end, ForDataCache& cache, const LinearAlgebra::Matrix& inputdata) {
		// 	cache.pre_act_func_output.push_back(i->parameters(inputdata));
		// 	cache.post_act_func_output.push_back(i->func(cache.pre_act_func_output.back()));
		// 	if(i+1!=end) {
		// 		forprop((++i),end,cache,cache.post_act_func_output.back());
		// 	}
		// }

		// static void backprop(std::list<Layer>::const_reverse_iterator i, std::list<Layer>::const_reverse_iterator end, const ForDataCache& for_cache, BackDataCache& back_cache) {

		// 	if(i+1!=end){
		// 		backprop((++i),end,for_cache,back_cache);
		// 	}
		// }

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

std::ostream& operator<<(std::ostream& os,const MachineLearning::Net n) {
	os << n.str();
	return os;
}

int main(int argc, char const *argv[]) {
	std::vector<MachineLearning::uint> def = {5,5,4,1};
	MachineLearning::Net n(def);
	std::cout << n;
}