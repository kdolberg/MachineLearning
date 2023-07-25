#include <list>
#include "types.h"
#include "activation_function.h"

namespace MachineLearning {
	class Net {
	protected:
		std::list<MachineLearning::Layer> layers;
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
		auto begin_layer_const_iter() const {
			return this->layers.cbegin();
		}
		auto end_layer_const_iter() const {
			return this->layers.cend();
		}
	// protected:
	// 	auto begin_layer_iter() {
	// 		return this->layers.begin();
	// 	}
	// 	auto end_layer_iter() {
	// 		return this->layers.end();
	// 	}
	public:
		std::string str() const {
			std::stringstream ss;
			for (auto i = this->begin_layer_const_iter(); i != this->end_layer_const_iter(); ++i) {
				ss << (*i).parameters.weights << std::endl;
				ss << (*i).parameters.biases << std::endl;
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