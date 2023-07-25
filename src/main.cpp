#include "types.h"
#include "activation_function.h"

#define NUM_ROWS_COLS ((LinearAlgebra::uint)5)

template <typename T>
void layer_test() {
	MachineLearning::LayerParams lp(NUM_ROWS_COLS,NUM_ROWS_COLS);
	T v(NUM_ROWS_COLS);
}

int main(int argc, char const *argv[]) {
	layer_test<LinearAlgebra::Matrix>();
	layer_test<LinearAlgebra::VerticalVector>();

	MachineLearning::ActivationFunction f = MachineLearning::get_leaky_ReLU();

	MachineLearning::Layer l((LinearAlgebra::uint)5,(LinearAlgebra::uint)5,MachineLearning::get_leaky_ReLU(),false);

	std::cout << l.parameters.weights << std::endl;
	std::cout << l.parameters.biases << std::endl;
}