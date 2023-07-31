#include <list>
#include <concepts>
#include <iterator>

// #define IS_MATRIXLIKE(_type) std::is_base_of_v<LinearAlgebra::MatrixLike,_type>

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