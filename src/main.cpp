#include <list>
#include <concepts>
#include <iterator>
#include <vector>
#include "machine_learning.h"

int main(int argc, char const *argv[]) {
	std::vector<MachineLearning::uint> def = {5,5,4,1};
	MachineLearning::Net n(def);
	std::cout << n << std::endl;
	MachineLearning::TrainingDatasetSig d;
}