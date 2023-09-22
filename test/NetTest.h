#include "UnitTest.h"
#include "net.h"

class NetTest {
	static MachineLearning::Gradient numerical_gradient(MachineLearning::Net n, MachineLearning::TrainingDataset td);
	static bool Net_constructor_gives_correct_num_inputs();
	static void forward_propagate();
	static void backward_propagate();
	static void calculate_gradient();
public:
	static void execute_all_tests();
};