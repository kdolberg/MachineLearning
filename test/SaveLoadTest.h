#include "UnitTest.h"

#define PRINT_FUNC() std::cout << "\nEXECUTING TESTS FOR " << __FUNCTION__ << "\n";

static int get_file_num();

namespace SaveLoadTest {
	template <typename T>
	void test_list(std::list<T> l1) {
		std::string str = std::string("test/saves/list") + std::to_string(get_file_num()) + std::string(".list");
		std::list<T> l2;
		TEST_RETURN_FUNC(MachineLearning::save(l1,str.c_str()),==,true);
		TEST_RETURN_FUNC(MachineLearning::load(l2,str.c_str()),==,true);
		TEST_RETURN_FUNC(l1,==,l2);
	}

	// Test functions
	void list();
	void Matrix();
	void LayerParams();
	void TrainingDataset();
	void ActivationFunction();
	void Net();

	/**
	 * @brief Executes all tests
	 */
	void execute_all_tests();
} // SaveLoadTest