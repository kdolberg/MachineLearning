#ifndef SAVELOADTEST_H
#define SAVELOADTEST_H

#define PRINT_FUNC() std::cout << "\nEXECUTING SAVE-LOAD TESTS FOR " << __FUNCTION__ << "\n";

namespace SaveLoadTest {

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

#endif // SAVELOADTEST_H