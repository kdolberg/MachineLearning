
#ifndef ACTFUNCTEST_H
#define ACTFUNCTEST_H

#include "UnitTest.h"
#include "activation_function.h"

namespace ActFuncTest {
	//Tests
	void str();
	void leaky_ReLU();
	void sigmoid();
	/**
	 * @brief executes all tests in the ActFuncTest namespace
	 */
	void execute_all_tests();
}

#endif // ACTFUNCTEST_H