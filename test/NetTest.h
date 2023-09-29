#ifndef NETTEST_H
#define NETTEST_H

#include "UnitTest.h"
#include "net.h"

// All functions of the NetTest classes should be named after functions in the Net class

namespace NetTest {
	class PrivateAPI {
	public:
		static void calculate_gradient();
	};

	class PublicAPI {
	public:
		static void Net_constructors();
		static void load_training_data();
		static void learn();
		static void verify_save();
	};
	class Performance;
	void execute_all_tests();
};

#endif // NETTEST_H