#include "UnitTest.h"
#include "net.h"

class NetTest {
	typedef struct NetUintIter {
		std::vector<MachineLearning::uint>::const_iterator def;
		MachineLearning::Net::const_iterator n;
		NetUintIter& operator++() {
			++this->def;
			++this->n;
			return (*this);
		}
		bool operator==(const NetUintIter& iter) {
			if((iter.n)==(this->n)) {
				assert((std::next(iter.def)==this->def) || (iter.def==std::next(this->def)));
				return true;
			} else {
				return false;
			}
		}
	} NetUintIter;

	static bool Net_constructor_gives_correct_num_inputs();

	static void forward_propagation_output();

public:
	static void execute_all_tests();
};