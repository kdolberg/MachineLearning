#include "main.h"

int main(int argc, char const *argv[]) {
	std::cout << "Compilation date: " << __DATE__ << " " << __TIME__ << std::endl;
	SaveLoadTest::execute_all_tests();
	NetTest::execute_all_tests();
	ActFuncTest::execute_all_tests();
	print_report_card();
	return 0;
}