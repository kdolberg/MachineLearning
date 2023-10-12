#include "main.h"

int main(int argc, char const *argv[]) {
	NetTest::execute_all_tests();
	ActFuncTest::execute_all_tests();
	SaveLoadTest::execute_all_tests();
	print_report_card();
	return 0;
}