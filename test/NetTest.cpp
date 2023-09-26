#include "NetTest.h"

#define H 0.05f

void NetTest::PrivateAPI::calculate_gradient() {
}

void NetTest::PublicAPI::load_training_data() {

}
void NetTest::PublicAPI::learn() {

}

void NetTest::execute_all_tests() {
	NetTest::PublicAPI::learn();
	NetTest::PublicAPI::load_training_data();
}