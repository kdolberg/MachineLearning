# Git version
# GIT_VERSION := "$(shell git describe --abbrev=4 --dirty --always --tags)"

# compiler
CXX = g++

# compiler flags
SCALAR_TYPE=float
DEFINE_SCALAR_TYPE_MACRO = -DSCALAR_TYPE=$(SCALAR_TYPE)
FLAGS = -DNOFLAGS -DUNIT_TEST
CXXFLAGS = -std=c++23 $(DEFINE_SCALAR_TYPE_MACRO) $(FLAGS) -Wall -g -O0

# directory of header files
INCLUDES = -I. -I./../Utilities -I./../Utilities/src -I./../Utilities/include -I./include -I./../UnitTest -I./../UnitTest/inc -I./../confirm

# sources and objects
LINALG_DIR = ../Utilities
LINALG_OBJS = obj/la_basic_types.la obj/la_matrix.la obj/la_matrix_like.la obj/la_vector.la obj/la_vector_overloads.la
ML_OBJ = obj/activation_function.ml obj/net.ml obj/layer.ml
OBJECTS = $(ML_OBJ) $(LINALG_OBJS)

# target executable
TARGET = a

TEST = test_exec
TEST_OBJECTS = obj/test_main.ml_test obj/test.unit_test

$(TARGET): $(OBJECTS) obj/main.ml
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(TARGET) $(OBJECTS)

obj/%.ml: src/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

obj/%.la: ../Utilities/src/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

obj/%.ml_test: test/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

obj/%.unit_test: ../UnitTest/src/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(ML_OBJ) $(TEST_OBJECTS) $(TARGET)

clean_linalg:
	rm -f $(LINALG_OBJS)

make_linalg: $(LINALG_OBJS)

linalg:
	$(MAKE) -C $(LINALG_DIR) clean
	$(MAKE) make_linalg

clean_all:
	rm -f $(OBJECTS)

test_stub:
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(TARGET) test_stubs/main.cpp

all: $(OBJECTS) $(TARGET)

run: $(TARGET)
	./$(TARGET).exe

$(TEST): $(OBJECTS) $(TEST_OBJECTS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(TEST) $(OBJECTS) $(TEST_OBJECTS)

test: $(TEST) $(TEST_OBJECTS)
	./$(TEST).exe

.PHONY: clean clean_linalg linalg clean_all run all make_linalg test