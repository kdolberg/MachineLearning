# Git version
# GIT_VERSION := "$(shell git describe --abbrev=4 --dirty --always --tags)"

# compiler
CXX = g++

# compiler flags
SCALAR_TYPE=float
DEFINE_SCALAR_TYPE_MACRO = -DSCALAR_TYPE=$(SCALAR_TYPE)
FLAGS = -DNOFLAGS -DUNIT_TEST
CXXFLAGS = -std=c++23 $(DEFINE_SCALAR_TYPE_MACRO) $(FLAGS) -DUNIT_TEST -Wall -g -O2 -MMD -MP

# directory of header files
INCLUDES = -I. -I./../Utilities -I./../Utilities/src -I./../Utilities/include -I./include -I./../UnitTest -I./../UnitTest/inc -I./../confirm

# sources and objects
ML_H_FILES = include/activation_function.h include/layer.h include/machine_learning.h include/main.h include/net.h include/types.h
LINALG_DIR = ../Utilities
LINALG_OBJS = obj/la_basic_types.la obj/la_matrix.la obj/la_matrix_like.la obj/la_vector.la obj/la_vector_overloads.la
ML_OBJ = obj/activation_function.ml obj/layer.ml obj/net.ml
ML_TEST_OBJ = obj/test_main.ml_test
UNIT_TEST_OBJ = obj/test.unit_test

# target executable
TARGET = a

TEST = test
OBJECTS = $(ML_OBJ) $(LINALG_OBJS)

$(TARGET): $(OBJECTS) obj/main.ml
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(TARGET) $(OBJECTS)

obj/%.ml: src/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

-include $(ML_OBJS:.ml=.d)

obj/%.la: ../Utilities/src/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

-include $(LINALG_OBJS:.la=.d)

obj/%.ml_test: test/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

-include $(ML_TEST_OBJ:.ml_test=.d)

obj/%.unit_test: ../UnitTest/src/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

-include $(UNIT_TEST_OBJ:.unit_test=.d)

clean:
	rm -f $(ML_OBJ) $(ML_TEST_OBJ) $(UNIT_TEST_OBJ) $(TARGET) $(TEST).exe

clean_linalg:
	rm -f $(LINALG_OBJS)

make_linalg: $(LINALG_OBJS)

linalg:
	$(MAKE) -C $(LINALG_DIR) clean
	$(MAKE) make_linalg

clean_all:
	rm -f $(OBJECTS)

all: $(OBJECTS) $(TARGET)

run: $(TARGET)
	./$(TARGET).exe

$(TEST): $(OBJECTS) $(ML_TEST_OBJ) $(UNIT_TEST_OBJ)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJECTS) $(ML_TEST_OBJ) $(UNIT_TEST_OBJ) -o $(TEST)
# 	gdb $(TEST).exe

.PHONY: clean clean_linalg linalg clean_all run all make_linalg test build_test