# Compiler
CXX = g++

# Compiler flags
SCALAR_TYPE=float
DEFINE_SCALAR_TYPE_MACRO = -DSCALAR_TYPE=$(SCALAR_TYPE)
FLAGS = -DNOFLAGS
CXXFLAGS = -std=c++23 $(DEFINE_SCALAR_TYPE_MACRO) $(FLAGS) -Wall -g -O2 -MMD -MP

# Directory of header files
INCLUDES = -I. -I./../Utilities -I./../Utilities/src -I./../Utilities/include -I./inc -I./../UnitTest -I./../UnitTest/inc -I./../confirm

# Sources and objects
LINALG_OBJ = obj/la_basic_types.la obj/la_matrix.la obj/la_matrix_like.la obj/la_vector.la obj/la_vector_overloads.la
ML_OBJ = obj/activation_function.ml obj/layer.ml obj/net.ml
ML_TEST_OBJ = obj/main.ml_test obj/NetTest.ml_test obj/NetPerformance.ml_test obj/ActFuncTest.ml_test
UNIT_TEST_OBJ = obj/test.unit_test
OBJECTS = $(ML_OBJ) $(LINALG_OBJ) $(ML_TEST_OBJ) $(UNIT_TEST_OBJ)

# Target executable
TARGET = test

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJECTS) -o $(TARGET)

obj/%.ml: src/%.cpp inc/%.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

obj/%.la: ../Utilities/src/%.cpp ../Utilities/src/%.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

obj/%.ml_test: test/%.cpp test/%.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

obj/%.unit_test: ../UnitTest/src/%.cpp ../UnitTest/inc/%.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# -include $(ML_OBJS:.ml=.d)
# -include $(LINALG_OBJ:.la=.d)
# -include $(ML_TEST_OBJ:.ml_test=.d)
# -include $(UNIT_TEST_OBJ:.unit_test=.d)

clean:
	rm -f $(ML_OBJ) $(ML_TEST_OBJ) $(UNIT_TEST_OBJ) $(TARGET).exe mynet.nn

clean_linalg:
	rm -f $(LINALG_OBJ)

linalg: $(LINALG_OBJ)

clean_all: clean clean_linalg

all: $(TARGET)

run: $(TARGET)
	./$(TARGET).exe

.PHONY: clean clean_linalg linalg clean_all run all