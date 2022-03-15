#include "../third_party/googletest/googletest/include/gtest/gtest.h"
#include "../Code/Optimization/ShapeFunction.h"

TEST(SHAPE_FUNCTION_OUTPUT, TEMP) {
    ShapeFunction sf(0,6,5);
    sf.set_constraints(0,1.5,0,0);
    std::vector<double> dof_values_initial(3);
    dof_values_initial[0] = 0.2081;
    dof_values_initial[1] = 0.727566;
    dof_values_initial[2] = 00.619589;
    sf.set_free_values(dof_values_initial);
    sf.print();
}

TEST(SHAPE_FUNCTION_TESTS, NDOFS) {
    ShapeFunction sf(0.0, 1.0, 2);
    ASSERT_EQ(sf.n_dofs, 5);
    ASSERT_EQ(sf.n_free_dofs, 0);
}

TEST(SHAPE_FUNCTION_TESTS, EVAL) {
    ShapeFunction sf(0.0, 1.0, 2);
    ASSERT_EQ(sf.n_free_dofs, 0);
    sf.set_constraints(0,1,0,0);
    std::vector<double> dof_values_initial;
    sf.set_free_values(dof_values_initial);
    ASSERT_EQ(sf.evaluate_at(0.5), 0.5);
    ASSERT_EQ(sf.evaluate_derivative_at(0.5), 2.0);
}

TEST(SHAPE_FUNCTION_TESTS, SET_AND_EVAL) {
    ShapeFunction sf(0.0, 3.0, 3);
    ASSERT_EQ(sf.n_dofs, 6);
    ASSERT_EQ(sf.n_free_dofs, 1);
    sf.set_constraints(0,4,0,0);
    std::vector<double> dof_values_initial;
    dof_values_initial.push_back(2);
    sf.set_free_values(dof_values_initial);
    ASSERT_EQ(sf.evaluate_derivative_at(1), 2);
    ASSERT_EQ(sf.evaluate_derivative_at(1.5), 2);
    ASSERT_EQ(sf.evaluate_derivative_at(2), 2);
    ASSERT_EQ(sf.evaluate_at(1), 1);
    ASSERT_EQ(sf.evaluate_at(1.5), 2);
    ASSERT_EQ(sf.evaluate_at(2), 3);
    
    sf.set_constraints(0,0,0,0);
    dof_values_initial[0] = 0;
    sf.set_free_values(dof_values_initial);
    ASSERT_EQ(sf.evaluate_at(0), 0);
    ASSERT_EQ(sf.evaluate_at(1), 0);
    ASSERT_EQ(sf.evaluate_at(1.5), 0);
    ASSERT_EQ(sf.evaluate_at(2), 0);
    ASSERT_EQ(sf.evaluate_at(2.5), 0);
    ASSERT_EQ(sf.evaluate_at(3), 0);
    ASSERT_EQ(sf.evaluate_derivative_at(0), 0);
    ASSERT_EQ(sf.evaluate_derivative_at(1), 0);
    ASSERT_EQ(sf.evaluate_derivative_at(1.5), 0);
    ASSERT_EQ(sf.evaluate_derivative_at(2), 0);
    ASSERT_EQ(sf.evaluate_derivative_at(2.5), 0);
    ASSERT_EQ(sf.evaluate_derivative_at(3), 0);
}

