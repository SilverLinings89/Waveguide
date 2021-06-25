#include "../Code/Core/Types.h"
#include "../Code/Solutions/ExactSolutionRamped.h"
#include "../third_party/googletest/googletest/include/gtest/gtest.h"
#include <deal.II/lac/full_matrix.h>

TEST(ExactSolutionRampedTests, RampingTestsC0) {
    GlobalParams.Signal_tapering_type = SignalTaperingType::C0;
    ExactSolutionRamped sol = {true,false};
    Position p = {0,0,0};
    ASSERT_EQ(sol.get_ramping_factor_for_position(p),1);
    ASSERT_EQ(sol.get_ramping_factor_derivative_for_position(p),-1);
    p[2] = 1;
    ASSERT_EQ(sol.get_ramping_factor_for_position(p),0);
    ASSERT_EQ(sol.get_ramping_factor_derivative_for_position(p),-1);
}

TEST(ExactSolutionRampedTests, RampingTestsC1) {
    GlobalParams.Signal_tapering_type = SignalTaperingType::C1;
    ExactSolutionRamped sol = {true,false};
    Position p = {0,0,0};
    ASSERT_EQ(sol.get_ramping_factor_for_position(p),1);
    ASSERT_EQ(sol.get_ramping_factor_derivative_for_position(p),0);
    p[2] = 1;
    ASSERT_EQ(sol.get_ramping_factor_for_position(p),0);
    ASSERT_EQ(sol.get_ramping_factor_derivative_for_position(p),0);
}

TEST(ExactSolutionTests, PhaseTest) {
    ExactSolution es = {true,false};
    Position p = {0,0,0};
    ASSERT_NEAR(es.value(p,0).imag(), 0.0, 0.001);
    ASSERT_NEAR(es.value(p,0).real(), 1.0, 0.001); 
}