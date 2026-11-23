#include "../Code/Core/Types.h"
#include "../Code/Solutions/ExactSolutionRamped.h"
#include "../Code/Helpers/staticfunctions.h"
#include <gtest/gtest.h>
#include <deal.II/lac/full_matrix.h>

TEST(ExactSolutionRampedTests, RampingTestsC0) {
    GlobalParams.Signal_tapering_type = SignalTaperingType::C0;
    // The ramp goes from z = 0 to z = 1; set here so that the test does not depend on the tests before it.
    GlobalParams.tapering_min_z = 0.0;
    GlobalParams.tapering_max_z = 1.0;
    ExactSolution::load_data(repository_path("Modes/output_mode.dat"));
    ExactSolutionRamped sol = {};
    Position p = {0,0,0};
    ASSERT_EQ(sol.get_ramping_factor_for_position(p),1);
    ASSERT_EQ(sol.get_ramping_factor_derivative_for_position(p),-1);
    p[2] = 1;
    ASSERT_EQ(sol.get_ramping_factor_for_position(p),0);
    ASSERT_EQ(sol.get_ramping_factor_derivative_for_position(p),-1);
}

TEST(ExactSolutionRampedTests, RampingTestsC1) {
    GlobalParams.Signal_tapering_type = SignalTaperingType::C1;
    // The ramp goes from z = 0 to z = 1; set here so that the test does not depend on the tests before it.
    GlobalParams.tapering_min_z = 0.0;
    GlobalParams.tapering_max_z = 1.0;
    ExactSolution::load_data(repository_path("Modes/output_mode.dat"));
    ExactSolutionRamped sol = {};
    Position p = {0,0,0};
    ASSERT_EQ(sol.get_ramping_factor_for_position(p),1);
    ASSERT_EQ(sol.get_ramping_factor_derivative_for_position(p),0);
    p[2] = 1;
    ASSERT_EQ(sol.get_ramping_factor_for_position(p),0);
    ASSERT_EQ(sol.get_ramping_factor_derivative_for_position(p),0);
}

TEST(ExactSolutionTests, PhaseTest) {
    ExactSolution::load_data(repository_path("Modes/output_mode.dat"));
    ExactSolution es = {};
    Position p = {0,0,0};
    ASSERT_NEAR(es.value(p,0).imag(), 0.0, 0.001);
    ASSERT_NEAR(es.value(p,0).real(), 1.0, 0.001); 
}