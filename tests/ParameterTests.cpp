// Tests for the parameter handling (Parameters, ParameterReader, ParameterOverride, GetParameters) and for the small
// data classes PointVal, ShapeDescription and the analytical fields in PointSourceField.
#include <array>
#include <sstream>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <cmath>
#include <complex>
#include <functional>
#include <set>
#include <stdexcept>
#include "../Code/Core/Types.h"
#include "../Code/GlobalObjects/GlobalObjects.h"
#include "../Code/Helpers/ParameterOverride.h"
#include "../Code/Helpers/ParameterReader.h"
#include "../Code/Helpers/Parameters.h"
#include "../Code/Helpers/PointSourceField.h"
#include "../Code/Helpers/PointVal.h"
#include "../Code/Helpers/ShapeDescription.h"
#include "../Code/Helpers/staticfunctions.h"
#include "./HelpersA.h"

namespace {

  using helpers_a::ScopedGeometry;
  using helpers_a::ScopedGlobalParams;
  using helpers_a::TemporaryDirectory;

  const double pi = 3.14159265358979323846;
  const ComplexNumber imaginary_unit(0, 1);

  // Writes a run file and a case file with the given entries ("name = value" per line) and reads them.
  Parameters read_parameters(const std::string & run_entries, const std::string & case_entries) {
    TemporaryDirectory directory("parameters");
    auto as_file = [](const std::string & section, const std::string & entries) {
      std::string ret = "subsection " + section + "\n";
      std::istringstream lines(entries);
      std::string line;
      while (std::getline(lines, line)) {
        if (line.find('=') != std::string::npos) ret += "  set " + line + "\n";
      }
      return ret + "end\n";
    };
    helpers_a::write_file(directory.file("run.prm"), as_file("Run parameters", run_entries));
    helpers_a::write_file(directory.file("case.prm"), as_file("Case parameters", case_entries));
    ParameterReader reader;
    return reader.read_parameters(directory.file("run.prm"), directory.file("case.prm"));
  }

  // complete_data() prints which source is used. The output is discarded.
  void complete_quietly(Parameters & parameters) {
    testing::internal::CaptureStdout();
    try {
      parameters.complete_data();
    } catch (...) {
      testing::internal::GetCapturedStdout();
      throw;
    }
    testing::internal::GetCapturedStdout();
  }

  // Parameters of a run on n_x * n_y * n_z processes that uses a source without input files.
  Parameters parameters_for_blocks(unsigned int n_x, unsigned int n_y, unsigned int n_z, unsigned int rank) {
    Parameters ret;
    ret.Blocks_in_x_direction = n_x;
    ret.Blocks_in_y_direction = n_y;
    ret.Blocks_in_z_direction = n_z;
    ret.NumberProcesses = n_x * n_y * n_z;
    ret.MPI_Rank = rank;
    ret.Point_Source_Type = 1;
    return ret;
  }

  typedef std::function<ComplexNumber(const Position &, unsigned int)> Field;

  Field field_of(const dealii::Function<3, ComplexNumber> & function) {
    return [&function](const Position & p, unsigned int component) {
      NumericVectorLocal values(3);
      function.vector_value(p, values);
      return values[component];
    };
  }

  Position shifted(Position p, unsigned int direction, double distance) {
    p[direction] += distance;
    return p;
  }

  // Central difference for the derivative of one component in one direction.
  ComplexNumber derivative(const Field & field, const Position & p, unsigned int component, unsigned int direction, double h) {
    return (field(shifted(p, direction, h), component) - field(shifted(p, direction, -h), component)) / (2 * h);
  }

  dealii::Tensor<1, 3, ComplexNumber> curl(const Field & field, const Position & p, double h) {
    dealii::Tensor<1, 3, ComplexNumber> ret;
    ret[0] = derivative(field, p, 2, 1, h) - derivative(field, p, 1, 2, h);
    ret[1] = derivative(field, p, 0, 2, h) - derivative(field, p, 2, 0, h);
    ret[2] = derivative(field, p, 1, 0, h) - derivative(field, p, 0, 1, h);
    return ret;
  }

  ComplexNumber divergence(const Field & field, const Position & p, double h) {
    return derivative(field, p, 0, 0, h) + derivative(field, p, 1, 1, h) + derivative(field, p, 2, 2, h);
  }

  ComplexNumber laplace(const Field & field, const Position & p, unsigned int component, double h) {
    ComplexNumber ret = -6.0 * field(p, component);
    for (unsigned int direction = 0; direction < 3; direction++) {
      ret += field(shifted(p, direction, h), component) + field(shifted(p, direction, -h), component);
    }
    return ret / (h * h);
  }

  double norm(const dealii::Tensor<1, 3, ComplexNumber> & tensor) {
    return std::sqrt(std::norm(tensor[0]) + std::norm(tensor[1]) + std::norm(tensor[2]));
  }

}

// Entries for which the default in ParameterReader::declare_parameters and the default in the struct agree.
TEST(ParameterReaderTests, DefaultsOfReaderMatchDefaultsOfStruct) {
  const Parameters read = read_parameters("", "");
  const Parameters defaults = Parameters();
  EXPECT_EQ(read.Solver_Precision, defaults.Solver_Precision);
  EXPECT_EQ(read.GMRES_Steps_before_restart, defaults.GMRES_Steps_before_restart);
  EXPECT_EQ(read.relative_convergence_criterion, defaults.relative_convergence_criterion);
  EXPECT_EQ(read.solve_directly, defaults.solve_directly);
  EXPECT_EQ(read.local_solver_compression_tolerance, defaults.local_solver_compression_tolerance);
  EXPECT_EQ(read.Blocks_in_x_direction, defaults.Blocks_in_x_direction);
  EXPECT_EQ(read.Blocks_in_y_direction, defaults.Blocks_in_y_direction);
  EXPECT_EQ(read.Blocks_in_z_direction, defaults.Blocks_in_z_direction);
  EXPECT_EQ(read.Sweeping_Level, defaults.Sweeping_Level);
  EXPECT_EQ(read.Cells_in_x, defaults.Cells_in_x);
  EXPECT_EQ(read.Cells_in_y, defaults.Cells_in_y);
  EXPECT_EQ(read.Cells_in_z, defaults.Cells_in_z);
  EXPECT_EQ(read.Output_transformed_solution, defaults.Output_transformed_solution);
  EXPECT_EQ(read.solver_type, defaults.solver_type);
  EXPECT_EQ(read.Point_Source_Type, defaults.Point_Source_Type);
  EXPECT_EQ(read.pec_side_walls, defaults.pec_side_walls);
  EXPECT_EQ(read.Geometry_Size_X, defaults.Geometry_Size_X);
  EXPECT_EQ(read.Geometry_Size_Y, defaults.Geometry_Size_Y);
  EXPECT_EQ(read.Geometry_Size_Z, defaults.Geometry_Size_Z);
  EXPECT_EQ(read.Epsilon_R_in_waveguide, defaults.Epsilon_R_in_waveguide);
  EXPECT_EQ(read.Epsilon_R_outside_waveguide, defaults.Epsilon_R_outside_waveguide);
  EXPECT_EQ(read.Epsilon_R_effective, defaults.Epsilon_R_effective);
  EXPECT_EQ(read.Mu_R_in_waveguide, defaults.Mu_R_in_waveguide);
  EXPECT_EQ(read.Mu_R_outside_waveguide, defaults.Mu_R_outside_waveguide);
  EXPECT_EQ(read.Amplitude_of_input_signal, defaults.Amplitude_of_input_signal);
  EXPECT_EQ(read.Nedelec_element_order, defaults.Nedelec_element_order);
  EXPECT_EQ(read.Width_of_waveguide, defaults.Width_of_waveguide);
  EXPECT_EQ(read.Height_of_waveguide, defaults.Height_of_waveguide);
  EXPECT_EQ(read.Enable_Parameter_Run, defaults.Enable_Parameter_Run);
  EXPECT_EQ(read.Min_HSIE_Order, defaults.Min_HSIE_Order);
  EXPECT_EQ(read.BoundaryCondition, defaults.BoundaryCondition);
  EXPECT_EQ(read.PML_N_Layers, defaults.PML_N_Layers);
  EXPECT_EQ(read.PML_skaling_order, defaults.PML_skaling_order);
  EXPECT_EQ(read.PML_thickness, defaults.PML_thickness);
  EXPECT_EQ(read.PML_Angle_Test, defaults.PML_Angle_Test);
  EXPECT_EQ(read.use_tapered_input_signal, defaults.use_tapered_input_signal);
  EXPECT_EQ(read.Signal_tapering_type, defaults.Signal_tapering_type);
  EXPECT_EQ(read.Perform_Convergence_Test, defaults.Perform_Convergence_Test);
  EXPECT_EQ(read.prescribe_0_on_input_side, defaults.prescribe_0_on_input_side);
  EXPECT_EQ(read.Use_Predefined_Shape, defaults.Use_Predefined_Shape);
  EXPECT_EQ(read.Number_of_Predefined_Shape, defaults.Number_of_Predefined_Shape);
  EXPECT_EQ(read.global_z_shift, defaults.global_z_shift);
  EXPECT_EQ(read.optimization_stepping_method, defaults.optimization_stepping_method);
  EXPECT_EQ(read.randomly_initialize_shape_dofs, defaults.randomly_initialize_shape_dofs);
  EXPECT_EQ(read.initial_shape_parameters, defaults.initial_shape_parameters);
  EXPECT_EQ(read.Perform_Optimization, defaults.Perform_Optimization);
  EXPECT_EQ(read.Perform_gradient_check, defaults.Perform_gradient_check);
  EXPECT_EQ(read.optimization_n_shape_steps, defaults.optimization_n_shape_steps);
  EXPECT_EQ(read.optimization_max_evaluations, defaults.optimization_max_evaluations);
  EXPECT_EQ(read.optimization_gradient_tolerance, defaults.optimization_gradient_tolerance);
  EXPECT_EQ(read.optimization_loss_tolerance, defaults.optimization_loss_tolerance);
  EXPECT_EQ(read.gradient_check_step_width, defaults.gradient_check_step_width);
  EXPECT_EQ(read.Vertical_displacement_of_waveguide, defaults.Vertical_displacement_of_waveguide);
  EXPECT_EQ(read.keep_waveguide_height_constant, defaults.keep_waveguide_height_constant);
  EXPECT_EQ(read.keep_waveguide_width_constant, defaults.keep_waveguide_width_constant);
  EXPECT_EQ(read.transformation_type, defaults.transformation_type);
}

// A parameter that is missing in the input files should get the same value as in a default constructed object (which
// is what the unit tests and Parameters.h document). For these eight entries the reader declares another default.
TEST(ParameterReaderTests, DISABLED_DefaultsOfReaderMatchDefaultsOfStructForAllEntries) {
  const Parameters read = read_parameters("", "");
  const Parameters defaults = Parameters();
  EXPECT_EQ(read.GMRES_max_steps, defaults.GMRES_max_steps);                                       // 30 and 100
  EXPECT_EQ(read.use_relative_convergence_criterion, defaults.use_relative_convergence_criterion); // true and false
  EXPECT_EQ(read.Logging_Level, defaults.Logging_Level);                                           // Production One and Debug All
  EXPECT_EQ(read.Max_HSIE_Order, defaults.Max_HSIE_Order);                                         // 21 and 10
  EXPECT_EQ(read.HSIE_polynomial_degree, defaults.HSIE_polynomial_degree);                         // 4 and 5
  EXPECT_EQ(read.Signal_coupling_method, defaults.Signal_coupling_method);                         // Dirichlet and Tapering
  EXPECT_EQ(read.PML_Sigma_Max, defaults.PML_Sigma_Max);                                           // 10 and 5
  EXPECT_EQ(read.Number_of_sectors, defaults.Number_of_sectors);                                   // 5 and 1
}

// Every entry gets a value that differs from its default and from the values of the related entries, so an entry
// that is stored in the wrong member is detected.
TEST(ParameterReaderTests, RunFileEntriesAreStoredInTheMatchingMembers) {
  const Parameters p = read_parameters(
      "solver precision = 1e-4\n"
      "GMRES restart after = 41\n"
      "GMRES maximum steps = 77\n"
      "use relative convergence criterion = false\n"
      "relative convergence criterion = 0.5\n"
      "solve directly = true\n"
      "local solver compression tolerance = 1e-5\n"
      "processes in x = 2\n"
      "processes in y = 3\n"
      "processes in z = 4\n"
      "sweeping level = 3\n"
      "cell count x = 5\n"
      "cell count y = 6\n"
      "cell count z = 7\n"
      "output transformed solution = true\n"
      "Logging Level = Debug One\n"
      "solver type = TFQMR\n",
      "");
  EXPECT_DOUBLE_EQ(p.Solver_Precision, 1e-4);
  EXPECT_EQ(p.GMRES_Steps_before_restart, 41u);
  EXPECT_EQ(p.GMRES_max_steps, 77u);
  EXPECT_FALSE(p.use_relative_convergence_criterion);
  EXPECT_DOUBLE_EQ(p.relative_convergence_criterion, 0.5);
  EXPECT_TRUE(p.solve_directly);
  EXPECT_DOUBLE_EQ(p.local_solver_compression_tolerance, 1e-5);
  EXPECT_EQ(p.Blocks_in_x_direction, 2u);
  EXPECT_EQ(p.Blocks_in_y_direction, 3u);
  EXPECT_EQ(p.Blocks_in_z_direction, 4u);
  EXPECT_EQ(p.Sweeping_Level, 3u);
  EXPECT_EQ(p.Cells_in_x, 5u);
  EXPECT_EQ(p.Cells_in_y, 6u);
  EXPECT_EQ(p.Cells_in_z, 7u);
  EXPECT_TRUE(p.Output_transformed_solution);
  EXPECT_EQ(p.Logging_Level, LoggingLevel::DEBUG_ONE);
  EXPECT_EQ(p.solver_type, SolverOptions::TFQMR);
}

TEST(ParameterReaderTests, CaseFileEntriesAreStoredInTheMatchingMembers) {
  const Parameters p = read_parameters("",
      "source type = 2\n"
      "Lateral boundary = PEC\n"
      "geometry size x = 3.5\n"
      "geometry size y = 4.5\n"
      "geometry size z = 6.5\n"
      "epsilon in = 2.1\n"
      "epsilon out = 1.3\n"
      "epsilon effective = 1.7\n"
      "mu in = 1.1\n"
      "mu out = 1.2\n"
      "fem order = 2\n"
      "signal amplitude = 0.3\n"
      "width of waveguide = 1.4\n"
      "height of waveguide = 0.9\n"
      "Enable Parameter Run = true\n"
      "Kappa 0 Real = 1.25\n"
      "Kappa 0 Imaginary = 0.75\n"
      "PML sigma max = 12.5\n"
      "HSIE polynomial degree = 9\n"
      "Min HSIE Order = 2\n"
      "Max HSIE Order = 17\n"
      "Boundary Method = PML\n"
      "PML thickness = 0.6\n"
      "PML skaling order = 4\n"
      "PML n layers = 11\n"
      "PML Test Angle = 0.35\n"
      "Prescribe input zero = true\n"
      "Predefined case number = 13\n"
      "Number of shape sectors = 8\n"
      "global z shift = 0.4\n"
      "Initialize Shape Dofs Randomly = true\n"
      "Initial shape parameters = 0.1, -0.2, 0.3\n"
      "perform optimization = true\n"
      "optimization maximum steps = 21\n"
      "optimization maximum evaluations = 33\n"
      "optimization gradient tolerance = 0.125\n"
      "optimization loss tolerance = 0.0625\n"
      "perform gradient check = true\n"
      "gradient check step width = 0.002\n"
      "vertical waveguide displacement = 0.8\n"
      "constant waveguide height = false\n"
      "constant waveguide width = false\n");
  EXPECT_EQ(p.Point_Source_Type, 2u);
  EXPECT_TRUE(p.pec_side_walls);
  EXPECT_DOUBLE_EQ(p.Geometry_Size_X, 3.5);
  EXPECT_DOUBLE_EQ(p.Geometry_Size_Y, 4.5);
  EXPECT_DOUBLE_EQ(p.Geometry_Size_Z, 6.5);
  EXPECT_DOUBLE_EQ(p.Epsilon_R_in_waveguide, 2.1);
  EXPECT_DOUBLE_EQ(p.Epsilon_R_outside_waveguide, 1.3);
  EXPECT_DOUBLE_EQ(p.Epsilon_R_effective, 1.7);
  EXPECT_DOUBLE_EQ(p.Mu_R_in_waveguide, 1.1);
  EXPECT_DOUBLE_EQ(p.Mu_R_outside_waveguide, 1.2);
  EXPECT_EQ(p.Nedelec_element_order, 2u);
  EXPECT_DOUBLE_EQ(p.Amplitude_of_input_signal, 0.3);
  EXPECT_DOUBLE_EQ(p.Width_of_waveguide, 1.4);
  EXPECT_DOUBLE_EQ(p.Height_of_waveguide, 0.9);
  EXPECT_TRUE(p.Enable_Parameter_Run);
  EXPECT_DOUBLE_EQ(p.kappa_0.real(), 1.25);
  EXPECT_DOUBLE_EQ(p.kappa_0.imag(), 0.75);
  EXPECT_DOUBLE_EQ(p.PML_Sigma_Max, 12.5);
  EXPECT_EQ(p.HSIE_polynomial_degree, 9u);
  EXPECT_EQ(p.Min_HSIE_Order, 2u);
  EXPECT_EQ(p.Max_HSIE_Order, 17u);
  EXPECT_EQ(p.BoundaryCondition, BoundaryConditionType::PML);
  EXPECT_DOUBLE_EQ(p.PML_thickness, 0.6);
  EXPECT_EQ(p.PML_skaling_order, 4u);
  EXPECT_EQ(p.PML_N_Layers, 11u);
  EXPECT_DOUBLE_EQ(p.PML_Angle_Test, 0.35);
  EXPECT_TRUE(p.prescribe_0_on_input_side);
  EXPECT_EQ(p.Number_of_Predefined_Shape, 13u);
  EXPECT_EQ(p.Number_of_sectors, 8u);
  EXPECT_DOUBLE_EQ(p.global_z_shift, 0.4);
  EXPECT_TRUE(p.randomly_initialize_shape_dofs);
  ASSERT_EQ(p.initial_shape_parameters.size(), 3u);
  EXPECT_DOUBLE_EQ(p.initial_shape_parameters[0], 0.1);
  EXPECT_DOUBLE_EQ(p.initial_shape_parameters[1], -0.2);
  EXPECT_DOUBLE_EQ(p.initial_shape_parameters[2], 0.3);
  EXPECT_TRUE(p.Perform_Optimization);
  EXPECT_EQ(p.optimization_n_shape_steps, 21u);
  EXPECT_EQ(p.optimization_max_evaluations, 33u);
  EXPECT_DOUBLE_EQ(p.optimization_gradient_tolerance, 0.125);
  EXPECT_DOUBLE_EQ(p.optimization_loss_tolerance, 0.0625);
  EXPECT_TRUE(p.Perform_gradient_check);
  EXPECT_DOUBLE_EQ(p.gradient_check_step_width, 0.002);
  EXPECT_DOUBLE_EQ(p.Vertical_displacement_of_waveguide, 0.8);
  EXPECT_FALSE(p.keep_waveguide_height_constant);
  EXPECT_FALSE(p.keep_waveguide_width_constant);
}

TEST(ParameterReaderTests, SelectionsAreTranslatedToEnums) {
  EXPECT_EQ(read_parameters("Logging Level = Debug All", "").Logging_Level, LoggingLevel::DEBUG_ALL);
  EXPECT_EQ(read_parameters("Logging Level = Debug One", "").Logging_Level, LoggingLevel::DEBUG_ONE);
  EXPECT_EQ(read_parameters("Logging Level = Production All", "").Logging_Level, LoggingLevel::PRODUCTION_ALL);
  EXPECT_EQ(read_parameters("Logging Level = Production One", "").Logging_Level, LoggingLevel::PRODUCTION_ONE);
  EXPECT_EQ(read_parameters("solver type = MINRES", "").solver_type, SolverOptions::MINRES);
  EXPECT_EQ(read_parameters("solver type = CG", "").solver_type, SolverOptions::S_CG);
  EXPECT_EQ(read_parameters("solver type = PCONLY", "").solver_type, SolverOptions::PCONLY);
  EXPECT_EQ(read_parameters("solver type = BICGS", "").solver_type, SolverOptions::BICGS);
  EXPECT_EQ(read_parameters("", "Boundary Method = HSIE").BoundaryCondition, BoundaryConditionType::HSIE);
  EXPECT_EQ(read_parameters("", "Boundary Method = PML").BoundaryCondition, BoundaryConditionType::PML);
  EXPECT_FALSE(read_parameters("", "Lateral boundary = Absorbing").pec_side_walls);
  EXPECT_EQ(read_parameters("", "Signal tapering type = C0").Signal_tapering_type, SignalTaperingType::C0);
  EXPECT_EQ(read_parameters("", "Signal tapering type = C1").Signal_tapering_type, SignalTaperingType::C1);
  EXPECT_EQ(read_parameters("", "Optimization Algorithm = Steepest").optimization_stepping_method, SteppingMethod::Steepest);
  EXPECT_EQ(read_parameters("", "Optimization Algorithm = BFGS").optimization_stepping_method, SteppingMethod::BFGS);
  EXPECT_EQ(read_parameters("", "transformation type = Waveguide Transformation").transformation_type, TransformationType::WavegeuideTransformationType);
  EXPECT_EQ(read_parameters("", "transformation type = Angle Waveguide Transformation").transformation_type, TransformationType::AngleWaveguideTransformationType);
  EXPECT_EQ(read_parameters("", "transformation type = Bend Transformation").transformation_type, TransformationType::BendTransformationType);
}

// The two members that describe the coupling of the input signal are set from the same entry and have to agree.
TEST(ParameterReaderTests, InputSignalMethodSetsBothMembers) {
  const Parameters taper = read_parameters("", "Input Signal Method = Taper");
  EXPECT_EQ(taper.Signal_coupling_method, SignalCouplingMethod::Tapering);
  EXPECT_TRUE(taper.use_tapered_input_signal);
  const Parameters dirichlet = read_parameters("", "Input Signal Method = Dirichlet");
  EXPECT_EQ(dirichlet.Signal_coupling_method, SignalCouplingMethod::Dirichlet);
  EXPECT_FALSE(dirichlet.use_tapered_input_signal);
}

TEST(ParameterReaderTests, PredefinedShapeSelectsThePredefinedShapeTransformation) {
  const Parameters p = read_parameters("", "Use predefined shape = true\ntransformation type = Bend Transformation");
  EXPECT_TRUE(p.Use_Predefined_Shape);
  EXPECT_EQ(p.transformation_type, TransformationType::PredefinedShapeTransformationType);
}

TEST(ParameterReaderTests, ConvergenceSequenceIsOnlyReadForConvergenceRuns) {
  const Parameters without = read_parameters("", "convergence sequence cell count = 2,4,8");
  EXPECT_FALSE(without.Perform_Convergence_Test);
  EXPECT_TRUE(without.convergence_cell_counts.empty());
  const Parameters with = read_parameters("", "perform convergence test = true\nconvergence sequence cell count = 2,4,8,12");
  EXPECT_TRUE(with.Perform_Convergence_Test);
  EXPECT_EQ(with.convergence_cell_counts, (std::vector<unsigned int>{2, 4, 8, 12}));
  EXPECT_EQ(with.convergence_max_cells, 12u);
}

TEST(ParameterReaderTests, EmptyListOfInitialShapeParameters) {
  EXPECT_TRUE(read_parameters("", "Initial shape parameters = ").initial_shape_parameters.empty());
}

TEST(ParameterReaderTests, InvalidInputIsRejected) {
  EXPECT_ANY_THROW(read_parameters("", "Boundary Method = Neither"));
  EXPECT_ANY_THROW(read_parameters("cell count x = many", ""));
  EXPECT_ANY_THROW(read_parameters("", "no such entry = 1"));
  EXPECT_ANY_THROW(read_parameters("", "optimization maximum steps = 0"));
}

// The run file entry "kappa angle" is declared with a description but its value is never stored: the member
// kappa_0_angle keeps its default 1.0. (The alternative fix is to remove the entry.)
TEST(ParameterReaderTests, DISABLED_KappaAngleEntryIsStored) {
  EXPECT_DOUBLE_EQ(read_parameters("kappa angle = 0.5", "").kappa_0_angle, 0.5);
}

// Ranks are numbered x-first: rank = i_x + n_x * (i_y + n_y * i_z).
TEST(ParametersCompleteDataTests, BlockIndicesFollowFromTheRank) {
  ScopedGlobalParams global_params;
  const unsigned int n_x = 2, n_y = 3, n_z = 4;
  std::set<std::array<unsigned int, 3>> seen;
  for (unsigned int rank = 0; rank < n_x * n_y * n_z; rank++) {
    Parameters p = parameters_for_blocks(n_x, n_y, n_z, rank);
    complete_quietly(p);
    ASSERT_LT(p.Index_in_x_direction, n_x);
    ASSERT_LT(p.Index_in_y_direction, n_y);
    ASSERT_LT(p.Index_in_z_direction, n_z);
    EXPECT_EQ(p.Index_in_x_direction + n_x * (p.Index_in_y_direction + n_y * p.Index_in_z_direction), rank);
    seen.insert({p.Index_in_x_direction, p.Index_in_y_direction, p.Index_in_z_direction});
  }
  EXPECT_EQ(seen.size(), n_x * n_y * n_z);
}

TEST(ParametersCompleteDataTests, WrongNumberOfProcessesIsAnError) {
  ScopedGlobalParams global_params;
  Parameters p = parameters_for_blocks(2, 2, 3, 0);
  p.NumberProcesses = 8;
  try {
    complete_quietly(p);
    FAIL() << "No exception was thrown.";
  } catch (const std::runtime_error & error) {
    // The message has to name the required number of processes.
    EXPECT_NE(std::string(error.what()).find("12"), std::string::npos) << error.what();
  }
}

TEST(ParametersCompleteDataTests, ParameterRunIgnoresTheNumberOfProcesses) {
  ScopedGlobalParams global_params;
  Parameters p = parameters_for_blocks(2, 2, 3, 0);
  p.NumberProcesses = 1;
  p.Enable_Parameter_Run = true;
  EXPECT_NO_THROW(complete_quietly(p));
  EXPECT_EQ(p.Index_in_x_direction, 0u);
  EXPECT_EQ(p.Index_in_y_direction, 0u);
  EXPECT_EQ(p.Index_in_z_direction, 0u);
}

TEST(ParametersCompleteDataTests, AngularFrequencyAndSectorThickness) {
  ScopedGlobalParams global_params;
  Parameters p = parameters_for_blocks(1, 1, 1, 0);
  p.Lambda = 1.25;
  p.Geometry_Size_Z = 9.0;
  p.Number_of_sectors = 6;
  complete_quietly(p);
  EXPECT_NEAR(p.Omega, 2 * pi / 1.25, 1e-12);
  EXPECT_DOUBLE_EQ(p.Sector_thickness, 1.5);
}

TEST(ParametersCompleteDataTests, SourceTypeSelectsTheSourceField) {
  ScopedGlobalParams global_params;
  Parameters p = parameters_for_blocks(1, 1, 1, 0);
  p.Lambda = 1.25;
  p.Epsilon_R_outside_waveguide = 2.25;
  p.Geometry_Size_X = 3.0;
  const double k = 2 * pi / 1.25 * 1.5;

  p.Point_Source_Type = 1;
  complete_quietly(p);
  EXPECT_NE(dynamic_cast<PointSourceFieldCosCos *>(p.source_field), nullptr);
  delete p.source_field;

  // The dipole radiates with the wavenumber of the medium outside of the waveguide.
  p.Point_Source_Type = 2;
  complete_quietly(p);
  PointSourceFieldHertz * dipole = dynamic_cast<PointSourceFieldHertz *>(p.source_field);
  ASSERT_NE(dipole, nullptr);
  EXPECT_NEAR(dipole->k, k, 1e-12);
  delete p.source_field;

  p.Point_Source_Type = 4;
  complete_quietly(p);
  MetallicWaveguideMode * mode = dynamic_cast<MetallicWaveguideMode *>(p.source_field);
  ASSERT_NE(mode, nullptr);
  EXPECT_DOUBLE_EQ(mode->width, 3.0);
  EXPECT_NEAR(mode->beta, std::sqrt(k * k - (pi / 3.0) * (pi / 3.0)), 1e-9);
  delete p.source_field;
}

// The predefined shapes are the lines of Modes/test.csv. The nodes are shifted such that the shape starts at z = 0.
TEST(ParametersCompleteDataTests, PredefinedShapeIsLoadedFromTheShapeFile) {
  ScopedGlobalParams global_params;
  const std::vector<std::string> lines = helpers_a::read_lines(repository_path("Modes/test.csv"));
  ASSERT_GT(lines.size(), 2u);
  Parameters p = parameters_for_blocks(1, 1, 1, 0);
  p.Use_Predefined_Shape = true;
  p.Number_of_Predefined_Shape = 2;
  complete_quietly(p);
  const std::vector<std::string> tokens = split(lines[2], ",");
  const unsigned int n_nodes = std::stoi(tokens[0]);
  ASSERT_GE(tokens.size(), 1 + 3 * n_nodes);
  ASSERT_EQ(p.sd.Sectors, static_cast<int>(n_nodes));
  ASSERT_EQ(p.sd.z.size(), n_nodes);
  ASSERT_EQ(p.sd.m.size(), n_nodes);
  ASSERT_EQ(p.sd.v.size(), n_nodes);
  EXPECT_EQ(p.sd.z[0], 0.0);
  for (unsigned int i = 0; i < n_nodes; i++) {
    EXPECT_NEAR(p.sd.z[i], std::stod(tokens[1 + i]) - std::stod(tokens[1]), 1e-12);
    EXPECT_DOUBLE_EQ(p.sd.m[i], std::stod(tokens[1 + n_nodes + i]));
    EXPECT_DOUBLE_EQ(p.sd.v[i], std::stod(tokens[1 + 2 * n_nodes + i]));
  }
  EXPECT_DOUBLE_EQ(p.Sector_thickness, p.sd.z[1] - p.sd.z[0]);
}

// The shape description of a straight waveguide has to span the z-range of the parameter object that is completed.
// ShapeDescription::SetStraight reads the length and the shift from the global object GlobalParams instead, which
// still holds the previous (during the start of the program: the default) values when GetParameters completes the
// new object.
TEST(ParametersCompleteDataTests, DISABLED_StraightShapeSpansTheZRangeOfTheCompletedObject) {
  ScopedGlobalParams global_params;
  Parameters p = parameters_for_blocks(1, 1, 1, 0);
  p.Geometry_Size_Z = 8.0;
  p.global_z_shift = 1.0;
  complete_quietly(p);
  ASSERT_EQ(p.sd.z.size(), 2u);
  EXPECT_DOUBLE_EQ(p.sd.z[0], 1.0);
  EXPECT_DOUBLE_EQ(p.sd.z[1], 9.0);
}

// check_validity should report if the mesh does not resolve the side walls of the waveguide core. Here the grid
// lines in x are at -2.5, -0.83, 0.83 and 2.5 and the core walls at -1 and 1. The function prints a warning
// but returns true in every case. (It also iterates with unsigned integers over the coordinates, so it does not
// terminate if the mesh width is smaller than 1; the values of this test avoid that.)
TEST(ParametersCheckValidityTests, DISABLED_UnresolvedWaveguideIsReported) {
  ScopedGlobalParams global_params;
  ScopedGeometry geometry;
  GlobalParams.Cells_in_x = 3;
  GlobalParams.Cells_in_y = 3;
  GlobalParams.Width_of_waveguide = 2.0;
  GlobalParams.Height_of_waveguide = 1.8;
  Geometry.global_x_range = std::make_pair(-2.5, 2.5);
  Geometry.global_y_range = std::make_pair(-2.5, 2.5);
  EXPECT_FALSE(GlobalParams.check_validity());
}

TEST(ParameterOverrideTests, NewObjectHasNoOverrides) {
  ParameterOverride override_object;
  EXPECT_FALSE(override_object.has_overrides);
}

TEST(ParameterOverrideTests, ValidatesTheFormat) {
  ParameterOverride override_object;
  EXPECT_TRUE(override_object.validate("pml_order=4"));
  EXPECT_TRUE(override_object.validate("pml_order=4;pml_sigma_max=10"));
  EXPECT_FALSE(override_object.validate(""));
  EXPECT_FALSE(override_object.validate("pml_order"));
  EXPECT_FALSE(override_object.validate("pml_order=4;pml_sigma_max"));
  EXPECT_FALSE(override_object.validate("pml_order=4=5"));
}

TEST(ParameterOverrideTests, InvalidInputIsNotStored) {
  ScopedGlobalParams global_params;
  ParameterOverride override_object;
  EXPECT_FALSE(override_object.read("pml_order=4;pml_sigma_max"));
  EXPECT_FALSE(override_object.has_overrides);
  Parameters p;
  override_object.perform_on(p);
  EXPECT_EQ(p.PML_skaling_order, Parameters().PML_skaling_order);
}

TEST(ParameterOverrideTests, OverridesAreAppliedToTheMatchingMembers) {
  ScopedGlobalParams global_params;
  ParameterOverride override_object;
  ASSERT_TRUE(override_object.read("n_pml_cells=13;pml_sigma_max=7.5;pml_order=5;solver_type=MINRES;geometry_size_z=12.5;processes_in_z=6;predefined_case_number=17"));
  EXPECT_TRUE(override_object.has_overrides);
  Parameters p;
  override_object.perform_on(p);
  EXPECT_EQ(p.PML_N_Layers, 13u);
  EXPECT_DOUBLE_EQ(p.PML_Sigma_Max, 7.5);
  EXPECT_EQ(p.PML_skaling_order, 5u);
  EXPECT_EQ(p.solver_type, SolverOptions::MINRES);
  EXPECT_DOUBLE_EQ(p.Geometry_Size_Z, 12.5);
  EXPECT_EQ(p.Blocks_in_z_direction, 6u);
  EXPECT_EQ(p.Number_of_Predefined_Shape, 17u);
}

TEST(ParameterOverrideTests, SystemLengthIsTheGeometrySizeInZ) {
  ScopedGlobalParams global_params;
  ParameterOverride override_object;
  ASSERT_TRUE(override_object.read("system_length=3.25"));
  Parameters p;
  override_object.perform_on(p);
  EXPECT_DOUBLE_EQ(p.Geometry_Size_Z, 3.25);
}

TEST(ParameterOverrideTests, OtherMembersAreNotChanged) {
  ScopedGlobalParams global_params;
  ParameterOverride override_object;
  ASSERT_TRUE(override_object.read("pml_order=5;unknown_key=3"));
  Parameters p;
  override_object.perform_on(p);
  const Parameters defaults = Parameters();
  EXPECT_EQ(p.PML_skaling_order, 5u);
  EXPECT_EQ(p.PML_N_Layers, defaults.PML_N_Layers);
  EXPECT_EQ(p.PML_Sigma_Max, defaults.PML_Sigma_Max);
  EXPECT_EQ(p.Geometry_Size_Z, defaults.Geometry_Size_Z);
  EXPECT_EQ(p.Blocks_in_z_direction, defaults.Blocks_in_z_direction);
  EXPECT_EQ(p.solver_type, defaults.solver_type);
}

// GetParameters reads both files, applies the overrides and then completes the data. The run file of this test asks
// for two processes, which is an error in the single process of the unit tests unless the override corrects it.
TEST(GetParametersTests, OverridesAreAppliedBeforeTheDataIsCompleted) {
  ScopedGlobalParams global_params;
  TemporaryDirectory directory("get_parameters");
  helpers_a::write_file(directory.file("run.prm"), "subsection Run parameters\n  set processes in z = 2\n  set cell count z = 9\nend\n");
  helpers_a::write_file(directory.file("case.prm"), "subsection Case parameters\n  set source type = 1\n  set geometry size z = 7.0\n  set Number of shape sectors = 2\nend\n");
  ParameterOverride no_override;
  testing::internal::CaptureStdout();
  EXPECT_THROW(GetParameters(directory.file("run.prm"), directory.file("case.prm"), no_override), std::runtime_error);
  ParameterOverride with_override;
  ASSERT_TRUE(with_override.read("processes_in_z=1;pml_order=6"));
  Parameters p;
  EXPECT_NO_THROW(p = GetParameters(directory.file("run.prm"), directory.file("case.prm"), with_override));
  testing::internal::GetCapturedStdout();
  EXPECT_EQ(p.Blocks_in_z_direction, 1u);
  EXPECT_EQ(p.PML_skaling_order, 6u);
  EXPECT_EQ(p.Cells_in_z, 9u);
  EXPECT_EQ(p.MPI_Rank, 0u);
  EXPECT_EQ(p.NumberProcesses, 1u);
  EXPECT_EQ(p.Index_in_z_direction, 0u);
  EXPECT_DOUBLE_EQ(p.Sector_thickness, 3.5);
  EXPECT_NE(dynamic_cast<PointSourceFieldCosCos *>(p.source_field), nullptr);
  delete p.source_field;
}

TEST(PointValTests, DefaultIsZero) {
  const PointVal value;
  EXPECT_EQ(value.Ex, ComplexNumber(0, 0));
  EXPECT_EQ(value.Ey, ComplexNumber(0, 0));
  EXPECT_EQ(value.Ez, ComplexNumber(0, 0));
}

// The arguments are the three real parts followed by the three imaginary parts.
TEST(PointValTests, ConstructorAndSetTakeRealPartsFirst) {
  PointVal value(1, 2, 3, 4, 5, 6);
  EXPECT_EQ(value.Ex, ComplexNumber(1, 4));
  EXPECT_EQ(value.Ey, ComplexNumber(2, 5));
  EXPECT_EQ(value.Ez, ComplexNumber(3, 6));
  value.set(-1, -2, -3, -4, -5, -6);
  EXPECT_EQ(value.Ex, ComplexNumber(-1, -4));
  EXPECT_EQ(value.Ey, ComplexNumber(-2, -5));
  EXPECT_EQ(value.Ez, ComplexNumber(-3, -6));
}

TEST(PointValTests, RescaleMultipliesAllComponents) {
  PointVal value(1, 2, 3, 4, 5, 6);
  value.rescale(-0.5);
  EXPECT_EQ(value.Ex, ComplexNumber(-0.5, -2));
  EXPECT_EQ(value.Ey, ComplexNumber(-1, -2.5));
  EXPECT_EQ(value.Ez, ComplexNumber(-1.5, -3));
}

TEST(ShapeDescriptionTests, SetByStringReadsNodesAndShiftsZToZero) {
  ShapeDescription description;
  description.SetByString("3,10,12,15.5,0.1,0.2,0.3,-1,-2,-3");
  EXPECT_EQ(description.Sectors, 3);
  EXPECT_EQ(description.z, (std::vector<double>{0, 2, 5.5}));
  EXPECT_EQ(description.m, (std::vector<double>{0.1, 0.2, 0.3}));
  EXPECT_EQ(description.v, (std::vector<double>{-1, -2, -3}));
}

TEST(ShapeDescriptionTests, StraightShapeSpansTheGlobalZRangeWithoutDisplacement) {
  ScopedGlobalParams global_params;
  GlobalParams.Geometry_Size_Z = 6.0;
  GlobalParams.global_z_shift = 0.5;
  ShapeDescription description;
  description.SetStraight();
  EXPECT_EQ(description.z, (std::vector<double>{0.5, 6.5}));
  EXPECT_EQ(description.m, (std::vector<double>{0, 0}));
}

TEST(CosCosFieldTests, ComponentsMatchTheFormula) {
  const PointSourceFieldCosCos field;
  const double c = 1.0 / std::sqrt(2.0);
  const Position p(0.3, -0.8, 1.9);
  NumericVectorLocal values(3);
  field.vector_value(p, values);
  EXPECT_NEAR(values[0].real(), std::cos(c * p[1]) * std::cos(c * p[2]), 1e-14);
  EXPECT_NEAR(values[1].real(), std::cos(c * p[0]) * std::cos(c * p[2]), 1e-14);
  EXPECT_NEAR(values[2].real(), std::cos(c * p[0]) * std::cos(c * p[1]), 1e-14);
  for (unsigned int component = 0; component < 3; component++) {
    EXPECT_EQ(values[component].imag(), 0.0);
    EXPECT_EQ(field.value(p, component), values[component]);
  }
  field.vector_value(Position(0, 0, 0), values);
  for (unsigned int component = 0; component < 3; component++) EXPECT_EQ(values[component], ComplexNumber(1, 0));
}

// Component i of the field does not depend on coordinate i, so the field is divergence free, and every component
// fulfills the Helmholtz equation with wavenumber 1: curl curl E = E.
TEST(CosCosFieldTests, IsDivergenceFreeAndSolvesTheCurlCurlEquationWithWavenumberOne) {
  const PointSourceFieldCosCos function;
  const Field field = field_of(function);
  const double h = 1e-3;
  for (const Position & p : {Position(0.3, -0.8, 1.9), Position(-1.2, 0.4, 0.1)}) {
    EXPECT_LT(std::abs(divergence(field, p, h)), 1e-9);
    for (unsigned int component = 0; component < 3; component++) {
      EXPECT_LT(std::abs(laplace(field, p, component, h) + field(p, component)), 1e-6);
    }
  }
}

// vector_curl has to be the curl of vector_value. The code returns only one of the two terms of every component
// (for example c cos(c x) sin(c z) instead of c cos(c x) (sin(c z) - sin(c y)) for the first one).
TEST(CosCosFieldTests, DISABLED_VectorCurlIsTheCurlOfTheField) {
  PointSourceFieldCosCos function;
  const Field field = field_of(function);
  const Position p(0.3, -0.8, 1.9);
  NumericVectorLocal computed(3);
  function.vector_curl(p, computed);
  const dealii::Tensor<1, 3, ComplexNumber> expected = curl(field, p, 1e-4);
  for (unsigned int component = 0; component < 3; component++) {
    EXPECT_LT(std::abs(computed[component] - expected[component]), 1e-6) << component;
  }
}

TEST(HertzDipoleTests, ValueIsAComponentOfVectorValue) {
  const PointSourceFieldHertz dipole(2.3);
  const Position p(0.4, -0.7, 0.9);
  NumericVectorLocal values(3);
  dipole.vector_value(p, values);
  for (unsigned int component = 0; component < 3; component++) EXPECT_EQ(dipole.value(p, component), values[component]);
}

// Away from the dipole the field is a solution of the homogeneous Maxwell equations with wavenumber k:
// div E = 0 and curl curl E - k^2 E = - laplace E - k^2 E = 0.
TEST(HertzDipoleTests, SolvesTheHomogeneousMaxwellEquationsAwayFromTheOrigin) {
  const double k = 2.3;
  const PointSourceFieldHertz dipole(k);
  const Field field = field_of(dipole);
  const double h = 2e-3;
  for (const Position & p : {Position(0.9, -0.7, 1.1), Position(-1.4, 0.3, -0.6), Position(0.2, 1.6, 0.5)}) {
    dealii::Tensor<1, 3, ComplexNumber> value, residual;
    for (unsigned int component = 0; component < 3; component++) {
      value[component] = field(p, component);
      residual[component] = laplace(field, p, component, h) + k * k * value[component];
    }
    ASSERT_GT(norm(value), 1e-3);
    EXPECT_LT(norm(residual), 1e-3 * k * k * norm(value));
    EXPECT_LT(std::abs(divergence(field, p, h)), 1e-4 * k * norm(value));
  }
}

// On the axis of the dipole only the near field (3 n (n . p) - p) (1 / r^2 - i k / r) = 2 e_z (1 / r^2 - i k / r)
// remains: E = i / (4 pi) exp(i k r) / r * 2 (1 / r^2 - i k / r) e_z.
TEST(HertzDipoleTests, FieldOnTheDipoleAxis) {
  const double k = 1.7;
  const PointSourceFieldHertz dipole(k);
  for (double z : {0.8, -1.3, 4.0}) {
    const double r = std::abs(z);
    NumericVectorLocal values(3);
    dipole.vector_value(Position(0, 0, z), values);
    const ComplexNumber expected = imaginary_unit / (4 * pi) * std::exp(imaginary_unit * k * r) / r * 2.0 * (1 / (r * r) - imaginary_unit * k / r);
    EXPECT_LT(std::abs(values[0]), 1e-14);
    EXPECT_LT(std::abs(values[1]), 1e-14);
    EXPECT_LT(std::abs(values[2] - expected), 1e-10 * std::abs(expected));
  }
}

// In the far field the wave is transverse and decays like 1 / r: E ~ i k^2 / (4 pi) exp(i k r) / r (n x p) x n.
TEST(HertzDipoleTests, FarFieldIsATransverseSphericalWave) {
  const double k = 2.0;
  const PointSourceFieldHertz dipole(k);
  const double r = 5000;
  const Position n(std::sin(1.1) * std::cos(0.4), std::sin(1.1) * std::sin(0.4), std::cos(1.1));
  const Position p = r * n;
  NumericVectorLocal values(3);
  dipole.vector_value(p, values);
  dealii::Tensor<1, 3, double> e_z;
  e_z[2] = 1.0;
  const dealii::Tensor<1, 3, double> n_tensor = n;
  const dealii::Tensor<1, 3, double> direction = dealii::cross_product_3d(dealii::cross_product_3d(n_tensor, e_z), n_tensor);
  const ComplexNumber amplitude = imaginary_unit * k * k / (4 * pi) * std::exp(imaginary_unit * k * r) / r;
  ComplexNumber radial(0, 0);
  for (unsigned int component = 0; component < 3; component++) {
    EXPECT_LT(std::abs(values[component] - amplitude * direction[component]), 1e-3 * std::abs(amplitude)) << component;
    radial += values[component] * n[component];
  }
  EXPECT_LT(std::abs(radial), 1e-3 * std::abs(amplitude));
}

// The field of a dipole in the z-direction is symmetric under rotations around the z-axis and under z -> -z
// (E_z is even, the transverse components are odd).
TEST(HertzDipoleTests, Symmetries) {
  const PointSourceFieldHertz dipole(1.3);
  NumericVectorLocal a(3), b(3);
  dipole.vector_value(Position(0.6, 0.0, 0.9), a);
  dipole.vector_value(Position(0.0, 0.6, 0.9), b);
  EXPECT_LT(std::abs(a[0] - b[1]), 1e-14);
  EXPECT_LT(std::abs(a[1]), 1e-14);
  EXPECT_LT(std::abs(b[0]), 1e-14);
  EXPECT_LT(std::abs(a[2] - b[2]), 1e-14);
  dipole.vector_value(Position(0.6, 0.0, -0.9), b);
  EXPECT_LT(std::abs(a[0] + b[0]), 1e-14);
  EXPECT_LT(std::abs(a[2] - b[2]), 1e-14);
}

// The field is singular at the origin. vector_value detects positions inside the cube of edge length cell_diameter
// around the origin and writes zero for them, but then continues and overwrites the zero with the singular formula
// (NaN at the origin).
TEST(HertzDipoleTests, DISABLED_FieldIsSetToZeroInsideTheCellAroundTheOrigin) {
  PointSourceFieldHertz dipole(1.3);
  dipole.set_cell_diameter(0.1);
  NumericVectorLocal values(3);
  for (const Position & p : {Position(0, 0, 0), Position(0.01, -0.02, 0.03)}) {
    dipole.vector_value(p, values);
    EXPECT_EQ(values.l2_norm(), 0.0) << p;
    EXPECT_EQ(dipole.value(p, 2), ComplexNumber(0, 0)) << p;
  }
}

// vector_curl has to be the curl of vector_value. The code returns (0, z E_x f, y E_x f) with a scalar factor f.
TEST(HertzDipoleTests, DISABLED_VectorCurlIsTheCurlOfTheField) {
  PointSourceFieldHertz dipole(1.3);
  const Field field = field_of(dipole);
  const Position p(0.9, -0.7, 1.1);
  NumericVectorLocal computed(3);
  dipole.vector_curl(p, computed);
  const dealii::Tensor<1, 3, ComplexNumber> expected = curl(field, p, 1e-4);
  ASSERT_GT(norm(expected), 1e-3);
  for (unsigned int component = 0; component < 3; component++) {
    EXPECT_LT(std::abs(computed[component] - expected[component]), 1e-5 * norm(expected)) << component;
  }
}

TEST(MetallicWaveguideModeTests, PropagationConstantFulfillsTheDispersionRelation) {
  const double k = 3.1, width = 2.5;
  const MetallicWaveguideMode mode(k, width);
  EXPECT_DOUBLE_EQ(mode.width, width);
  EXPECT_NEAR(mode.beta * mode.beta + (pi / width) * (pi / width), k * k, 1e-9);
  EXPECT_GT(mode.beta, 0);
}

// E = (0, cos(pi x / a), 0) exp(i beta z): the tangential components vanish on the walls x = +-a/2 (E_y) and
// y = const (E_x, E_z).
TEST(MetallicWaveguideModeTests, ProfilePhaseAndBoundaryConditions) {
  const double k = 3.1, width = 2.5;
  const MetallicWaveguideMode mode(k, width);
  NumericVectorLocal values(3);
  mode.vector_value(Position(0.4, 0.3, 1.7), values);
  EXPECT_EQ(values[0], ComplexNumber(0, 0));
  EXPECT_EQ(values[2], ComplexNumber(0, 0));
  EXPECT_LT(std::abs(values[1] - std::cos(pi * 0.4 / width) * std::exp(imaginary_unit * mode.beta * 1.7)), 1e-10);
  EXPECT_EQ(mode.value(Position(0.4, 0.3, 1.7), 1), values[1]);
  EXPECT_EQ(mode.value(Position(0.4, 0.3, 1.7), 0), ComplexNumber(0, 0));
  EXPECT_EQ(mode.value(Position(0.4, 0.3, 1.7), 2), ComplexNumber(0, 0));
  EXPECT_LT(std::abs(mode.value(Position(0, 0.3, 0), 1) - 1.0), 1e-14);
  for (double z : {0.0, 0.8}) {
    EXPECT_LT(std::abs(mode.value(Position(width / 2, 0.3, z), 1)), 1e-10);
    EXPECT_LT(std::abs(mode.value(Position(-width / 2, 0.3, z), 1)), 1e-10);
  }
  // The field does not depend on y.
  EXPECT_EQ(mode.value(Position(0.4, -1.0, 1.7), 1), values[1]);
}

TEST(MetallicWaveguideModeTests, SolvesTheHomogeneousMaxwellEquations) {
  const double k = 3.1, width = 2.5;
  const MetallicWaveguideMode mode(k, width);
  const Field field = field_of(mode);
  const double h = 1e-3;
  for (const Position & p : {Position(0.4, 0.3, 1.7), Position(-0.9, -0.2, 0.2)}) {
    const ComplexNumber value = field(p, 1);
    EXPECT_LT(std::abs(laplace(field, p, 1, h) + k * k * value), 1e-5 * k * k * std::abs(value));
    EXPECT_LT(std::abs(divergence(field, p, h)), 1e-9);
  }
}
