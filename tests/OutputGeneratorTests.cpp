// Tests for the generators of the gnuplot scripts for residual histories and convergence studies. The scripts are
// read back and compared with the values that were handed to the generators. gnuplot itself is not started.
#include <gtest/gtest.h>
#include <cmath>
#include <filesystem>
#include <map>
#include <regex>
#include "../Code/GlobalObjects/GlobalObjects.h"
#include "../Code/OutputGenerators/Images/ConvergenceOutputGenerator.h"
#include "../Code/OutputGenerators/Images/ResidualOutputGenerator.h"
#include "./HelpersA.h"

namespace {

  using helpers_a::ScopedGlobalParams;
  using helpers_a::ScopedOutputFolder;
  using helpers_a::TemporaryDirectory;

  // The arrays of a gnuplot script: "array name[size]" and "name[index] = value" with indices that start at 1.
  struct GnuplotArrays {
    std::map<std::string, unsigned int> declared_sizes;
    std::map<std::string, std::map<unsigned int, double>> values;

    explicit GnuplotArrays(const std::vector<std::string> & lines) {
      const std::regex declaration("^array (\\w+)\\[(\\d+)\\]$");
      const std::regex assignment("^(\\w+)\\[(\\d+)\\] = (.+)$");
      std::smatch match;
      for (const std::string & line : lines) {
        if (std::regex_match(line, match, declaration)) declared_sizes[match[1]] = std::stoul(match[2]);
        if (std::regex_match(line, match, assignment)) values[match[1]][std::stoul(match[2])] = std::stod(match[3]);
      }
    }

    // The values of an array in the order of the indices. Fails if an index in 1..size is missing.
    std::vector<double> array(const std::string & name) const {
      std::vector<double> ret;
      if (declared_sizes.count(name) == 0) {
        ADD_FAILURE() << "The script does not declare the array " << name;
        return ret;
      }
      const unsigned int size = declared_sizes.at(name);
      if (size == 0) return ret;
      if (values.count(name) == 0 || values.at(name).size() != size) {
        ADD_FAILURE() << "The array " << name << " is declared with " << size << " values but another number is set.";
        return ret;
      }
      for (unsigned int i = 1; i <= size; i++) {
        if (values.at(name).count(i) == 0) {
          ADD_FAILURE() << "The array " << name << " has no value " << i;
          return ret;
        }
        ret.push_back(values.at(name).at(i));
      }
      return ret;
    }
  };

  // The values are written with the default precision of streams (6 significant digits).
  void expect_equal_up_to_output_precision(const std::vector<double> & read, const std::vector<double> & expected) {
    ASSERT_EQ(read.size(), expected.size());
    for (unsigned int i = 0; i < read.size(); i++) {
      EXPECT_NEAR(read[i], expected[i], 1e-5 * std::abs(expected[i])) << i;
    }
  }

  bool contains_line(const std::vector<std::string> & lines, const std::string & line) {
    return std::find(lines.begin(), lines.end(), line) != lines.end();
  }

  // Globals for the generators: the rank appears in the file names and the level defines the indentation.
  struct GeneratorEnvironment {
    GeneratorEnvironment() : directory("output_generators"), folder(directory.path()) {
      GlobalParams.MPI_Rank = 0;
      GlobalParams.Sweeping_Level = 2;
    }
    ScopedGlobalParams params;
    TemporaryDirectory directory;
    ScopedOutputFolder folder;
  };

}

TEST(ResidualOutputGeneratorTests, ValuesAreAddedToTheLatestSeries) {
  GeneratorEnvironment environment;
  ResidualOutputGenerator generator("residuals", "Residuals", 0, 2, -1);
  EXPECT_TRUE(generator.data_series.empty());
  generator.new_series("first solve");
  generator.push_value(1.0);
  generator.push_value(0.1);
  generator.close_current_series();
  generator.new_series("second solve");
  generator.push_value(2.0);
  ASSERT_EQ(generator.data_series.size(), 2u);
  EXPECT_EQ(generator.data_series[0].name, "first solve");
  EXPECT_EQ(generator.data_series[0].values, (std::vector<double>{1.0, 0.1}));
  EXPECT_TRUE(generator.data_series[0].is_closed);
  EXPECT_EQ(generator.data_series[1].name, "second solve");
  EXPECT_EQ(generator.data_series[1].values, (std::vector<double>{2.0}));
  EXPECT_FALSE(generator.data_series[1].is_closed);
}

TEST(ResidualOutputGeneratorTests, ValueWithoutSeriesIsIgnored) {
  GeneratorEnvironment environment;
  ResidualOutputGenerator generator("residuals", "Residuals", 0, 2, -1);
  generator.push_value(1.0);
  EXPECT_TRUE(generator.data_series.empty());
}

TEST(ResidualOutputGeneratorTests, ScriptContainsAllSeries) {
  GeneratorEnvironment environment;
  GlobalParams.MPI_Rank = 3;
  ResidualOutputGenerator generator("residuals_level", "Convergence on level 2", 0, 2, -1);
  const std::vector<double> first = {1.0, 0.123456, 1.5e-3, 2.25e-7, 3.0e-12};
  const std::vector<double> second = {4.0, 0.5};
  generator.new_series("first solve");
  for (double value : first) generator.push_value(value);
  generator.close_current_series();
  generator.new_series("second solve");
  for (double value : second) generator.push_value(value);
  generator.close_current_series();
  generator.new_series("no steps");
  generator.write_gnuplot_file();

  // The file names contain the rank of the process.
  const std::string script = environment.directory.file("residuals_level3.plt");
  ASSERT_TRUE(std::filesystem::exists(script));
  const std::vector<std::string> lines = helpers_a::read_lines(script);
  const GnuplotArrays arrays(lines);
  EXPECT_EQ(arrays.declared_sizes.size(), 3u);
  expect_equal_up_to_output_precision(arrays.array("series0"), first);
  expect_equal_up_to_output_precision(arrays.array("series1"), second);
  EXPECT_TRUE(arrays.array("series2").empty());
  EXPECT_TRUE(contains_line(lines, "set output \"" + environment.directory.file("residuals_level3.png") + "\""));
  EXPECT_TRUE(contains_line(lines, "set title \"Convergence on level 2\""));
  // Residuals are plotted on a logarithmic axis.
  EXPECT_TRUE(contains_line(lines, "set logscale y"));
  // The plot command uses every series with its name.
  std::string plot_command;
  for (const std::string & line : lines) {
    if (line.rfind("plot ", 0) == 0) plot_command = line;
  }
  for (const std::string name : {"series0", "series1", "series2", "first solve", "second solve", "no steps"}) {
    EXPECT_NE(plot_command.find(name), std::string::npos) << name;
  }
}

// The line for the console contains the number of the step, the residual and the reduction factor relative to the
// previous step.
TEST(ResidualOutputGeneratorTests, ConsoleStatementContainsStepResidualAndReduction) {
  GeneratorEnvironment environment;
  ResidualOutputGenerator generator("residuals", "Residuals", 0, 2, 5);
  generator.new_series("solve");
  generator.push_value(0.5);
  generator.push_value(0.125);
  generator.push_value(0.025);
  testing::internal::CaptureStdout();
  generator.write_residual_statement_to_console();
  const std::string output = testing::internal::GetCapturedStdout();
  std::smatch match;
  ASSERT_TRUE(std::regex_search(output, match, std::regex("Residual in step (\\d+) : (\\S+) Factor: (\\S+)"))) << output;
  EXPECT_EQ(std::stoi(match[1]), 3);
  EXPECT_NEAR(std::stod(match[2]), 0.025, 1e-7);
  EXPECT_NEAR(std::stod(match[3]), 0.2, 1e-6);
  // The rank of the process in the sweep of the level above.
  EXPECT_NE(output.find("[5]: "), std::string::npos) << output;
  EXPECT_EQ(std::count(output.begin(), output.end(), '\n'), 1);
}

TEST(ResidualOutputGeneratorTests, ConsoleStatementForTheFirstStep) {
  GeneratorEnvironment environment;
  ResidualOutputGenerator generator("residuals", "Residuals", 0, 2, -1);
  generator.new_series("solve");
  generator.push_value(0.75);
  testing::internal::CaptureStdout();
  generator.write_residual_statement_to_console();
  const std::string output = testing::internal::GetCapturedStdout();
  std::smatch match;
  ASSERT_TRUE(std::regex_search(output, match, std::regex("Residual in step 1 : (\\S+)"))) << output;
  EXPECT_NEAR(std::stod(match[1]), 0.75, 1e-6);
  // Without a parent sweep there is no rank prefix.
  EXPECT_EQ(output.find("["), std::string::npos) << output;
}

// The statement is indented by one tabulator per level below the highest sweeping level.
TEST(ResidualOutputGeneratorTests, ConsoleStatementIsIndentedByLevel) {
  GeneratorEnvironment environment;
  GlobalParams.Sweeping_Level = 3;
  for (unsigned int level = 1; level <= 3; level++) {
    ResidualOutputGenerator generator("residuals", "Residuals", 0, level, -1);
    generator.new_series("solve");
    generator.push_value(0.75);
    testing::internal::CaptureStdout();
    generator.write_residual_statement_to_console();
    const std::string output = testing::internal::GetCapturedStdout();
    EXPECT_EQ(output.find("Residual"), 3 - level) << output;
    EXPECT_EQ(output.substr(0, 3 - level), std::string(3 - level, '\t'));
  }
}

// Only the process with rank 0 in the sweep prints.
TEST(ResidualOutputGeneratorTests, OtherRanksDoNotPrint) {
  GeneratorEnvironment environment;
  ResidualOutputGenerator generator("residuals", "Residuals", 1, 2, -1);
  generator.new_series("solve");
  generator.push_value(0.75);
  generator.push_value(0.5);
  testing::internal::CaptureStdout();
  generator.write_residual_statement_to_console();
  EXPECT_EQ(testing::internal::GetCapturedStdout(), "");
}

TEST(ConvergenceOutputGeneratorTests, ScriptContainsAllValues) {
  GeneratorEnvironment environment;
  GlobalParams.MPI_Rank = 2;
  ConvergenceOutputGenerator generator;
  generator.set_title("Convergence of the error");
  const std::vector<double> cells = {4, 8, 16, 32};
  std::vector<double> numerical, theoretical;
  for (double n : cells) {
    numerical.push_back(1.3 / (n * n));
    theoretical.push_back(1.0 / (n * n));
    generator.push_values(n, numerical.back(), theoretical.back());
  }
  generator.write_gnuplot_file();

  const std::string script = environment.directory.file("Convergence2.plt");
  ASSERT_TRUE(std::filesystem::exists(script));
  const std::vector<std::string> lines = helpers_a::read_lines(script);
  const GnuplotArrays arrays(lines);
  EXPECT_EQ(arrays.declared_sizes.size(), 3u);
  expect_equal_up_to_output_precision(arrays.array("xvals"), cells);
  expect_equal_up_to_output_precision(arrays.array("yvalsnum"), numerical);
  expect_equal_up_to_output_precision(arrays.array("yvalstheo"), theoretical);
  EXPECT_TRUE(contains_line(lines, "set output \"" + environment.directory.file("Convergence2.png") + "\""));
  EXPECT_TRUE(contains_line(lines, "set title \"Convergence of the error\""));
  // Errors over mesh sizes are plotted with two logarithmic axes.
  EXPECT_TRUE(contains_line(lines, "set logscale x"));
  EXPECT_TRUE(contains_line(lines, "set logscale y"));
  std::string plot_command;
  for (const std::string & line : lines) {
    if (line.rfind("plot ", 0) == 0) plot_command = line;
  }
  for (const std::string name : {"xvals", "yvalsnum", "yvalstheo"}) {
    EXPECT_NE(plot_command.find(name), std::string::npos) << name;
  }
}

TEST(ConvergenceOutputGeneratorTests, ScriptWithoutValues) {
  GeneratorEnvironment environment;
  ConvergenceOutputGenerator generator;
  generator.write_gnuplot_file();
  const std::vector<std::string> lines = helpers_a::read_lines(environment.directory.file("Convergence0.plt"));
  const GnuplotArrays arrays(lines);
  EXPECT_TRUE(arrays.array("xvals").empty());
  EXPECT_TRUE(arrays.array("yvalsnum").empty());
  EXPECT_TRUE(arrays.array("yvalstheo").empty());
}
