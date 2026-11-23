// Tests for TimerManager and OutputManager. Both are tested with local objects, not with the global instances. All
// files are written to temporary directories.
#include <algorithm>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <filesystem>
#include <regex>
#include "../Code/GlobalObjects/GlobalObjects.h"
#include "../Code/GlobalObjects/OutputManager.h"
#include "../Code/GlobalObjects/TimerManager.h"
#include "./HelpersA.h"

namespace {

  using helpers_a::ScopedEnvironmentVariable;
  using helpers_a::ScopedGeometry;
  using helpers_a::ScopedGlobalParams;
  using helpers_a::ScopedOutputFolder;
  using helpers_a::TemporaryDirectory;

  // The timers use the communicators of the levels. This provides them for a single process and restores the global
  // object afterwards.
  class ScopedCommunicators {
   public:
    ScopedCommunicators() : saved(GlobalMPI.communicators_by_level) {
      GlobalMPI.communicators_by_level = {MPI_COMM_SELF, MPI_COMM_WORLD, MPI_COMM_WORLD, MPI_COMM_WORLD};
    }
    ~ScopedCommunicators() { GlobalMPI.communicators_by_level = saved; }

   private:
    std::vector<MPI_Comm> saved;
  };

  struct TimerEnvironment {
    TimerEnvironment() : directory("timer"), folder(directory.path()) {}
    ScopedGlobalParams params;
    ScopedCommunicators communicators;
    TemporaryDirectory directory;
    ScopedOutputFolder folder;
  };

  bool contains_line(const std::vector<std::string> & lines, const std::string & line) {
    return std::find(lines.begin(), lines.end(), line) != lines.end();
  }

  // initialize() of a local output manager below the given base folder. The messages are discarded.
  void initialize_quietly(OutputManager & manager) {
    testing::internal::CaptureStdout();
    manager.initialize();
    testing::internal::GetCapturedStdout();
  }

}

// There is one timer for the local problems (level 0) and one for every direction with more than one block.
TEST(TimerManagerTests, OneLevelPerDirectionWithSeveralBlocks) {
  const unsigned int blocks[5][3] = {{1, 1, 1}, {1, 1, 4}, {1, 3, 4}, {2, 3, 4}, {2, 1, 1}};
  const unsigned int expected_levels[5] = {1, 2, 3, 4, 2};
  for (unsigned int i = 0; i < 5; i++) {
    TimerEnvironment environment;
    GlobalParams.Blocks_in_x_direction = blocks[i][0];
    GlobalParams.Blocks_in_y_direction = blocks[i][1];
    GlobalParams.Blocks_in_z_direction = blocks[i][2];
    TimerManager timers;
    timers.initialize();
    EXPECT_EQ(timers.level_count, expected_levels[i]);
    EXPECT_EQ(timers.timer_outputs.size(), expected_levels[i]);
    ASSERT_EQ(timers.filenames.size(), expected_levels[i]);
    ASSERT_EQ(timers.filestreams.size(), expected_levels[i]);
    for (unsigned int level = 0; level < expected_levels[i]; level++) {
      EXPECT_EQ(timers.filenames[level], environment.directory.file("timer_output" + std::to_string(level) + ".dat"));
      EXPECT_TRUE(std::filesystem::exists(timers.filenames[level]));
      EXPECT_TRUE(timers.filestreams[level]->is_open());
    }
    timers.finalize();
  }
}

// Every level reports its own sections with the number of times they were entered.
TEST(TimerManagerTests, OutputContainsTheSectionsOfEveryLevelWithTheirCallCounts) {
  TimerEnvironment environment;
  GlobalParams.Blocks_in_z_direction = 2;
  TimerManager timers;
  timers.initialize();
  ASSERT_EQ(timers.level_count, 2u);
  for (unsigned int i = 0; i < 3; i++) {
    timers.switch_context("assemble", 0);
    timers.leave_context(0);
  }
  timers.switch_context("factorize", 0);
  timers.leave_context(0);
  timers.switch_context("sweep", 1);
  timers.leave_context(1);
  timers.switch_context("sweep", 1);
  timers.leave_context(1);
  timers.write_output();
  const std::string level_0 = helpers_a::read_file(environment.directory.file("timer_output0.dat"));
  const std::string level_1 = helpers_a::read_file(environment.directory.file("timer_output1.dat"));
  timers.finalize();

  // Rows of the summary table of deal.II: | name | number of calls | ...
  std::smatch match;
  ASSERT_TRUE(std::regex_search(level_0, match, std::regex("\\| assemble\\s*\\|\\s*(\\d+) \\|"))) << level_0;
  EXPECT_EQ(std::stoi(match[1]), 3);
  ASSERT_TRUE(std::regex_search(level_0, match, std::regex("\\| factorize\\s*\\|\\s*(\\d+) \\|"))) << level_0;
  EXPECT_EQ(std::stoi(match[1]), 1);
  EXPECT_EQ(level_0.find("sweep"), std::string::npos);
  ASSERT_TRUE(std::regex_search(level_1, match, std::regex("\\| sweep\\s*\\|\\s*(\\d+) \\|"))) << level_1;
  EXPECT_EQ(std::stoi(match[1]), 2);
  EXPECT_EQ(level_1.find("assemble"), std::string::npos);
  EXPECT_EQ(level_1.find("factorize"), std::string::npos);
}

TEST(TimerManagerTests, FinalizeReleasesTimersAndFiles) {
  TimerEnvironment environment;
  GlobalParams.Blocks_in_z_direction = 2;
  TimerManager timers;
  timers.initialize();
  timers.finalize();
  EXPECT_TRUE(timers.timer_outputs.empty());
  EXPECT_TRUE(timers.filestreams.empty());
  EXPECT_TRUE(timers.filenames.empty());
  // The files stay in the output folder.
  EXPECT_TRUE(std::filesystem::exists(environment.directory.file("timer_output0.dat")));
  EXPECT_TRUE(std::filesystem::exists(environment.directory.file("timer_output1.dat")));
}

TEST(OutputManagerTests, FilenamesAreInsideTheOutputFolder) {
  OutputManager manager;
  manager.output_folder_path = "/base/run7";
  EXPECT_EQ(manager.get_full_filename("solution.vtk"), "/base/run7/solution.vtk");
  EXPECT_EQ(manager.get_numbered_filename("solution", 12, "vtu"), "/base/run7/solution12.vtu");
  EXPECT_EQ(manager.get_numbered_filename("timer_output", 0, "dat"), "/base/run7/timer_output0.dat");
}

// WAVEGUIDE_OUTPUT_DIR is the base folder of the runs. Every run gets the first folder runN that does not exist yet.
TEST(OutputManagerTests, InitializeCreatesTheNextRunFolder) {
  ScopedGlobalParams params;
  ScopedGeometry geometry;
  TemporaryDirectory directory("output_manager");
  const std::string base = directory.file("outputs");
  ScopedEnvironmentVariable output_dir("WAVEGUIDE_OUTPUT_DIR", base.c_str());
  ScopedEnvironmentVariable work("WORK", "/this/folder/does/not/exist");
  {
    OutputManager first;
    initialize_quietly(first);
    EXPECT_EQ(first.output_folder_path, base + "/run0");
    EXPECT_TRUE(std::filesystem::is_directory(base + "/run0"));
    EXPECT_TRUE(std::filesystem::exists(base + "/run0/run_description.txt"));
    EXPECT_TRUE(first.log_stream.is_open());
    EXPECT_TRUE(std::filesystem::exists(base + "/run0/main0.log"));
  }
  {
    OutputManager second;
    initialize_quietly(second);
    EXPECT_EQ(second.output_folder_path, base + "/run1");
    EXPECT_TRUE(std::filesystem::is_directory(base + "/run1"));
  }
  // A gap in the numbering is filled.
  std::filesystem::create_directories(base + "/run3");
  {
    OutputManager third;
    initialize_quietly(third);
    EXPECT_EQ(third.output_folder_path, base + "/run2");
  }
  {
    OutputManager fourth;
    initialize_quietly(fourth);
    EXPECT_EQ(fourth.output_folder_path, base + "/run4");
  }
}

// Without WAVEGUIDE_OUTPUT_DIR the folder Solutions in $WORK is used (HPC workspaces).
TEST(OutputManagerTests, WorkFolderIsUsedWithoutOutputDirectory) {
  ScopedGlobalParams params;
  ScopedGeometry geometry;
  TemporaryDirectory directory("output_manager_work");
  ScopedEnvironmentVariable output_dir("WAVEGUIDE_OUTPUT_DIR", nullptr);
  ScopedEnvironmentVariable work("WORK", directory.path().c_str());
  OutputManager manager;
  initialize_quietly(manager);
  EXPECT_EQ(manager.output_folder_path, directory.path() + "/Solutions/run0");
  EXPECT_TRUE(std::filesystem::is_directory(directory.path() + "/Solutions/run0"));
}

TEST(OutputManagerTests, LogLinesAreWrittenToTheLogFileOfTheProcess) {
  ScopedGlobalParams params;
  ScopedGeometry geometry;
  TemporaryDirectory directory("output_manager_log");
  ScopedEnvironmentVariable output_dir("WAVEGUIDE_OUTPUT_DIR", directory.path().c_str());
  std::string log_file;
  {
    OutputManager manager;
    initialize_quietly(manager);
    log_file = manager.get_full_filename("main0.log");
    manager.write_log_ling("first line");
    manager.write_log_ling("second line");
  }
  EXPECT_EQ(helpers_a::read_lines(log_file), (std::vector<std::string>{"first line", "second line"}));
}

TEST(OutputManagerTests, RunDescriptionContainsTheSetup) {
  ScopedGlobalParams params;
  ScopedGeometry geometry;
  TemporaryDirectory directory("output_manager_description");
  GlobalParams.NumberProcesses = 24;
  GlobalParams.Sweeping_Level = 3;
  GlobalParams.BoundaryCondition = BoundaryConditionType::PML;
  GlobalParams.use_tapered_input_signal = true;
  GlobalParams.prescribe_0_on_input_side = true;
  GlobalParams.Use_Predefined_Shape = true;
  GlobalParams.Number_of_Predefined_Shape = 17;
  GlobalParams.Blocks_in_x_direction = 2;
  GlobalParams.Blocks_in_y_direction = 3;
  GlobalParams.Blocks_in_z_direction = 4;
  GlobalParams.Cells_in_x = 5;
  GlobalParams.Cells_in_y = 7;
  GlobalParams.Cells_in_z = 11;
  GlobalParams.PML_N_Layers = 9;
  GlobalParams.use_relative_convergence_criterion = true;
  GlobalParams.relative_convergence_criterion = 0.125;
  Geometry.global_x_range = std::make_pair(-1.5, 1.5);
  Geometry.global_y_range = std::make_pair(-2.5, 2.5);
  Geometry.global_z_range = std::make_pair(0.5, 8.5);
  OutputManager manager;
  manager.output_folder_path = directory.path();
  manager.write_run_description("0123abcd");
  const std::vector<std::string> lines = helpers_a::read_lines(directory.file("run_description.txt"));
  EXPECT_TRUE(contains_line(lines, "Number of processes: \t24"));
  EXPECT_TRUE(contains_line(lines, "Sweeping level: 3"));
  EXPECT_TRUE(contains_line(lines, "Truncation Method: PML"));
  EXPECT_TRUE(contains_line(lines, "Signal input method: Taper"));
  EXPECT_TRUE(contains_line(lines, "Set 0 on input interface: true"));
  EXPECT_TRUE(contains_line(lines, "Use predefined shape: true"));
  EXPECT_TRUE(contains_line(lines, "Predefined Shape Number: 17"));
  EXPECT_TRUE(contains_line(lines, "Block Counts: [2x3x4]"));
  // The global cell counts are the products of the blocks and the cells per block.
  EXPECT_TRUE(contains_line(lines, "Global cell count x: 10"));
  EXPECT_TRUE(contains_line(lines, "Global cell count y: 21"));
  EXPECT_TRUE(contains_line(lines, "Global cell count z: 44"));
  EXPECT_TRUE(contains_line(lines, "Number of PML cell layers: 9"));
  EXPECT_TRUE(contains_line(lines, "Use relative convergence limiter: true"));
  EXPECT_TRUE(contains_line(lines, "Relative convergence limit: 0.125"));
  EXPECT_TRUE(contains_line(lines, "Global x range: -1.5 to 1.5"));
  EXPECT_TRUE(contains_line(lines, "Global y range: -2.5 to 2.5"));
  EXPECT_TRUE(contains_line(lines, "Global z range: 0.5 to 8.5"));
  EXPECT_TRUE(contains_line(lines, "Git commit hash: 0123abcd"));
}

TEST(OutputManagerTests, RunDescriptionOmitsValuesThatAreNotUsed) {
  ScopedGlobalParams params;
  ScopedGeometry geometry;
  TemporaryDirectory directory("output_manager_description_2");
  GlobalParams.BoundaryCondition = BoundaryConditionType::HSIE;
  GlobalParams.use_tapered_input_signal = false;
  GlobalParams.Use_Predefined_Shape = false;
  GlobalParams.use_relative_convergence_criterion = false;
  OutputManager manager;
  manager.output_folder_path = directory.path();
  manager.write_run_description("");
  const std::vector<std::string> lines = helpers_a::read_lines(directory.file("run_description.txt"));
  const std::string content = helpers_a::read_file(directory.file("run_description.txt"));
  EXPECT_TRUE(contains_line(lines, "Truncation Method: HSIE"));
  EXPECT_TRUE(contains_line(lines, "Signal input method: Dirichlet"));
  EXPECT_TRUE(contains_line(lines, "Use predefined shape: false"));
  EXPECT_TRUE(contains_line(lines, "Use relative convergence limiter: false"));
  EXPECT_EQ(content.find("Predefined Shape Number"), std::string::npos);
  EXPECT_EQ(content.find("Relative convergence limit:"), std::string::npos);
}
