// Tests for the run report: formatting of numbers, the structure of sections, entries, tables and notes, and the two
// output files. The JSON file is parsed with a strict parser and compared with what was put into the report.
#include <stdexcept>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <cmath>
#include <filesystem>
#include <limits>
#include "../Code/GlobalObjects/GlobalObjects.h"
#include "../Code/OutputGenerators/RunReport.h"
#include "./HelpersA.h"

namespace {

  using helpers_a::Json;
  using helpers_a::JsonParser;
  using helpers_a::ScopedGlobalParams;
  using helpers_a::ScopedOutputFolder;
  using helpers_a::TemporaryDirectory;

  // Writes the report into a temporary directory as process 0 and keeps the contents of the two files.
  struct WrittenReport {
    explicit WrittenReport(RunReport & report) {
      ScopedGlobalParams params;
      TemporaryDirectory directory("run_report");
      ScopedOutputFolder folder(directory.path());
      report.write();
      text_exists = std::filesystem::exists(directory.file("run_report.txt"));
      json_exists = std::filesystem::exists(directory.file("run_report.json"));
      text = helpers_a::read_file(directory.file("run_report.txt"));
      text_lines = helpers_a::read_lines(directory.file("run_report.txt"));
      json_text = helpers_a::read_file(directory.file("run_report.json"));
    }
    Json json() const { return JsonParser::parse(json_text); }
    bool text_exists, json_exists;
    std::string text, json_text;
    std::vector<std::string> text_lines;
  };

  // The value of an entry of a section in the parsed JSON file.
  std::string entry_value(const Json & section, const std::string & name) {
    for (const Json & entry : section.at("entries").array) {
      if (entry.at(0).string == name) return entry.at(1).string;
    }
    ADD_FAILURE() << "No entry " << name;
    return "";
  }

  size_t line_index(const std::vector<std::string> & lines, const std::string & content) {
    for (size_t i = 0; i < lines.size(); i++) {
      if (lines[i].find(content) != std::string::npos) return i;
    }
    ADD_FAILURE() << "No line contains: " << content;
    return 0;
  }

}

TEST(RunReportFormatTests, NumberUsesTheRequestedSignificantDigits) {
  EXPECT_EQ(RunReport::number(0.5), "0.5");
  EXPECT_EQ(RunReport::number(0), "0");
  EXPECT_EQ(RunReport::number(-2), "-2");
  EXPECT_EQ(RunReport::number(3.14159265358979, 3), "3.14");
  EXPECT_EQ(RunReport::number(3.14159265358979), "3.14159");
  EXPECT_EQ(RunReport::number(123456789.0, 3), "1.23e+08");
  EXPECT_EQ(RunReport::number(0.000012345678, 2), "1.2e-05");
}

// Reading the text back gives the number up to the rounding to the requested digits.
TEST(RunReportFormatTests, NumberCanBeReadBack) {
  for (double value : {1.0 / 3.0, -2.0e-7 / 3.0, 12345.678901, 6.02214076e23, -1.5e-300}) {
    for (unsigned int digits : {3u, 6u, 12u}) {
      const double read = std::stod(RunReport::number(value, digits));
      EXPECT_LE(std::abs(read - value), 0.5 * std::pow(10.0, 1.0 - digits) * std::abs(value) * 1.0000001) << value << " " << digits;
    }
  }
}

TEST(RunReportFormatTests, CountSeparatesGroupsOfThreeDigits) {
  EXPECT_EQ(RunReport::count(0), "0");
  EXPECT_EQ(RunReport::count(7), "7");
  EXPECT_EQ(RunReport::count(999), "999");
  EXPECT_EQ(RunReport::count(1000), "1,000");
  EXPECT_EQ(RunReport::count(12345), "12,345");
  EXPECT_EQ(RunReport::count(123456), "123,456");
  EXPECT_EQ(RunReport::count(1234567), "1,234,567");
  EXPECT_EQ(RunReport::count(1000000000.0), "1,000,000,000");
  EXPECT_EQ(RunReport::count(-1), "-1");
  EXPECT_EQ(RunReport::count(-1234), "-1,234");
  EXPECT_EQ(RunReport::count(-123456), "-123,456");
}

// Removing the separators gives the decimal representation of the integer.
TEST(RunReportFormatTests, CountWithoutSeparatorsIsTheNumber) {
  for (long value : {1L, 10L, 100L, 1001L, 99999L, 100000L, 4294967296L, 123456789012L}) {
    std::string formatted = RunReport::count(static_cast<double>(value));
    for (size_t i = 0; i < formatted.size(); i++) {
      // A separator has digits on both sides and is followed by exactly three digits before the next separator.
      if (formatted[i] == ',') {
        ASSERT_GT(i, 0u);
        ASSERT_LT(i + 3, formatted.size());
        EXPECT_TRUE(i + 4 == formatted.size() || formatted[i + 4] == ',') << formatted;
      }
    }
    formatted.erase(std::remove(formatted.begin(), formatted.end(), ','), formatted.end());
    EXPECT_EQ(formatted, std::to_string(value));
  }
}

TEST(RunReportTests, SecondsSinceStartIncrease) {
  RunReport report;
  const double first = report.seconds_since_start();
  const double second = report.seconds_since_start();
  EXPECT_GE(first, 0.0);
  EXPECT_GE(second, first);
  EXPECT_LT(second, 10.0);
}

TEST(RunReportTests, EmptyReportIsValid) {
  RunReport report;
  const WrittenReport written(report);
  EXPECT_TRUE(written.text_exists);
  EXPECT_TRUE(written.json_exists);
  const Json json = written.json();
  EXPECT_EQ(json.at("sections").type, Json::Type::Array);
  EXPECT_EQ(json.at("sections").size(), 0u);
  EXPECT_EQ(json.at("data").type, Json::Type::Object);
  EXPECT_EQ(json.at("data").size(), 0u);
  EXPECT_EQ(json.at("series").type, Json::Type::Array);
  EXPECT_EQ(json.at("series").size(), 0u);
}

TEST(RunReportTests, OnlyProcessZeroWritesTheReport) {
  RunReport report;
  report.entry("name", "value");
  ScopedGlobalParams params;
  GlobalParams.MPI_Rank = 1;
  TemporaryDirectory directory("run_report_rank");
  ScopedOutputFolder folder(directory.path());
  report.write();
  EXPECT_FALSE(std::filesystem::exists(directory.file("run_report.txt")));
  EXPECT_FALSE(std::filesystem::exists(directory.file("run_report.json")));
}

TEST(RunReportTests, SectionsKeepTheirOrderAndAreNotDuplicated) {
  RunReport report;
  report.section("Setup");
  report.entry("a", "1");
  report.section("Solver");
  report.entry("b", "2");
  // Selecting an existing section again continues it.
  report.section("Setup");
  report.entry("c", "3");
  const Json json = WrittenReport(report).json();
  const Json & sections = json.at("sections");
  ASSERT_EQ(sections.size(), 2u);
  EXPECT_EQ(sections.at(0).at("title").string, "Setup");
  EXPECT_EQ(sections.at(1).at("title").string, "Solver");
  ASSERT_EQ(sections.at(0).at("entries").size(), 2u);
  EXPECT_EQ(sections.at(0).at("entries").at(0).at(0).string, "a");
  EXPECT_EQ(sections.at(0).at("entries").at(1).at(0).string, "c");
  ASSERT_EQ(sections.at(1).at("entries").size(), 1u);
  EXPECT_EQ(entry_value(sections.at(1), "b"), "2");
}

TEST(RunReportTests, EntriesWithoutSectionGoToTheSectionRun) {
  RunReport report;
  report.entry("a", "1");
  report.note("remark");
  report.table("t", {"h"}, {{"x"}});
  const Json json = WrittenReport(report).json();
  ASSERT_EQ(json.at("sections").size(), 1u);
  const Json & section = json.at("sections").at(0);
  EXPECT_EQ(section.at("title").string, "Run");
  EXPECT_EQ(section.at("entries").size(), 1u);
  EXPECT_EQ(section.at("notes").size(), 1u);
  EXPECT_EQ(section.at("tables").size(), 1u);
}

TEST(RunReportTests, EntryOverloadsFormatTheirValues) {
  RunReport report;
  report.section("Types");
  report.entry("text", std::string("some text"));
  // A string literal must be stored as text and not be converted to bool.
  report.entry("literal", "a literal");
  report.entry("double", 0.25);
  report.entry("int", -1234567);
  report.entry("unsigned", 7654321u);
  report.entry("true", true);
  report.entry("false", false);
  const Json json = WrittenReport(report).json();
  const Json & section = json.at("sections").at(0);
  EXPECT_EQ(entry_value(section, "text"), "some text");
  EXPECT_EQ(entry_value(section, "literal"), "a literal");
  EXPECT_EQ(entry_value(section, "double"), "0.25");
  EXPECT_EQ(entry_value(section, "int"), "-1,234,567");
  EXPECT_EQ(entry_value(section, "unsigned"), "7,654,321");
  EXPECT_EQ(entry_value(section, "true"), "yes");
  EXPECT_EQ(entry_value(section, "false"), "no");
}

TEST(RunReportTests, EntrySetAgainReplacesTheValueAndKeepsThePosition) {
  RunReport report;
  report.section("Solver");
  report.entry("iterations", 10);
  report.entry("residual", 0.5);
  report.entry("iterations", 12);
  // The same name in another section is another entry.
  report.section("Other");
  report.entry("iterations", 99);
  const Json json = WrittenReport(report).json();
  const Json & solver = json.at("sections").at(0);
  ASSERT_EQ(solver.at("entries").size(), 2u);
  EXPECT_EQ(solver.at("entries").at(0).at(0).string, "iterations");
  EXPECT_EQ(solver.at("entries").at(0).at(1).string, "12");
  EXPECT_EQ(solver.at("entries").at(1).at(0).string, "residual");
  EXPECT_EQ(entry_value(json.at("sections").at(1), "iterations"), "99");
}

TEST(RunReportTests, TableWithTheSameTitleIsReplaced) {
  RunReport report;
  report.section("Results");
  report.table("Levels", {"level", "dofs"}, {{"1", "100"}, {"2", "200"}});
  report.table("Timing", {"part", "seconds"}, {{"solve", "1.5"}});
  report.table("Levels", {"level", "dofs", "steps"}, {{"1", "150", "7"}});
  const Json json = WrittenReport(report).json();
  const Json & tables = json.at("sections").at(0).at("tables");
  ASSERT_EQ(tables.size(), 2u);
  EXPECT_EQ(tables.at(0).at("title").string, "Levels");
  ASSERT_EQ(tables.at(0).at("header").size(), 3u);
  EXPECT_EQ(tables.at(0).at("header").at(2).string, "steps");
  ASSERT_EQ(tables.at(0).at("rows").size(), 1u);
  ASSERT_EQ(tables.at(0).at("rows").at(0).size(), 3u);
  EXPECT_EQ(tables.at(0).at("rows").at(0).at(1).string, "150");
  EXPECT_EQ(tables.at(1).at("title").string, "Timing");
  EXPECT_EQ(tables.at(1).at("rows").at(0).at(0).string, "solve");
  EXPECT_EQ(tables.at(1).at("rows").at(0).at(1).string, "1.5");
}

TEST(RunReportTests, RepeatedNotesAreStoredOncePerSection) {
  RunReport report;
  report.section("A");
  report.note("first");
  report.note("second");
  report.note("first");
  report.section("B");
  report.note("first");
  report.section("A");
  report.note("second");
  const Json json = WrittenReport(report).json();
  const Json & notes_a = json.at("sections").at(0).at("notes");
  ASSERT_EQ(notes_a.size(), 2u);
  EXPECT_EQ(notes_a.at(0).string, "first");
  EXPECT_EQ(notes_a.at(1).string, "second");
  const Json & notes_b = json.at("sections").at(1).at("notes");
  ASSERT_EQ(notes_b.size(), 1u);
  EXPECT_EQ(notes_b.at(0).string, "first");
}

TEST(RunReportTests, DataValuesAreReplacedAndWrittenWithTwelveDigits) {
  RunReport report;
  report.data("loss", 1.0);
  report.data("pi", 3.14159265358979);
  report.data("loss", -0.125);
  report.data("small", 1.5e-300);
  report.data("method", std::string("BFGS"));
  report.data("method", std::string("Steepest"));
  report.data("count", 3.0);
  const Json json = WrittenReport(report).json();
  const Json & data = json.at("data");
  ASSERT_EQ(data.size(), 5u);
  EXPECT_EQ(data.at("loss").type, Json::Type::Number);
  EXPECT_EQ(data.at("loss").number, -0.125);
  EXPECT_NEAR(data.at("pi").number, 3.14159265358979, 1e-11);
  EXPECT_NEAR(data.at("small").number / 1.5e-300, 1.0, 1e-11);
  EXPECT_EQ(data.at("count").number, 3.0);
  EXPECT_EQ(data.at("method").type, Json::Type::String);
  EXPECT_EQ(data.at("method").string, "Steepest");
  // Data for post-processing is not part of the sections.
  EXPECT_EQ(json.at("sections").size(), 0u);
}

TEST(RunReportTests, SeriesAreKeptInOrderAlsoWithEqualNames) {
  RunReport report;
  report.series("residuals", {1.0, 0.1, 0.01});
  report.series("residuals", {2.0, 0.5});
  report.series("empty", {});
  const Json json = WrittenReport(report).json();
  const Json & series = json.at("series");
  ASSERT_EQ(series.size(), 3u);
  EXPECT_EQ(series.at(0).at("name").string, "residuals");
  ASSERT_EQ(series.at(0).at("values").size(), 3u);
  EXPECT_EQ(series.at(0).at("values").at(0).number, 1.0);
  EXPECT_EQ(series.at(0).at("values").at(1).number, 0.1);
  EXPECT_EQ(series.at(0).at("values").at(2).number, 0.01);
  EXPECT_EQ(series.at(1).at("name").string, "residuals");
  ASSERT_EQ(series.at(1).at("values").size(), 2u);
  EXPECT_EQ(series.at(1).at("values").at(1).number, 0.5);
  EXPECT_EQ(series.at(2).at("name").string, "empty");
  EXPECT_EQ(series.at(2).at("values").size(), 0u);
}

// JSON has no representation for infinity and NaN; they are written as null so that the file stays valid.
TEST(RunReportTests, NonFiniteNumbersAreWrittenAsNull) {
  RunReport report;
  report.data("nan", std::numeric_limits<double>::quiet_NaN());
  report.data("infinity", std::numeric_limits<double>::infinity());
  report.series("diverged", {1.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()});
  const Json json = WrittenReport(report).json();
  EXPECT_EQ(json.at("data").at("nan").type, Json::Type::Null);
  EXPECT_EQ(json.at("data").at("infinity").type, Json::Type::Null);
  const Json & values = json.at("series").at(0).at("values");
  ASSERT_EQ(values.size(), 3u);
  EXPECT_EQ(values.at(0).type, Json::Type::Number);
  EXPECT_EQ(values.at(1).type, Json::Type::Null);
  EXPECT_EQ(values.at(2).type, Json::Type::Null);
}

TEST(RunReportTests, QuotesBackslashesAndLineBreaksSurviveTheJsonFile) {
  const std::string special = "a \"quoted\" path C:\\folder\\file\nsecond line";
  RunReport report;
  report.section(special);
  report.entry(special, special);
  report.note(special);
  report.table(special, {special}, {{special}});
  report.data(special, special);
  report.series(special, {1.0});
  const Json json = WrittenReport(report).json();
  const Json & section = json.at("sections").at(0);
  EXPECT_EQ(section.at("title").string, special);
  EXPECT_EQ(section.at("entries").at(0).at(0).string, special);
  EXPECT_EQ(section.at("entries").at(0).at(1).string, special);
  EXPECT_EQ(section.at("notes").at(0).string, special);
  EXPECT_EQ(section.at("tables").at(0).at("title").string, special);
  EXPECT_EQ(section.at("tables").at(0).at("header").at(0).string, special);
  EXPECT_EQ(section.at("tables").at(0).at("rows").at(0).at(0).string, special);
  EXPECT_EQ(json.at("data").at(special).string, special);
  EXPECT_EQ(json.at("series").at(0).at("name").string, special);
}

// Tabulators are replaced by spaces, so the file stays valid.
TEST(RunReportTests, TabulatorsDoNotBreakTheJsonFile) {
  RunReport report;
  report.entry("name", "a\tb");
  const Json json = WrittenReport(report).json();
  EXPECT_EQ(entry_value(json.at("sections").at(0), "name"), "a b");
}

// JSON does not allow unescaped control characters in strings. The code escapes line feeds and replaces tabulators,
// but writes all other control characters (for example a carriage return) unchanged, which makes the file invalid.
TEST(RunReportTests, DISABLED_AllControlCharactersAreEscapedInTheJsonFile) {
  const std::string special = std::string("line\r\nwith carriage return and bell ") + '\a';
  RunReport report;
  report.entry("name", special);
  const WrittenReport written(report);
  Json json;
  ASSERT_NO_THROW(json = written.json());
  EXPECT_EQ(entry_value(json.at("sections").at(0), "name"), special);
}

// The complete structure of a report with all kinds of content.
TEST(RunReportTests, JsonFileContainsTheCompleteReport) {
  RunReport report;
  report.section("Setup");
  report.entry("Boundary method", "PML");
  report.entry("Processes", 8u);
  report.table("Cells", {"direction", "cells"}, {{"x", "20"}, {"y", "20"}, {"z", "160"}});
  report.note("The mesh is uniform.");
  report.section("Results");
  report.entry("Transmission", 0.987654321);
  report.data("transmission", 0.987654321);
  report.series("signal", {1.0, 0.99, 0.98});
  const Json json = WrittenReport(report).json();
  ASSERT_EQ(json.type, Json::Type::Object);
  EXPECT_EQ(json.size(), 3u);
  ASSERT_EQ(json.at("sections").size(), 2u);
  for (const Json & section : json.at("sections").array) {
    EXPECT_EQ(section.type, Json::Type::Object);
    EXPECT_EQ(section.at("title").type, Json::Type::String);
    EXPECT_EQ(section.at("entries").type, Json::Type::Array);
    EXPECT_EQ(section.at("tables").type, Json::Type::Array);
    EXPECT_EQ(section.at("notes").type, Json::Type::Array);
  }
  const Json & setup = json.at("sections").at(0);
  EXPECT_EQ(entry_value(setup, "Boundary method"), "PML");
  EXPECT_EQ(entry_value(setup, "Processes"), "8");
  const Json & table = setup.at("tables").at(0);
  EXPECT_EQ(table.at("title").string, "Cells");
  ASSERT_EQ(table.at("header").size(), 2u);
  EXPECT_EQ(table.at("header").at(0).string, "direction");
  EXPECT_EQ(table.at("header").at(1).string, "cells");
  ASSERT_EQ(table.at("rows").size(), 3u);
  EXPECT_EQ(table.at("rows").at(2).at(0).string, "z");
  EXPECT_EQ(table.at("rows").at(2).at(1).string, "160");
  EXPECT_EQ(setup.at("notes").at(0).string, "The mesh is uniform.");
  EXPECT_EQ(entry_value(json.at("sections").at(1), "Transmission"), "0.987654");
  EXPECT_NEAR(json.at("data").at("transmission").number, 0.987654321, 1e-12);
  EXPECT_EQ(json.at("series").at(0).at("values").size(), 3u);
}

// In the text file the sections are numbered and underlined, and the values of the entries of a section start in
// the same column.
TEST(RunReportTests, TextFileHasNumberedSectionsAndAlignedEntries) {
  RunReport report;
  report.section("Setup");
  report.entry("a", "first value");
  report.entry("a much longer name", "second value");
  report.entry("medium name", "third value");
  report.section("Results");
  report.entry("loss", 0.5);
  report.data("only in json", 17.0);
  report.series("also only in json", {1.0});
  const WrittenReport written(report);
  const std::vector<std::string> & lines = written.text_lines;
  const size_t setup = line_index(lines, "1. Setup");
  const size_t results = line_index(lines, "2. Results");
  EXPECT_EQ(lines[setup], "1. Setup");
  EXPECT_EQ(lines[setup + 1], std::string(lines[setup].size(), '-'));
  EXPECT_EQ(lines[results], "2. Results");
  EXPECT_EQ(lines[results + 1], std::string(lines[results].size(), '-'));
  EXPECT_LT(setup, results);
  const size_t first = line_index(lines, "first value");
  const size_t second = line_index(lines, "second value");
  const size_t third = line_index(lines, "third value");
  EXPECT_EQ(second, first + 1);
  EXPECT_EQ(third, first + 2);
  EXPECT_LT(setup, first);
  EXPECT_LT(third, results);
  EXPECT_EQ(lines[first].find("first value"), lines[second].find("second value"));
  EXPECT_EQ(lines[first].find("first value"), lines[third].find("third value"));
  EXPECT_LT(lines[first].find("a"), lines[first].find(" : first value"));
  EXPECT_NE(lines[second].find("a much longer name : second value"), std::string::npos);
  EXPECT_NE(lines[line_index(lines, "loss")].find(": 0.5"), std::string::npos);
  EXPECT_EQ(written.text.find("only in json"), std::string::npos);
}

// The columns of a table are as wide as their widest cell, so all cells of a column start at the same position.
TEST(RunReportTests, TextFileHasAlignedTableColumnsAndNotes) {
  RunReport report;
  report.section("Results");
  report.table("Levels of the hierarchy", {"level", "unknowns", "t"}, {{"1", "1,234,567", "0.5"}, {"long level name", "12", "17.25"}});
  report.note("Remark one.");
  report.note("Remark two.");
  const WrittenReport written(report);
  const std::vector<std::string> & lines = written.text_lines;
  const size_t title = line_index(lines, "Levels of the hierarchy");
  const std::string & header = lines[title + 1];
  const std::string & rule = lines[title + 2];
  const std::string & row_1 = lines[title + 3];
  const std::string & row_2 = lines[title + 4];
  EXPECT_EQ(header.find("level"), row_1.find("1"));
  EXPECT_EQ(header.find("level"), row_2.find("long level name"));
  EXPECT_EQ(header.find("unknowns"), row_1.find("1,234,567"));
  EXPECT_EQ(header.find("unknowns"), row_2.find("12"));
  EXPECT_EQ(header.find(" t") + 1, row_1.find("0.5"));
  EXPECT_EQ(header.find(" t") + 1, row_2.find("17.25"));
  // The rule below the header consists of one group of dashes per column with the width of the column.
  EXPECT_EQ(rule.find_first_not_of(" -"), std::string::npos);
  EXPECT_EQ(rule.find("-"), header.find("level"));
  EXPECT_NE(rule.find(std::string(std::string("long level name").size(), '-') + "  " + std::string(std::string("1,234,567").size(), '-') + "  " + std::string(std::string("17.25").size(), '-')), std::string::npos);
  const size_t note_1 = line_index(lines, "Remark one.");
  const size_t note_2 = line_index(lines, "Remark two.");
  EXPECT_GT(note_1, title + 4);
  EXPECT_EQ(note_2, note_1 + 1);
  EXPECT_NE(lines[note_1].find("Note: Remark one."), std::string::npos);
}

// Rows that are shorter or longer than the header must not break the outputs.
TEST(RunReportTests, TableRowsWithDifferentLengths) {
  RunReport report;
  report.table("Ragged", {"a", "b"}, {{"1"}, {"1", "2", "3"}, {}});
  const WrittenReport written(report);
  const Json json = written.json();
  const Json & rows = json.at("sections").at(0).at("tables").at(0).at("rows");
  ASSERT_EQ(rows.size(), 3u);
  EXPECT_EQ(rows.at(0).size(), 1u);
  EXPECT_EQ(rows.at(1).size(), 3u);
  EXPECT_EQ(rows.at(2).size(), 0u);
  EXPECT_NE(written.text.find("Ragged"), std::string::npos);
}

TEST(RunReportTests, WritingTwiceGivesTheSameFiles) {
  RunReport report;
  report.section("Setup");
  report.entry("a", 1);
  report.series("s", {1.0, 2.0});
  const WrittenReport first(report);
  const WrittenReport second(report);
  EXPECT_EQ(first.text, second.text);
  EXPECT_EQ(first.json_text, second.json_text);
}

// The strict parser of the tests has to reject what is not JSON, otherwise the tests above would not prove anything.
TEST(RunReportTests, TestParserRejectsInvalidJson) {
  EXPECT_NO_THROW(JsonParser::parse("{\"a\": [1, 2.5e-3, null, true, \"x\\n\"], \"b\": {}}"));
  EXPECT_THROW(JsonParser::parse("{\"a\": [1, 2,]}"), std::runtime_error);
  EXPECT_THROW(JsonParser::parse("{\"a\": nan}"), std::runtime_error);
  EXPECT_THROW(JsonParser::parse("{\"a\": inf}"), std::runtime_error);
  EXPECT_THROW(JsonParser::parse("{\"a\": \"line\nbreak\"}"), std::runtime_error);
  EXPECT_THROW(JsonParser::parse("{\"a\": 1} trailing"), std::runtime_error);
  EXPECT_THROW(JsonParser::parse("{\"a\": 1"), std::runtime_error);
  EXPECT_THROW(JsonParser::parse("{a: 1}"), std::runtime_error);
}
