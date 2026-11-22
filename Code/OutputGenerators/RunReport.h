#pragma once

/**
 * @file RunReport.h
 * @brief Collects a description of the run (setup, unknowns, solver statistics, results) and writes it as a text
 * file for reading and as a JSON file for post-processing.
 */

#include <chrono>
#include <map>
#include <string>
#include <utility>
#include <vector>

/**
 * \class RunReport
 * \brief The report of one run of the solver.
 *
 * The report consists of sections. A section has entries (name and value), tables and notes. In addition, numbers
 * and series of numbers can be stored for post-processing; they only appear in the JSON file. The different parts
 * of the code add what they know (the runner, the hierarchical problem, the optimization). At the end of the run
 * process 0 writes run_report.txt and run_report.json to the output folder. examples/run_report.py turns these and
 * the field outputs into report.html with figures.
 */
class RunReport {
 public:
  struct Table {
    std::string title;
    std::vector<std::string> header;
    std::vector<std::vector<std::string>> rows;
  };

  struct Section {
    std::string title;
    std::vector<std::pair<std::string, std::string>> entries;
    std::vector<Table> tables;
    std::vector<std::string> notes;
  };

  RunReport();

  /** Selects the section with this title for the following calls. It is created if it does not exist. */
  void section(const std::string & title);

  void entry(const std::string & name, const std::string & value);
  void entry(const std::string & name, const char * value);
  void entry(const std::string & name, double value);
  void entry(const std::string & name, int value);
  void entry(const std::string & name, unsigned int value);
  void entry(const std::string & name, bool value);

  void table(const std::string & title, const std::vector<std::string> & header, const std::vector<std::vector<std::string>> & rows);

  void note(const std::string & text);

  /** Stores a number for post-processing (JSON only). */
  void data(const std::string & name, double value);

  /** Stores a text for post-processing (JSON only). */
  void data(const std::string & name, const std::string & value);

  /** Stores a series of numbers for post-processing (JSON only). Several series can have the same name. */
  void series(const std::string & name, const std::vector<double> & values);

  /** Wall time since the start of the program in seconds. */
  double seconds_since_start() const;

  /** Writes run_report.txt and run_report.json. Does nothing on processes other than process 0. */
  void write();

  /** Formats a number with the given number of significant digits. */
  static std::string number(double value, unsigned int digits = 6);

  /** Formats an integer with separators between groups of three digits. */
  static std::string count(double value);

 private:
  std::vector<Section> sections;
  unsigned int current = 0;
  std::vector<std::pair<std::string, double>> numbers;
  std::vector<std::pair<std::string, std::string>> texts;
  std::vector<std::pair<std::string, std::vector<double>>> all_series;
  std::chrono::steady_clock::time_point start;

  void write_text(const std::string & filename) const;
  void write_json(const std::string & filename) const;
};

extern RunReport GlobalRunReport;
