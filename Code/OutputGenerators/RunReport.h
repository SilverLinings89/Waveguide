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
  /** @brief A table of a section: a title, the column headers and the rows. */
  struct Table {
    /** @brief Title of the table. It identifies the table within its section. */
    std::string title;
    /** @brief Column headers. */
    std::vector<std::string> header;
    /** @brief Rows of the table. Each row has one formatted text per column. */
    std::vector<std::vector<std::string>> rows;
  };

  /** @brief A section of the report. */
  struct Section {
    /** @brief Title of the section. It identifies the section. */
    std::string title;
    /** @brief Name and formatted value of every entry, in the order in which they were added. */
    std::vector<std::pair<std::string, std::string>> entries;
    /** @brief Tables of the section. */
    std::vector<Table> tables;
    /** @brief Free text remarks of the section. */
    std::vector<std::string> notes;
  };

  /** @brief Stores the current time as the start of the run. */
  RunReport();

  /** Selects the section with this title for the following calls. It is created if it does not exist. */
  void section(const std::string & title);

  /**
   * @brief Adds an entry to the current section. An entry with the same name is replaced.
   * If no section has been selected yet, the section "Run" is used.
   *
   * @param name Name of the entry.
   * @param value Value of the entry.
   */
  void entry(const std::string & name, const std::string & value);
  /** @brief Adds a text entry, see entry(const std::string &, const std::string &). */
  void entry(const std::string & name, const char * value);
  /** @brief Adds an entry whose value is formatted by number(). */
  void entry(const std::string & name, double value);
  /** @brief Adds an entry whose value is formatted by count(). */
  void entry(const std::string & name, int value);
  /** @brief Adds an entry whose value is formatted by count(). */
  void entry(const std::string & name, unsigned int value);
  /** @brief Adds an entry with the value "yes" or "no". */
  void entry(const std::string & name, bool value);

  /**
   * @brief Adds a table to the current section. A table with the same title is replaced.
   *
   * @param title Title of the table.
   * @param header Column headers.
   * @param rows Rows with one formatted text per column.
   */
  void table(const std::string & title, const std::vector<std::string> & header, const std::vector<std::vector<std::string>> & rows);

  /**
   * @brief Adds a remark to the current section unless the section already contains the same text.
   *
   * @param text The remark.
   */
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
  /** @brief All sections in the order of their creation. */
  std::vector<Section> sections;
  /** @brief Index in sections of the section selected by section(). */
  unsigned int current = 0;
  /** @brief Numbers stored by data() for the JSON file. */
  std::vector<std::pair<std::string, double>> numbers;
  /** @brief Texts stored by data() for the JSON file. */
  std::vector<std::pair<std::string, std::string>> texts;
  /** @brief Series stored by series() for the JSON file. */
  std::vector<std::pair<std::string, std::vector<double>>> all_series;
  /** @brief Time of construction of the report, used as the start of the run. */
  std::chrono::steady_clock::time_point start;

  /** @brief Writes the sections with their entries, tables and notes as plain text. */
  void write_text(const std::string & filename) const;
  /** @brief Writes the sections and the data stored for post-processing as JSON. */
  void write_json(const std::string & filename) const;
};

/** @brief The report of the current run. All parts of the code add to it and the runner writes it at the end of the run. */
extern RunReport GlobalRunReport;
