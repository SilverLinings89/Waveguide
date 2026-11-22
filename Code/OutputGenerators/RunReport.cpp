#include "RunReport.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include "../GlobalObjects/GlobalObjects.h"

RunReport GlobalRunReport;

RunReport::RunReport() : start(std::chrono::steady_clock::now()) {}

void RunReport::section(const std::string & title) {
  for(unsigned int i = 0; i < sections.size(); i++) {
    if(sections[i].title == title) {
      current = i;
      return;
    }
  }
  Section new_section;
  new_section.title = title;
  sections.push_back(new_section);
  current = sections.size() - 1;
}

void RunReport::entry(const std::string & name, const std::string & value) {
  if(sections.size() == 0) section("Run");
  // An entry that is set again replaces the earlier value (for example after the last solve of an optimization).
  for(auto & existing : sections[current].entries) {
    if(existing.first == name) {
      existing.second = value;
      return;
    }
  }
  sections[current].entries.push_back(std::make_pair(name, value));
}

void RunReport::entry(const std::string & name, const char * value) { entry(name, std::string(value)); }
void RunReport::entry(const std::string & name, double value) { entry(name, number(value)); }
void RunReport::entry(const std::string & name, int value) { entry(name, count(value)); }
void RunReport::entry(const std::string & name, unsigned int value) { entry(name, count(value)); }
void RunReport::entry(const std::string & name, bool value) { entry(name, std::string(value ? "yes" : "no")); }

void RunReport::table(const std::string & title, const std::vector<std::string> & header, const std::vector<std::vector<std::string>> & rows) {
  if(sections.size() == 0) section("Run");
  Table new_table;
  new_table.title = title;
  new_table.header = header;
  new_table.rows = rows;
  for(auto & existing : sections[current].tables) {
    if(existing.title == title) {
      existing = new_table;
      return;
    }
  }
  sections[current].tables.push_back(new_table);
}

void RunReport::note(const std::string & text) {
  if(sections.size() == 0) section("Run");
  if(std::find(sections[current].notes.begin(), sections[current].notes.end(), text) == sections[current].notes.end()) {
    sections[current].notes.push_back(text);
  }
}

void RunReport::data(const std::string & name, double value) {
  for(auto & existing : numbers) {
    if(existing.first == name) {
      existing.second = value;
      return;
    }
  }
  numbers.push_back(std::make_pair(name, value));
}

void RunReport::data(const std::string & name, const std::string & value) {
  for(auto & existing : texts) {
    if(existing.first == name) {
      existing.second = value;
      return;
    }
  }
  texts.push_back(std::make_pair(name, value));
}

void RunReport::series(const std::string & name, const std::vector<double> & values) {
  all_series.push_back(std::make_pair(name, values));
}

double RunReport::seconds_since_start() const {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

std::string RunReport::number(double value, unsigned int digits) {
  std::ostringstream out;
  out << std::setprecision(digits) << value;
  return out.str();
}

std::string RunReport::count(double value) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(0) << std::abs(value);
  std::string digits = out.str();
  std::string ret;
  for(unsigned int i = 0; i < digits.size(); i++) {
    if(i > 0 && (digits.size() - i) % 3 == 0) ret += ",";
    ret += digits[i];
  }
  return (value < 0 ? "-" : "") + ret;
}

void RunReport::write() {
  if(GlobalParams.MPI_Rank != 0) return;
  write_text(GlobalOutputManager.get_full_filename("run_report.txt"));
  write_json(GlobalOutputManager.get_full_filename("run_report.json"));
}

void RunReport::write_text(const std::string & filename) const {
  std::ofstream out(filename);
  out << "RUN REPORT" << std::endl << "==========" << std::endl;
  for(unsigned int s = 0; s < sections.size(); s++) {
    const Section & sec = sections[s];
    const std::string title = std::to_string(s + 1) + ". " + sec.title;
    out << std::endl << title << std::endl << std::string(title.size(), '-') << std::endl;
    size_t width = 0;
    for(const auto & e : sec.entries) width = std::max(width, e.first.size());
    for(const auto & e : sec.entries) {
      out << "  " << std::left << std::setw(width) << e.first << " : " << e.second << std::endl;
    }
    for(const Table & t : sec.tables) {
      out << std::endl << "  " << t.title << std::endl;
      std::vector<size_t> widths(t.header.size(), 0);
      for(unsigned int c = 0; c < t.header.size(); c++) widths[c] = t.header[c].size();
      for(const auto & row : t.rows) {
        for(unsigned int c = 0; c < row.size() && c < widths.size(); c++) widths[c] = std::max(widths[c], row[c].size());
      }
      out << "  ";
      for(unsigned int c = 0; c < t.header.size(); c++) out << "  " << std::left << std::setw(widths[c]) << t.header[c];
      out << std::endl << "  ";
      for(unsigned int c = 0; c < t.header.size(); c++) out << "  " << std::string(widths[c], '-');
      out << std::endl;
      for(const auto & row : t.rows) {
        out << "  ";
        for(unsigned int c = 0; c < row.size() && c < widths.size(); c++) out << "  " << std::left << std::setw(widths[c]) << row[c];
        out << std::endl;
      }
    }
    if(sec.notes.size() > 0) out << std::endl;
    for(const auto & n : sec.notes) {
      out << "  Note: " << n << std::endl;
    }
  }
  out << std::endl << "Figures: run examples/run_report.py on this folder to get report.html with plots of the field," << std::endl;
  out << "the signal along the waveguide and the convergence of the solver." << std::endl;
}

static std::string json_string(const std::string & in) {
  std::string ret = "\"";
  for(char c : in) {
    if(c == '"' || c == '\\') {
      ret += '\\';
      ret += c;
    } else if(c == '\n') {
      ret += "\\n";
    } else if(c == '\t') {
      ret += " ";
    } else {
      ret += c;
    }
  }
  return ret + "\"";
}

static std::string json_number(double value) {
  if(!std::isfinite(value)) return "null";
  std::ostringstream out;
  out << std::setprecision(12) << value;
  return out.str();
}

void RunReport::write_json(const std::string & filename) const {
  std::ofstream out(filename);
  out << "{" << std::endl << "\"sections\": [" << std::endl;
  for(unsigned int s = 0; s < sections.size(); s++) {
    const Section & sec = sections[s];
    out << "  {\"title\": " << json_string(sec.title) << "," << std::endl << "   \"entries\": [";
    for(unsigned int i = 0; i < sec.entries.size(); i++) {
      out << (i > 0 ? ", " : "") << "[" << json_string(sec.entries[i].first) << ", " << json_string(sec.entries[i].second) << "]";
    }
    out << "]," << std::endl << "   \"tables\": [";
    for(unsigned int t = 0; t < sec.tables.size(); t++) {
      const Table & tab = sec.tables[t];
      out << (t > 0 ? ", " : "") << "{\"title\": " << json_string(tab.title) << ", \"header\": [";
      for(unsigned int c = 0; c < tab.header.size(); c++) out << (c > 0 ? ", " : "") << json_string(tab.header[c]);
      out << "], \"rows\": [";
      for(unsigned int r = 0; r < tab.rows.size(); r++) {
        out << (r > 0 ? ", " : "") << "[";
        for(unsigned int c = 0; c < tab.rows[r].size(); c++) out << (c > 0 ? ", " : "") << json_string(tab.rows[r][c]);
        out << "]";
      }
      out << "]}";
    }
    out << "]," << std::endl << "   \"notes\": [";
    for(unsigned int i = 0; i < sec.notes.size(); i++) out << (i > 0 ? ", " : "") << json_string(sec.notes[i]);
    out << "]}" << (s + 1 < sections.size() ? "," : "") << std::endl;
  }
  out << "]," << std::endl << "\"data\": {";
  bool first = true;
  for(const auto & n : numbers) {
    out << (first ? "" : ", ") << std::endl << "  " << json_string(n.first) << ": " << json_number(n.second);
    first = false;
  }
  for(const auto & t : texts) {
    out << (first ? "" : ", ") << std::endl << "  " << json_string(t.first) << ": " << json_string(t.second);
    first = false;
  }
  out << std::endl << "}," << std::endl << "\"series\": [";
  for(unsigned int i = 0; i < all_series.size(); i++) {
    out << (i > 0 ? "," : "") << std::endl << "  {\"name\": " << json_string(all_series[i].first) << ", \"values\": [";
    for(unsigned int j = 0; j < all_series[i].second.size(); j++) out << (j > 0 ? ", " : "") << json_number(all_series[i].second[j]);
    out << "]}";
  }
  out << std::endl << "]" << std::endl << "}" << std::endl;
}
