#pragma once

// Helpers for the unit tests of the helper classes, the parameters, the run report and the output generators:
// temporary directories, guards that restore the global objects, and a small strict JSON parser.
#include <gtest/gtest.h>
#include <unistd.h>
#include <array>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "../Code/GlobalObjects/GlobalObjects.h"

namespace helpers_a {

  /** A directory below the temporary directory of the system that is removed when the object goes out of scope. */
  class TemporaryDirectory {
   public:
    explicit TemporaryDirectory(const std::string & name) {
      static unsigned int counter = 0;
      path_ = std::filesystem::temp_directory_path() / ("waveguide_tests_" + name + "_" + std::to_string(::getpid()) + "_" + std::to_string(counter++));
      std::filesystem::remove_all(path_);
      std::filesystem::create_directories(path_);
    }
    ~TemporaryDirectory() {
      std::error_code ignored;
      std::filesystem::remove_all(path_, ignored);
    }
    std::string path() const { return path_.string(); }
    std::string file(const std::string & name) const { return (path_ / name).string(); }

   private:
    std::filesystem::path path_;
  };

  inline void write_file(const std::string & filename, const std::string & content) {
    std::ofstream out(filename);
    out << content;
  }

  inline std::string read_file(const std::string & filename) {
    std::ifstream in(filename);
    std::stringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
  }

  inline std::vector<std::string> read_lines(const std::string & filename) {
    std::ifstream in(filename);
    std::vector<std::string> ret;
    std::string line;
    while (std::getline(in, line)) ret.push_back(line);
    return ret;
  }

  /**
   * Replaces GlobalParams by default parameters of a single process that only prints messages of the highest level,
   * and restores the previous object at the end of the scope.
   */
  class ScopedGlobalParams {
   public:
    ScopedGlobalParams() : saved(GlobalParams) {
      GlobalParams = Parameters();
      GlobalParams.MPI_Rank = 0;
      GlobalParams.NumberProcesses = 1;
      GlobalParams.Index_in_x_direction = 0;
      GlobalParams.Index_in_y_direction = 0;
      GlobalParams.Index_in_z_direction = 0;
      GlobalParams.Logging_Level = LoggingLevel::PRODUCTION_ONE;
    }
    ~ScopedGlobalParams() { GlobalParams = saved; }

   private:
    Parameters saved;
  };

  /** Restores the coordinate ranges and the surface types of the global geometry at the end of the scope. */
  class ScopedGeometry {
   public:
    ScopedGeometry() :
        global_x(Geometry.global_x_range), global_y(Geometry.global_y_range), global_z(Geometry.global_z_range),
        local_x(Geometry.local_x_range), local_y(Geometry.local_y_range), local_z(Geometry.local_z_range) {
      for (unsigned int level = 0; level < 4; level++) surface_types[level] = Geometry.levels[level].surface_type;
    }
    ~ScopedGeometry() {
      Geometry.global_x_range = global_x;
      Geometry.global_y_range = global_y;
      Geometry.global_z_range = global_z;
      Geometry.local_x_range = local_x;
      Geometry.local_y_range = local_y;
      Geometry.local_z_range = local_z;
      for (unsigned int level = 0; level < 4; level++) Geometry.levels[level].surface_type = surface_types[level];
    }

   private:
    std::pair<double, double> global_x, global_y, global_z, local_x, local_y, local_z;
    std::array<std::array<SurfaceType, 6>, 4> surface_types;
  };

  /** Points the output folder of the global output manager to a directory and restores it at the end of the scope. */
  class ScopedOutputFolder {
   public:
    explicit ScopedOutputFolder(const std::string & folder) : saved(GlobalOutputManager.output_folder_path) {
      GlobalOutputManager.output_folder_path = folder;
    }
    ~ScopedOutputFolder() { GlobalOutputManager.output_folder_path = saved; }

   private:
    std::string saved;
  };

  /** Sets or removes an environment variable and restores its previous state at the end of the scope. */
  class ScopedEnvironmentVariable {
   public:
    ScopedEnvironmentVariable(const std::string & in_name, const char * value) : name(in_name) {
      const char * old = std::getenv(name.c_str());
      was_set = old != nullptr;
      if (was_set) saved = old;
      if (value != nullptr) {
        setenv(name.c_str(), value, 1);
      } else {
        unsetenv(name.c_str());
      }
    }
    ~ScopedEnvironmentVariable() {
      if (was_set) {
        setenv(name.c_str(), saved.c_str(), 1);
      } else {
        unsetenv(name.c_str());
      }
    }

   private:
    std::string name;
    std::string saved;
    bool was_set;
  };

  /** A parsed JSON value. */
  struct Json {
    enum class Type { Null, Bool, Number, String, Array, Object };
    Type type = Type::Null;
    bool boolean = false;
    double number = 0;
    std::string string;
    std::vector<Json> array;
    std::vector<std::pair<std::string, Json>> object;

    bool has(const std::string & key) const {
      for (const auto & member : object) {
        if (member.first == key) return true;
      }
      return false;
    }
    const Json & at(const std::string & key) const {
      for (const auto & member : object) {
        if (member.first == key) return member.second;
      }
      throw std::runtime_error("JSON object has no member " + key);
    }
    const Json & at(size_t index) const { return array.at(index); }
    size_t size() const { return type == Type::Object ? object.size() : array.size(); }
  };

  /**
   * A strict parser for JSON as defined in RFC 8259. It throws std::runtime_error for every input that is not valid
   * JSON, also for unescaped control characters in strings and for trailing content.
   */
  class JsonParser {
   public:
    static Json parse(const std::string & text) {
      JsonParser parser(text);
      parser.skip_whitespace();
      Json ret = parser.value();
      parser.skip_whitespace();
      if (parser.pos != text.size()) parser.fail("content after the end of the document");
      return ret;
    }

   private:
    explicit JsonParser(const std::string & in_text) : text(in_text) {}
    const std::string & text;
    size_t pos = 0;

    [[noreturn]] void fail(const std::string & message) const {
      throw std::runtime_error("Invalid JSON at offset " + std::to_string(pos) + ": " + message);
    }
    char peek() const {
      if (pos >= text.size()) fail("unexpected end");
      return text[pos];
    }
    void expect(char c) {
      if (peek() != c) fail(std::string("expected ") + c);
      pos++;
    }
    void skip_whitespace() {
      while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\n' || text[pos] == '\r' || text[pos] == '\t')) pos++;
    }
    void literal(const std::string & word) {
      if (text.compare(pos, word.size(), word) != 0) fail("unknown literal");
      pos += word.size();
    }
    Json value() {
      Json ret;
      const char c = peek();
      if (c == '{') {
        ret.type = Json::Type::Object;
        pos++;
        skip_whitespace();
        if (peek() == '}') {
          pos++;
          return ret;
        }
        while (true) {
          skip_whitespace();
          const std::string key = string();
          skip_whitespace();
          expect(':');
          skip_whitespace();
          ret.object.emplace_back(key, value());
          skip_whitespace();
          if (peek() == ',') {
            pos++;
            continue;
          }
          expect('}');
          return ret;
        }
      }
      if (c == '[') {
        ret.type = Json::Type::Array;
        pos++;
        skip_whitespace();
        if (peek() == ']') {
          pos++;
          return ret;
        }
        while (true) {
          skip_whitespace();
          ret.array.push_back(value());
          skip_whitespace();
          if (peek() == ',') {
            pos++;
            continue;
          }
          expect(']');
          return ret;
        }
      }
      if (c == '"') {
        ret.type = Json::Type::String;
        ret.string = string();
        return ret;
      }
      if (c == 't') {
        literal("true");
        ret.type = Json::Type::Bool;
        ret.boolean = true;
        return ret;
      }
      if (c == 'f') {
        literal("false");
        ret.type = Json::Type::Bool;
        return ret;
      }
      if (c == 'n') {
        literal("null");
        return ret;
      }
      ret.type = Json::Type::Number;
      ret.number = number();
      return ret;
    }
    std::string string() {
      expect('"');
      std::string ret;
      while (true) {
        const char c = peek();
        pos++;
        if (c == '"') return ret;
        if (static_cast<unsigned char>(c) < 0x20) fail("control character in a string");
        if (c != '\\') {
          ret += c;
          continue;
        }
        const char escaped = peek();
        pos++;
        switch (escaped) {
          case '"': ret += '"'; break;
          case '\\': ret += '\\'; break;
          case '/': ret += '/'; break;
          case 'b': ret += '\b'; break;
          case 'f': ret += '\f'; break;
          case 'n': ret += '\n'; break;
          case 'r': ret += '\r'; break;
          case 't': ret += '\t'; break;
          case 'u': {
            if (pos + 4 > text.size()) fail("incomplete unicode escape");
            const unsigned long code = std::stoul(text.substr(pos, 4), nullptr, 16);
            pos += 4;
            if (code > 0x7f) fail("the test parser only supports ASCII unicode escapes");
            ret += static_cast<char>(code);
            break;
          }
          default: fail("unknown escape sequence");
        }
      }
    }
    double number() {
      const size_t begin = pos;
      if (peek() == '-') pos++;
      if (pos >= text.size() || !std::isdigit(static_cast<unsigned char>(text[pos]))) fail("expected a digit");
      if (text[pos] == '0' && pos + 1 < text.size() && std::isdigit(static_cast<unsigned char>(text[pos + 1]))) fail("leading zero");
      while (pos < text.size() && std::isdigit(static_cast<unsigned char>(text[pos]))) pos++;
      if (pos < text.size() && text[pos] == '.') {
        pos++;
        if (pos >= text.size() || !std::isdigit(static_cast<unsigned char>(text[pos]))) fail("expected a digit after the decimal point");
        while (pos < text.size() && std::isdigit(static_cast<unsigned char>(text[pos]))) pos++;
      }
      if (pos < text.size() && (text[pos] == 'e' || text[pos] == 'E')) {
        pos++;
        if (pos < text.size() && (text[pos] == '+' || text[pos] == '-')) pos++;
        if (pos >= text.size() || !std::isdigit(static_cast<unsigned char>(text[pos]))) fail("expected a digit in the exponent");
        while (pos < text.size() && std::isdigit(static_cast<unsigned char>(text[pos]))) pos++;
      }
      return std::stod(text.substr(begin, pos - begin));
    }
  };

}
