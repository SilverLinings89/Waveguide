#pragma once

#include "../Core/Types.h"
#include <sys/stat.h>
#include <iostream>
#include <fstream>

class OutputManager {
    public: 
    std::string base_path;
    unsigned int run_number;
    std::string output_folder_path;
    std::ofstream log_stream;
    
    OutputManager();
    ~OutputManager();

    void initialize();
    std::string get_full_filename(std::string filename);
    std::string get_numbered_filename(std::string filename, unsigned int number, std::string extension);
    void write_log_ling(std::string in_line);
    void write_run_description(std::string git_commit_hash);
};