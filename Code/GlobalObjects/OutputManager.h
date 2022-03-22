#pragma once

/**
 * @file OutputManager.h
 * @author your name (you@domain.com)
 * @brief Creates filenames and manages file system paths.
 * @version 0.1
 * @date 2022-03-22
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "../Core/Types.h"
#include <sys/stat.h>
#include <iostream>
#include <fstream>

/**
 * @brief Whenever we write output, we require filenames. This object wraps the functionality of generating unique filenames for each process, boundary etc.
 * 
 */

class OutputManager {
    public: 
    std::string base_path;
    unsigned int run_number;
    std::string output_folder_path;
    std::ofstream log_stream;
    
    OutputManager();
    ~OutputManager();

    /**
     * @brief Ensures the output directory exists and writes some basic output files like the run description.
     * 
     */
    void initialize();

    /**
     * @brief Creates a full filename that can be used with an std::ofstream based on a core part provided as an argument.
     * 
     * @param filename The core bit of the full path (in Solutions/run356/solution.vtk this would be solution.vtk)-
     * @return std::string The full filename with relative path.
     */
    std::string get_full_filename(std::string filename);

    /**
     * @brief Gives a full filename with relative path for a provided core part, identifier and extension.
     * This can be used whenever we know that multiple processes will call the same output method and provide the rank on every process to make sure the processess dont interfere with eachothers files.
     * 
     * @param filename Main part of the filename
     * @param number Unique bit to differentiate between processes or boundary conditions or levels.
     * @param extension File extension to be appended at the end
     * @return std::string Fully qualified filename to use for the generation of output.
     */
    std::string get_numbered_filename(std::string filename, unsigned int number, std::string extension);

    /**
     * @brief Writes a line of output to the processes output text file.
     * 
     * @param in_line The text to be written to the log.
     */
    void write_log_ling(std::string in_line);

    /**
     * @brief Generates a file in the output folder with some core data about the run.
     * 
     * @param git_commit_hash This git hash will be included in the output to describe in which state the code was.
     */
    void write_run_description(std::string git_commit_hash);
};