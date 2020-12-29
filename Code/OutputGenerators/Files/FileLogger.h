#pragma once

#include "../../Core/Types.h"
#include <string>
#include <iostream>

/**
 * There will be one global instance of this object. It creates file paths and provides file names.
 * Every IO operation will be piped through this object.
 * The other loggers use it to persist their data.
 * 
 */

class FileLogger {

    FileLogger();
    ~FileLogger(); 

    auto initialize() -> void;
    auto get_file_name(FileType, FileMetaData) -> std::string;
    auto get_ofstream(FileType, FileMetaData) -> std::ofstream;
};