#pragma once
#include <string>
#include "Parameters.h"

class ParameterOverride {
    std::vector<std::pair<std::string, std::string>> overrides;
    public: 
    bool has_overrides;
    ParameterOverride();
    bool read(std::string);
    void perform_on(Parameters &);
    bool validate(std::string);
};