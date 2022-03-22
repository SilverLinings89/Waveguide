#pragma once

/**
 * @file ParameterOverride.h
 * @author your name (you@domain.com)
 * @brief A utility class that overrides certain parameters from an input file.
 * @version 0.1
 * @date 2022-03-22
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <string>
#include "Parameters.h"

/**
 * @brief An object used to interpret command line arguments of type --override. This is usefull when we re-run the same code and only want to vary one or few parameter values.
 * Without this object type we would need paramter files for all combinations. With this type, we define the overrides and create base parameter files for all the other parameters.
 * 
 */
class ParameterOverride {
    std::vector<std::pair<std::string, std::string>> overrides;
    public: 
    bool has_overrides;
    ParameterOverride();

    /**
     * @brief Checks if the provided override string is valid and if so parses it.
     * 
     * @return true The input was valid and parsing it was successful.
     * @return false There was an error
     */
    bool read(std::string);

    /**
     * @brief Performs the parsed overrides on the provided parameter object.
     * 
     * @param in_p The parameter object to be updated (in place)
     */
    void perform_on(Parameters & in_p);

    /**
     * @brief Checks if the provided override string is a valid set of parameters and values.
     * 
     * @param in_arg The parameter value of the override argument passed to the main application.
     * @return true This can be used as an override
     * @return false There was an error
     */
    bool validate(std::string in_arg);
};