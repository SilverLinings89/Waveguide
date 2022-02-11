#include "ParameterOverride.h"
#include "staticfunctions.h"

ParameterOverride::ParameterOverride() {
    has_overrides = false;
}

bool ParameterOverride::read(std::string in_string) {
    if(!validate(in_string)) {
        return false;
    }
    std::vector<std::string> blocks = split(in_string, ";");
    for(unsigned int i = 0; i < blocks.size(); i++) {
        std::vector<std::string> line_split = split(blocks[i], "=");
        overrides.push_back(std::pair<std::string, std::string>(line_split[0], line_split[1]));
        has_overrides = true;
    }
    return true;
}

bool ParameterOverride::validate(std::string in_string) {
    if(in_string.size() < 4) {
        return false;
    }
    if (in_string.find('=') == std::string::npos) {
        return false;
    }
    std::vector<std::string> blocks = split(in_string, ";");
    for(unsigned int i = 0; i < blocks.size(); i++) {
        std::vector<std::string> line_split = split(blocks[i], "=");
        if(line_split.size() != 2) {
            return false;
        }
    }
    return true;
}

void ParameterOverride::perform_on(Parameters& in_parameters) {
    for(unsigned int i = 0; i < overrides.size(); i++) {
        if(overrides[i].first == "n_pml_cells") {
            print_info("ParameterOverride", "Replacing pml_n_cells with " + overrides[i].second);
            in_parameters.PML_N_Layers = std::stoi(overrides[i].second);
        }
        if(overrides[i].first == "pml_sigma_max") {
            print_info("ParameterOverride", "Replacing pml_sigma_max with " + overrides[i].second);
            in_parameters.PML_Sigma_Max = std::stod(overrides[i].second);
        }
        if(overrides[i].first == "pml_order") {
            print_info("ParameterOverride", "Replacing pml_order with " + overrides[i].second);
            in_parameters.PML_skaling_order = std::stoi(overrides[i].second);
        }
        if(overrides[i].first == "solver_type") {
            print_info("ParameterOverride", "Replacing iterative solver with " + overrides[i].second);
            in_parameters.solver_type = solver_option(overrides[i].second);
        }
        if(overrides[i].first == "geometry_size_z") {
            print_info("ParameterOverride", "Replacing geometry size z with " + overrides[i].second);
            in_parameters.Geometry_Size_Z = stod(overrides[i].second);
        }
        if(overrides[i].first == "processes_in_z") {
            print_info("ParameterOverride", "Replacing number of processes in z with " + overrides[i].second);
            in_parameters.Blocks_in_z_direction = stoi(overrides[i].second);
        }
    }
}