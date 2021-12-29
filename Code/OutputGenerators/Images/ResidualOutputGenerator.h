#pragma once

#include "../../Core/Types.h"

class ResidualOutputGenerator {
    std::vector<DataSeries> data_series;
    const std::string name;
    std::string fname, ofname;
    std::string plot_title;
    bool generated_plot = false;
    bool generated_script = false;
    const bool print_to_console = false;
    const unsigned int level;
    std::string spacing = "";
    unsigned int rank_in_sweep;
    int parent_sweeping_rank;


public:
    ResidualOutputGenerator();
    ResidualOutputGenerator(std::string in_name, std::string in_title, unsigned int in_rank_in_sweep, unsigned int in_level, int in_parent_sweeping_rank);
    
    void push_value(double value);
    void close_current_series();
    void new_series(std::string name);
    void write_gnuplot_file();
    void run_gnuplot();
    void write_residual_statement_to_console();
};


