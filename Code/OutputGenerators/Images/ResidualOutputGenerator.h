#pragma once

#include "../../Core/Types.h"

class ResidualOutputGenerator {
    std::vector<DataSeries> data_series;
    const std::string name;
    std::string fname, ofname;
    std::string plot_title;
    bool generated_plot = false;
    bool generated_script = false;

public:
    ResidualOutputGenerator();
    ResidualOutputGenerator(std::string in_name, std::string in_title);
    
    void push_value(double value);
    void close_current_series();
    void new_series(std::string name);
    void write_gnuplot_file();
    void run_gnuplot();
};


