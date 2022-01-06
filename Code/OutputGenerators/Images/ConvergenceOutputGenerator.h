#pragma once

#include "../../Core/Types.h"

class ConvergenceOutputGenerator {
    std::string title;
    std::vector<double> x;
    std::vector<double> y;
    std::string x_label;
    std::string y_label;
    std::string fname;
    std::string ofname;

public:
    ConvergenceOutputGenerator();
    ~ConvergenceOutputGenerator();

    void set_title(std::string in_title);
    void set_labels(std::string x_label, std::string y_label);
    void push_values(double x, double y);
    void write_gnuplot_file();
    void run_gnuplot();
};


