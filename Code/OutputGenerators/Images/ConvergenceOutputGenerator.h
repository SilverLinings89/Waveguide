#pragma once

/**
 * @file ConvergenceOutputGenerator.h
 * @author Pascal Kraft
 * @brief Contains the class ConvergenceOutputGenerator, which plots the errors of a convergence study.
 */
#include <string>
#include <vector>
#include "../../Core/Types.h"

/**
 * @brief Collects the errors of a convergence study (ConvergenceRun) and plots them with gnuplot.
 *
 * Every data point consists of an x-value and two errors: the numerical error (difference to the solution on the
 * finest mesh) and the theoretical error (difference to the exact solution). The plot is doubly logarithmic.
 */
class ConvergenceOutputGenerator {
    /** @brief Title of the plot. */
    std::string title;
    /** @brief x-values of the data points. ConvergenceRun passes the number of dofs. */
    std::vector<double> x;
    /** @brief Numerical error of every data point. */
    std::vector<double> y_numerical;
    /** @brief Theoretical error of every data point. */
    std::vector<double> y_theoretical;
    /** @brief Label of the x-axis. Stays empty because set_labels() is not implemented. */
    std::string x_label;
    /** @brief Label of the y-axis. Stays empty because set_labels() is not implemented. */
    std::string y_label;
    /** @brief Name of the gnuplot script in the output folder: "Convergence" followed by the MPI rank and ".plt". */
    std::string fname;
    /** @brief Name of the image gnuplot writes. Same as the script name with the extension ".png". */
    std::string ofname;

public:
    /** @brief Sets an empty title. */
    ConvergenceOutputGenerator();
    /** @brief Does nothing. */
    ~ConvergenceOutputGenerator();

    /**
     * @brief Sets the title of the plot.
     *
     * @param in_title The title.
     */
    void set_title(std::string in_title);
    /**
     * @brief Declared but not implemented.
     *
     * @param x_label Label of the x-axis.
     * @param y_label Label of the y-axis.
     */
    void set_labels(std::string x_label, std::string y_label);
    /**
     * @brief Appends one data point.
     *
     * @param x x-value, for example the number of dofs.
     * @param y_num Numerical error.
     * @param y_theo Theoretical error.
     */
    void push_values(double x, double y_num, double y_theo);
    /** @brief Writes the gnuplot script with all data points collected so far. */
    void write_gnuplot_file();
    /** @brief Writes the gnuplot script and runs gnuplot on it, which creates the image. */
    void run_gnuplot();
};


