#pragma once

/**
 * @file ResidualOutputGenerator.h
 * @author Pascal Kraft
 * @brief Contains the class ResidualOutputGenerator, which stores and plots the residuals of the iterative solver.
 */
#include "../../Core/Types.h"

/**
 * @brief Stores the residual history of the iterative solver of one NonLocalProblem and writes it to the console
 * and to a gnuplot plot.
 *
 * Every solve of the problem is one series. The solver monitor pushes the residual of every GMRES step.
 */
class ResidualOutputGenerator {
    /** @brief Base name of the output files. */
    const std::string name;
    /** @brief Names of the gnuplot script (.plt) and of the image (.png) in the output folder. Both contain the global MPI rank. */
    std::string fname, ofname;
    /** @brief Title of the plot. */
    std::string plot_title;
    /** @brief True once gnuplot has been run. It is run at most once. */
    bool generated_plot = false;
    /** @brief True once the gnuplot script has been written. It is written at most once. */
    bool generated_script = false;
    /** @brief True if this process prints the residuals to the console. This is the process with rank 0 in the sweep of the problem. */
    const bool print_to_console = false;
    /** @brief Level of the problem the residuals belong to. */
    const unsigned int level;
    /** @brief Indentation of the console output: one tab per level below the top level. */
    std::string spacing = "";
    /** @brief Rank of this process among the processes of the problem (NonLocalProblem::total_rank_in_sweep). */
    unsigned int rank_in_sweep;
    /** @brief Index of the problem in the sweep of the level above, printed in front of each residual. -1 for the top level. */
    int parent_sweeping_rank;


public:
    /** @brief The residual histories. One series per solve, in the order of the solves. */
    std::vector<DataSeries> data_series; // one series of residuals per solve
    /** @brief Creates an object for level 0 without name that does not print to the console. */
    ResidualOutputGenerator();
    /**
     * @brief Sets the file names and the console output behavior.
     *
     * @param in_name Base name of the script and the image file.
     * @param in_title Title of the plot.
     * @param in_rank_in_sweep Rank of this process among the processes of the problem. Only rank 0 prints to the console.
     * @param in_level Level of the problem.
     * @param in_parent_sweeping_rank Index of the problem in the sweep of the level above or -1.
     */
    ResidualOutputGenerator(std::string in_name, std::string in_title, unsigned int in_rank_in_sweep, unsigned int in_level, int in_parent_sweeping_rank);
    
    /**
     * @brief Appends a residual to the current series. Does nothing if there is no series yet.
     *
     * @param value The residual norm.
     */
    void push_value(double value);
    /** @brief Marks the current series as closed. Called after a solve has finished. */
    void close_current_series();
    /**
     * @brief Starts a new series. Called at the beginning of every solve.
     *
     * @param name Name of the series, used as its legend entry in the plot.
     */
    void new_series(std::string name);
    /** @brief Writes the gnuplot script that plots all series over the step index with a logarithmic y-axis. Only the first call has an effect. */
    void write_gnuplot_file();
    /** @brief Writes the gnuplot script and runs gnuplot on it. Only the first call has an effect. */
    void run_gnuplot();
    /** @brief Prints the step number and the last residual of the current series and, from the second step on, its ratio to the previous residual. Only on the process that prints to the console. */
    void write_residual_statement_to_console();
};


