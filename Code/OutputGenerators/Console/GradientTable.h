#ifndef GRADIENTTABLE_H_
#define GRADIENTTABLE_H_
/**
 * @file GradientTable.h
 * @author Pascal Kraft
 * @brief Contains the class GradientTable, which prints one step of a shape optimization as a table. Legacy code.
 */
#include <deal.II/lac/vector.h>

/**
 * \class GradientTable
 * \brief The Gradient Table is an OutputGenerator, intended to write information about the shape gradient to the console upon its computation.
 *
 * \date 28.11.2016
 * \author Pascal Kraft
 **/
class GradientTable {

private:

    /** @brief Step applied to each single dof, set by AddComputationResult(). Row "Delta" of the table. */
    dealii::Vector<double> steps;
    /** @brief Quality value reported for the step of each single dof, set by AddComputationResult(). Row "Quality" of the table. */
    dealii::Vector<double> qualities;
    /** @brief Change of each dof in the full step, set by AddFullStepResult(). Row "Gradient Step" of the table. */
    dealii::Vector<double> grad_step;
    /** @brief Dof values of the initial configuration. */
    dealii::Vector<double> ref_configuration;
    /** @brief Dof values before this optimization step. */
    dealii::Vector<double> last_configuration;

    /** @brief Quality value after the full step. */
    double final_quality;
    /** @brief Quality value of the initial configuration. */
    double initial_quality;
    /** @brief Quality value before this optimization step. */
    double last_quality;
public:
    /**
     * @brief Stores the data that is known before the gradient is computed.
     *
     * @param in_step Index of the optimization step.
     * @param in_configuration Dof values of the initial configuration. Only the first nfreedofs entries are used.
     * @param in_quality Quality value of the initial configuration.
     * @param in_last_configuration Dof values before this step.
     * @param in_last_quality Quality value before this step.
     */
    GradientTable(unsigned int in_step, dealii::Vector<double> in_configuration, double in_quality,
                  dealii::Vector<double> in_last_configuration, double in_last_quality);

    /** @brief Does nothing. */
    ~GradientTable();

    /** @brief Number of dofs of the shape. Fixed to 100 by the constructor and not used. */
    const int ndofs;

    /** @brief Number of columns of the table and length of all stored vectors. Fixed to 100 by the constructor. */
    const int nfreedofs;

    /** @brief Index of the optimization step this table belongs to. */
    const unsigned int GlobalStep;

    /**
     * @brief Declared but not implemented.
     *
     * @param in_quality Quality value of the initial configuration.
     */
    void SetInitialQuality(double in_quality);

    /**
     * @brief Stores the result of the computation in which one dof was changed.
     *
     * @param in_component Index of the dof.
     * @param in_step Step by which the dof was changed.
     * @param in_quality Quality value of the changed configuration.
     */
    void AddComputationResult(int in_component, double in_step, double in_quality);

    /**
     * @brief Stores the full optimization step and its result.
     *
     * @param in_step Change of every dof.
     * @param in_quality Quality value after the step.
     */
    void AddFullStepResult(dealii::Vector<double> in_step, double in_quality);

    /** @brief Prints a line of dashes across the width of the terminal. */
    void PrintFullLine();

    /** @brief Prints the table to the console: initial and last configuration, the single dof steps and their qualities, the full step and the new configuration. */
    void PrintTable();

    /**
     * @brief Empty implementation.
     *
     * @param in_filename Not used.
     */
    void WriteTableToFile(std::string in_filename);
};

#endif /* GRADIENTTABLE_H_ */
