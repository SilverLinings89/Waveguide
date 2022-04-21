#pragma once
/**
 * @file ShapeFunction.h
 * @author Pascal Kraft
 * @brief Stores the implementation of the ShapeFunction Class.
 * @version 0.1
 * @date 2022-04-21
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#include <vector>

/**
 * @brief These objects are used in the shape optimization code. They have a certain number of degrees of freedom and are used for the description of coordinate transformations.
 * These functions are described in the optimization chapter of the dissertation document.
 */
class ShapeFunction {

    const double sector_length = 0;
    unsigned int n_sectors;
    std::vector<double> dof_values;
    double f_0 = 0;
    double f_1 = 0;
    double df_0 = 0;
    double df_1 = 0;
    double z_min = 0;
    double z_max = 0;
    bool bad_init = false;
    public:
    const unsigned int n_free_dofs;
    const unsigned int n_dofs;

    /**
     * @brief For a provided number of sectors, this provides the number of degrees of freedom the function will have.
     * See the chapter in the dissertation for details.
     * @param in_n_sectors Number of sectors of the function.
     * @return unsigned int Number of degrees of freedom of the shape function.
     */
    static unsigned int compute_n_dofs(unsigned int in_n_sectors);
    
    /**
     * @brief Computes how many unconstrained dofs a shape function will have (static).
     * See the chapter in the dissertation for details.
     * @param in_n_sectors Number of sectors of the function.
     * @return unsigned int Number of degrees of unconstrained degrees of freedom of the shape function.
     */
    static unsigned int compute_n_free_dofs(unsigned int in_n_sectors);

    /**
     * @brief Construct a new Shape Function object
     * These functions a parametrized by the z coordinate. Therefore, the constructor requires the z-range. Additionally we need the number of sectors. Per sector, there is an additional degree of freedom.
     * The bad init flag triggers a bad initialization of the values such that an optimization algorithm has some space for optimization.
     * @param in_z_min Lower end-point of the range.
     * @param in_z_max Upper end-point of the range.
     * @param in_n_sectors Number of sectors.
     * @param in_bad_init Bad-init flag triggers 0-initialization to give optimization some play.
     */
    ShapeFunction(double in_z_min, double in_z_max, unsigned int in_n_sectors, bool in_bad_init = false);

    /**
     * @brief Evaluates the shape function for a given z-coordinate.
     * 
     * @param z z-coordinate to evaluate the function at.
     * @return double function value at that z-coordinate.
     */
    double evaluate_at(double z) const;

    /**
     * @brief Evaluates the shape function derivative for a given z-coordinate.
     * 
     * @param z z-coordinate to evaluate the derivative of the function at.
     * @return double derivative of the function at provided z-coordinate.
     */
    double evaluate_derivative_at(double z) const;

    /**
     * @brief Sets the default constraints for these types of function.
     * The constraints are usually function value and derivative at the upper and lower boundary, i.e. for \f$z = z_{min}\f$ and \f$z = z_{max}\f$.
     * @param in_f_0 \f$f(z_{min})\f$ 
     * @param in_f_1 \f$f(z_{max})\f$
     * @param in_df_0 \f$\frac{\partial f}{\partial z}(z_{min})\f$
     * @param in_df_1 \f$\frac{\partial f}{\partial z}(z_{max})\f$
     */
    void set_constraints(double in_f_0, double in_f_1, double in_df_0, double in_df_1);

    /**
     * @brief We only store the derivative values and the values of the function at the lower and upper limit. During the computation we only consider the derivatives for the shape gradient. Of these values the highest and lowest index are constrained directly (typically to zero) and an additional constraint is computed based on the difference between the function value at input and output.
     * 
     */
    void update_constrained_values();
    
    /**
     * @brief Set the free dof values.
     * This function gets called by the optimization method.
     * @param in_dof_values The values to set.
     */
    void set_free_values(std::vector<double> in_dof_values);

    /**
     * @brief Get the number of degrees of freedom of this object.
     * 
     * @return unsigned int Number of dofs.
     */
    unsigned int get_n_dofs() const;

    /**
     * @brief Get the number of unconstrained degrees of freedom of this object.
     * This is the number of dofs that can be varied during the optimization.
     * 
     * @return unsigned int Number of free dofs.
     */
    unsigned int get_n_free_dofs() const;

    /**
     * @brief Get the value of a dof.
     * 
     * @param index The index of the dof.
     * @return double The value of the dof.
     */
    double get_dof_value(unsigned int index) const;

    /**
     * @brief Same as get_dof_value but in free dof numbering, so index 0 is the first free dof and the last one  is the last free dof.
     * 
     * @param index Index of the free dof to query for.
     * @return double Value of that dof.
     */
    double get_free_dof_value(unsigned int index) const;

    /**
     * @brief Sets up the object by computing initial values for the shape dofs based on the boundary constraints.
     * 
     */
    void initialize();

    /**
     * @brief Set the value of the index-th free dof to value.
     * 
     * @param index Index of the dof.
     * @param value Value of the dof.
     */
    void set_free_dof_value(unsigned int index, double value);

    /**
     * @brief Prints some cosmetic output about a shape function.
     * 
     */
    void print();
};