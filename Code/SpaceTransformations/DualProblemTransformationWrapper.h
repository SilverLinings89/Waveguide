#ifndef DualProblemTransformationWrapperFlag
#define DualProblemTransformationWrapperFlag

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/vector.h>
#include <math.h>
#include <vector>
#include "../Core/Sector.h"
#include "SpaceTransformation.h"

using namespace dealii;

/**
 * \class DualProblemTransformationWrapper
 * \brief If we do an adjoint computation, we need a SpaceTransformation, which
 * has the same properties as the primal one but measures in transformed
 * coordinates. This Wrapper contains the space transformation of the primal
 * version but maps input parameters to their dual equivalent.
 *
 * Essentially this class enables us to write a waveguide class which is unaware
 * of its being primal or dual. Using this wrapper makes us compute the solution
 * of the inverse order shape parametrization. \author Pascal Kraft
 * \date 1.12.2016
 */
class DualProblemTransformationWrapper : public SpaceTransformation {
public:
    /**
     * Since this object encapsulates another Space Transformation, the
     * construction is straight forward. \param non_dual_st This pointer points to
     * the actual transformation that is being wrapped.
     */
    DualProblemTransformationWrapper(SpaceTransformation *non_dual_st, int rank);

    ~DualProblemTransformationWrapper();

    /**
     * One of the core functionalities of a SpaceTransformation is to map a
     * mathematical coordinate to a physical one (so an transformed to an
     * untransformed coordinate). \param coord the coordinate to be transformed.
     * In this class we simply pass the
     */
    Point<3> math_to_phys(Point<3> coord) const;

    /**
     * This function does the same as math_to_phys only in the opposit direction.
     * \param coord the coordinate to be transformed.
     */
    Point<3> phys_to_math(Point<3> coord) const;

    /**
     * In order to test implementation, this function was added to check, if the
     * transformation-tensor at a given coordinate is the identity or not. \param
     * coord This is the coordinate to test.
     */
    bool is_identity(Point<3> coord) const;

    Tensor<2, 3, std::complex<double>> get_Tensor(Point<3> &coordinate) const;

    Tensor<2, 3, std::complex<double>> get_Preconditioner_Tensor(
            Point<3> &coordinate, int block) const;

    Tensor<2, 3, std::complex<double>> Apply_PML_To_Tensor(
            Point<3> &coordinate, Tensor<2, 3, double> Tensor_input) const;

    Tensor<2, 3, std::complex<double>> Apply_PML_To_Tensor_For_Preconditioner(
            Point<3> &coordinate, Tensor<2, 3, double> Tensor_input, int block) const;

    Tensor<2, 3, double> get_Space_Transformation_Tensor(
            Point<3> &coordinate) const;

    Tensor<2, 3, double> get_Space_Transformation_Tensor_Homogenized(
            Point<3> &coordinate) const;

    const double XMinus, XPlus, YMinus, YPlus, ZMinus, ZPlus;

    /**
     * This function is used to determine, if a system-coordinate belongs to a
     * PML-region for the PML that limits the computational domain along the
     * x-axis. Since there are 3 blocks of PML-type material, there are 3
     * functions. \param position Stores the position in which to test for
     * presence of a PML-Material.
     */
    bool PML_in_X(Point<3> &position) const;

    /**
     * This function is used to determine, if a system-coordinate belongs to a
     * PML-region for the PML that limits the computational domain along the
     * y-axis. Since there are 3 blocks of PML-type material, there are 3
     * functions. \param position Stores the position in which to test for
     * presence of a PML-Material.
     */
    bool PML_in_Y(Point<3> &position) const;

    /**
     * This function is used to determine, if a system-coordinate belongs to a
     * PML-region for the PML that limits the computational domain along the
     * z-axis. Since there are 3 blocks of PML-type material, there are 3
     * functions. \param position Stores the position in which to test for
     * presence of a PML-Material.
     */
    bool PML_in_Z(Point<3> &position) const;

    /**
     * This function fulfills the same purpose as those with similar names but it
     * is supposed to be used together with Preconditioner_PML_in_Z instead of the
     * versions without "Preconditioner".
     */
    double Preconditioner_PML_Z_Distance(Point<3> &p, unsigned int block) const;

    /**
     * This function calculates for a given point, its distance to a PML-boundary
     * limiting the computational domain. This function is used merely to make
     * code more readable. There is a function for every one of the dimensions
     * since the normal vectors of PML-regions in this implementation are the
     * coordinate-axis. This value is set to zero outside the PML and positive
     * inside both PML-domains (only one for the z-direction). \param position
     * Stores the position from which to calculate the distance to the
     * PML-surface.
     */
    double PML_X_Distance(Point<3> &position) const;

    /**
     * This function calculates for a given point, its distance to a PML-boundary
     * limiting the computational domain. This function is used merely to make
     * code more readable. There is a function for every one of the dimensions
     * since the normal vectors of PML-regions in this implementation are the
     * coordinate-axis. This value is set to zero outside the PML and positive
     * inside both PML-domains (only one for the z-direction). \param position
     * Stores the position from which to calculate the distance to the
     * PML-surface.
     */

    double PML_Y_Distance(Point<3> &position) const;

    /**
     * This function calculates for a given point, its distance to a PML-boundary
     * limiting the computational domain. This function is used merely to make
     * code more readable. There is a function for every one of the dimensions
     * since the normal vectors of PML-regions in this implementation are the
     * coordinate-axis. This value is set to zero outside the PML and positive
     * inside both PML-domains (only one for the z-direction). \param position
     * Stores the position from which to calculate the distance to the
     * PML-surface.
     */
    double PML_Z_Distance(Point<3> &position) const;

    /**
     * This member contains all the Sectors who, as a sum, form the complete
     * Waveguide. These Sectors are a partition of the simulated domain.
     */
    std::vector<Sector<3>> case_sectors;

    /**
     * The material-property \f$\epsilon_r\f$ has a different value inside and
     * outside of the waveguides core. This variable stores its value inside the
     * core.
     */
    const double epsilon_K;
    /**
     *  The material-property \f$\epsilon_r\f$ has a different value inside and
     * outside of the waveguides core. This variable stores its value outside the
     * core.
     */
    const double epsilon_M;
    /**
     * Since the computational domain is split into subdomains (called sectors),
     * it is important to keep track of the amount of subdomains. This member
     * stores the number of Sectors the computational domain has been split into.
     */
    const int sectors;

    /**
     * This value is initialized with the value Delta from the input-file.
     */
    const double deltaY;

    /**
     * At the beginning (before the first solution of a system) only the boundary
     * conditions for the shape of the waveguide are known. Therefore the values
     * for the degrees of freedom need to be estimated. This function sets all
     * variables to appropiate values and estimates an appropriate shape based on
     * averages and a polynomial interpolation of the boundary conditions on the
     * shape.
     */
    void estimate_and_initialize();

    /**
     * This member calculates the value of Q1 for a provided \f$z\f$-coordinate.
     * This value is used in the transformation of the solution-vector in
     * transformed coordinates (solution of the system-matrix) to real coordinates
     * (physical field). \param z The value of Q1 is independent of \f$x\f$ and
     * \f$y\f$. Therefore only a \f$z\f$-coordinate is provided in a call to the
     * function.
     */
    double get_Q1(double z) const;

    /**
     * This member calculates the value of Q2 for a provided \f$z\f$-coordinate.
     * This value is used in the transformation of the solution-vector in
     * transformed coordinates (solution of the system-matrix) to real coordinates
     * (physical field). \param z The value of Q2 is independent of \f$x\f$ and
     * \f$y\f$. Therefore only a \f$z\f$-coordinate is provided in a call to the
     * function.
     */
    double get_Q2(double z) const;

    /**
     * This member calculates the value of Q3 for a provided \f$z\f$-coordinate.
     * This value is used in the transformation of the solution-vector in
     * transformed coordinates (solution of the system-matrix) to real coordinates
     * (physical field). \param z The value of Q3 is independent of \f$x\f$ and
     * \f$y\f$. Therefore only a \f$z\f$-coordinate is provided in a call to the
     * function.
     */
    double get_Q3(double z) const;

    /**
     * This is a getter for the values of degrees of freedom. A getter-setter
     * interface was introduced since the values are estimated automatically
     * during the optimization and non-physical systems should be excluded from
     * the domain of possible cases. \param dof The index of the degree of freedom
     * to be retrieved from the structure of the modelled waveguide. \return This
     * function returns the value of the requested degree of freedom. Should this
     * dof not exist, 0 will be returnd.
     */
    double get_dof(int dof) const;

    /**
     * This function sets the value of the dof provided to the given value. It is
     * important to consider, that some dofs are non-writable (i.e. the values of
     * the degrees of freedom on the boundary, like the radius of the
     * input-connector cannot be changed). \param dof The index of the parameter
     * to be changed. \param value The value, the dof should be set to.
     */
    void set_dof(int dof, double value);

    /**
     * This is a getter for the values of degrees of freedom. A getter-setter
     * interface was introduced since the values are estimated automatically
     * during the optimization and non-physical systems should be excluded from
     * the domain of possible cases. \param dof The index of the degree of freedom
     * to be retrieved from the structure of the modelled waveguide. \return This
     * function returns the value of the requested degree of freedom. Should this
     * dof not exist, 0 will be returnd.
     */
    double get_free_dof(int dof) const;

    /**
     * This function sets the value of the dof provided to the given value. It is
     * important to consider, that some dofs are non-writable (i.e. the values of
     * the degrees of freedom on the boundary, like the radius of the
     * input-connector cannot be changed). \param dof The index of the parameter
     * to be changed. \param value The value, the dof should be set to.
     */
    void set_free_dof(int dof, double value);

    /**
     * Using this method unifies the usage of coordinates. This function takes a
     * global \f$z\f$ coordinate (in the computational domain) and returns both a
     * Sector-Index and an internal \f$z\f$ coordinate indicating which sector
     * this coordinate belongs to and how far along in the sector it is located.
     * \param double in_z global system \f$z\f$ coordinate for the transformation.
     */
    std::pair<int, double> Z_to_Sector_and_local_z(double in_z) const;

    /**
     * Returns the complete length of the computational domain.
     */
    double System_Length() const;

    /**
     * Returns the length of one sector
     */
    double Sector_Length() const;

    /**
     * Returns the length of one layer
     */
    double Layer_Length() const;

    /**
     * Returns the radius for a system-coordinate;
     */
    double get_r(double in_z) const;

    /**
     * Returns the shift for a system-coordinate;
     */
    double get_m(double in_z) const;

    /**
     * Returns the tilt for a system-coordinate;
     */
    double get_v(double in_z) const;

    int Z_to_Layer(double) const;

    /**
     * This vector of values saves the initial configuration
     */
    Vector<double> InitialDofs;

    /**
     * Other objects can use this function to retrieve an array of the current
     * values of the degrees of freedom of the functional we are optimizing. This
     * also includes restrained degrees of freedom and other functions can be used
     * to determine this property. This has to be done because in different cases
     * the number of restrained degrees of freedom can vary and we want no logic
     * about this in other functions.
     */
    Vector<double> Dofs() const;

    /**
     * This function returns the number of unrestrained degrees of freedom of the
     * current optimization run.
     */
    unsigned int NFreeDofs() const;

    /**
     * This function returns the total number of DOFs including restrained ones.
     * This is the lenght of the array returned by Dofs().
     */
    unsigned int NDofs() const;

    /**
     * Since Dofs() also returns restrained degrees of freedom, this function can
     * be applied to determine if a degree of freedom is indeed free or
     * restrained. "restrained" means that for example the DOF represents the
     * radius at one of the connectors (input or output) and therefore we forbid
     * the optimization scheme to vary this value.
     */
    bool IsDofFree(int) const;

    /**
     * Console output of the current Waveguide Structure.
     */
    void Print() const;

    std::complex<double> evaluate_for_z_with_sum(double, double, NumericProblem *);

    std::complex<double> gauss_product_2D_sphere(double z, int n, double R,
                                                 double Xc, double Yc,
                                                 NumericProblem *in_w,
                                                 Evaluation_Metric in_m);

    SpaceTransformation *st;
};

#endif
