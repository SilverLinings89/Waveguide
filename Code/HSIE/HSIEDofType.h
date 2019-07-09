/*
 * HSIEDofType.h
 *  Important remark. I always consider the infinite direction to be the third
 * one internally and then rotate vectors accordingly. I get a 2D surface
 * triangulation and the "infinite" direction is orthogonal to those. It is
 * therefore important to notice the differences to the paper, where x is the
 * direction which is removed. I want to be able to also use this implementation
 * for boundaries in the x and y direction. Created on: Oct 8, 2018 Author:
 * kraft
 */

#ifndef CODE_HSIE_HSIEDOFTYPE_H_
#define CODE_HSIE_HSIEDOFTYPE_H_

#include <deal.II/base/tensor.h>
#include <deal.II/lac/full_matrix.h>
#include <complex>
#include <vector>

enum HSIE_Infinite_Direction {
    x = 0, y = 1, z = 2, general = 3
};

template<int hsie_order>
class HSIE_Dof_Type {
private:
    const HSIE_Infinite_Direction dir;
    static bool D_and_I_initialized;
    static dealii::Tensor<2, hsie_order + 1, std::complex<double>> D;
    static dealii::Tensor<2, hsie_order + 1, std::complex<double>> I;
    unsigned int type;
    unsigned int order;
    std::vector<std::complex<double>> hardy_monomial_base;
    std::vector<std::complex<double>> IPsiK;
    std::vector<std::complex<double>> dxiPsiK;
    dealii::FullMatrix<double> *shape_functions;
    dealii::FullMatrix<double> *gradients;

public:
    HSIE_Dof_Type(unsigned int in_type, unsigned int in_order,
                  unsigned int q_count, HSIE_Infinite_Direction dir,
                  dealii::FullMatrix<double> *shape_functions,
                  dealii::FullMatrix<double> *gradients);

    virtual ~HSIE_Dof_Type();

    std::complex<double> eval_base(std::vector<std::complex<double>> *in_base,
                                   std::complex<double> in_x);

    /**
     * This describes the properties of a HSIE dof type. The versions are
  descirbed in 3.2 of "High order Curl-conforming Hardy space infinite elements
  for exterior Maxwell problems". Possible types are:
     * 0. edge functions
     * 1. surface functions
     * 2. ray functions
     * 3. infinite face functions type 1
     * 4. infinite face functions type 2
     * 5. segment functions type 1
     * 6. segment functions type 2
     */
    unsigned int get_type();

    /*
     * returns the order of the dof.
     */
    unsigned int get_order();

    // This functions computes the values of w_k and v_k for a given set of
    // quadrature points.
    void prepare_for_quadrature_points(
            std::vector<dealii::Point<2, double>> q_points);

    void compute_IPsik();

    void compute_dxiPsik();

    std::vector<std::complex<double>> evaluate_U(dealii::Point<2, double>,
                                                 double xi);

    std::vector<std::complex<double>> evaluate_U_for_ACT(dealii::Point<2, double>,
                                                         double xi);

    std::complex<double> component_1a(double x, double y,
                                      std::complex<double> xhi);

    std::complex<double> component_1b(double x, double y,
                                      std::complex<double> xhi);

    std::complex<double> component_2a(double x, double y,
                                      std::complex<double> xhi);

    std::complex<double> component_2b(double x, double y,
                                      std::complex<double> xhi);

    std::complex<double> component_3a(double x, double y,
                                      std::complex<double> xhi);

    std::complex<double> component_3b(double x, double y,
                                      std::complex<double> xhi);

    std::vector<std::complex<double>> apply_T_plus(
            std::vector<std::complex<double>>, double);

    std::vector<std::complex<double>> apply_T_minus(
            std::vector<std::complex<double>>, double);
};

#endif /* CODE_HSIE_HSIEDOFTYPE_H_ */
