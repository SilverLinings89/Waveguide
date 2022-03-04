#pragma once

#include <deal.II/lac/full_matrix.h>
#include "DofData.h"
#include "../Core/Types.h"

/**
 * \class HSIEPolynomial
 * 
 * \brief This class basically represents a polynomial and its derivative. It is required for the HSIE implementation.
 * 
 * \details The core data in this class is a vector a, which stores the coefficients of the polynomials and a vector da, which stores the coefficients of the derivative. Both can be evaluated for a given x with the respective functions.
 * Additionally, there are functions to initialize a polynomial that are required by the hardy space infinite elements and some operators can be applied (like T_plus and T_minus).
 * As an important remark: The value kappa_0 used in HSIE is also kept in these values because we want to be able to apply the operators D and I to one a polynomial. Since they aren't cheap to compute, I precomute them once as static members of this class.
 * If you only intend to use evaluation, evaluation of the derivative, summation and multiplication with constants, then that value is not relevant.
 * 
 * \see HSIESurface
 */

class HSIEPolynomial {
 public:
  std::vector<ComplexNumber> a; // This array stores the coefficients.
  std::vector<ComplexNumber> da; // This array stores the coefficients of the the derivative.
  ComplexNumber k0;
  
  /**
     *  @brief Evaluates the polynomial represented by this object at the given position x.
     *
     *  @details Performs the evaluation of the polynomial at x, meaning \f[f(x) = \sum\limits_{i=0}^D a_i x^i.\f]
     * 
     *  @param x The poisition to evaluate the polynomial at.
     *  @return The value of the polynomial at x.
    */
  ComplexNumber evaluate(ComplexNumber x);

  /**
     *  @brief Evaluates the derivative of the polynomial represented by this object at the given position x.
     *
     *  @details Performs the evaluation of the derivative of the polynomial at x, meaning \f[f(x) = \sum\limits_{i=1}^{D-1} i a_i x^{i-1}.\f]
     * 
     *  @param x The poisition to evaluate the derivative at.
     *  @return The value of the derivative of the polynomial at x.
     */
  ComplexNumber evaluate_dx(ComplexNumber x);

  /**
     *  @brief Updates the cached data for faster evaluation of the derivative.
     *
     *  @details Internally, the derivative is stored as a polynomial. The cached parameters are simply \f$i a_i\f$. This function gets called a lot internally, so calling it yourself is likely not required.
     * 
     *  @return Nothing.
     */
  void update_derivative();

  /**
     *  @brief Prepares the Tensors D and I that are required for some of the computations.
     *
     *  @details For the defnition of D see the publication on "High order Curl-conforming Hardy spce infinite elements for exterior Maxwell problems" equation 21. D has tri-diagonal shape and represents the derivative for the Laplace-Moebius transformed shape of a function.
     * The matrix I is the inverse of D and also gets computed in this function. These matrices are required in many places and never change. They, therefore, are only computed once and made available statically.
     * The operator D (and I in turn) can be applied to polynomials of any degree. The computation of I, however gets more expensive the larger the maximal degree of the polynomials becomes. We therefore provide the maximal value of the dimension of polynomials.
     * 
     *  @param dim Maximal polynomial degree of polynomials that D and I should be applied to.
     *  @param k_0 This is a parameter of HSIE and also impacts D (and I).
     *  @return Nothing.
     */
  static void computeDandI(unsigned int dim,ComplexNumber k_0 );

  /*
    *  @brief Computes a polynomial that is required for the shape functions of HSIE. 
    *
    *  @details See "High order Curl-conforming Hardy spce infinite elements for exterior Maxwell problems" equation 44.
    * 
    *  @param k_0 This is a parameter of  the underlying Moebius transform of HSIE.
    *  @return An object of type HSIEPolynomial that fulfills the equation mentioned above.
    */
  static HSIEPolynomial PsiMinusOne(ComplexNumber k0);
  
  /*
    *  @brief Computes a polynomial that is required for the shape functions of HSIE. 
    *
    *  @details See "High order Curl-conforming Hardy spce infinite elements for exterior Maxwell problems" equation 44.
    * 
    *  @param j The order of the Polynomial, i.e. the element degree it is associated with.
    *  @param k_0 This is a parameter of  the underlying Moebius transform of HSIE.
    *  @return An object of type HSIEPolynomial that fulfills the equation mentioned above.
    */
  static HSIEPolynomial PsiJ(int j, ComplexNumber k0);
  
  /*
     *  @brief Prepares an empty HSIE Polynomial
     *
     *  @return An empty polynomial.
     */
  static HSIEPolynomial ZeroPolynomial();

  /*
    *  @brief Computes a polynomial that is required for the shape functions of HSIE. 
    *
    *  @details See "High order Curl-conforming Hardy spce infinite elements for exterior Maxwell problems" equation 41.
    * 
    *  @param k_0 This is a parameter of  the underlying Moebius transform of HSIE.
    *  @return An object of type HSIEPolynomial that fulfills the equation mentioned above.
    */
  static HSIEPolynomial PhiMinusOne(ComplexNumber k0);

  /*
    *  @brief Computes a polynomial that is required for the shape functions of HSIE. 
    *
    *  @details See "High order Curl-conforming Hardy spce infinite elements for exterior Maxwell problems" equation 41.
    * 
    *  @param j The order of the Polynomial, i.e. the element degree it is associated with.
    *  @param k_0 This is a parameter of  the underlying Moebius transform of HSIE.
    *  @return An object of type HSIEPolynomial that fulfills the equation mentioned above.
    */
  static HSIEPolynomial PhiJ(int j, ComplexNumber k0);

  /*
    *  @brief Constructor for empty polynomial.
    *
    *  @details See "High order Curl-conforming Hardy spce infinite elements for exterior Maxwell problems" equation 41.
    * 
    *  @param dim Order of the polynomial.
    *  @param k_0 This is a parameter of  the underlying Moebius transform of HSIE.
    */
  HSIEPolynomial(unsigned int dim ,ComplexNumber k0);

    /*
    *  @brief Constructor shape function polynomials.
    *
    *  @details Uses the data from the DofData object to initialize a polynomial used in a shape function. It constructs a monomial of the order of the dof passed in as data.
    * 
    *  @param data This object contains the dimension as well as the order of the degree of freedom this polynomial should be used in.
    *  @param k_0 This is a parameter of  the underlying Moebius transform of HSIE.
    */
  HSIEPolynomial(DofData& data, ComplexNumber k_0);

    /*
    *  @brief Constructor shape function polynomials.
    *
    *  @details Uses the data from the DofData object to initialize a polynomial used in a shape function. It constructs a monomial of the order of the dof passed in as data.
    * 
    *  @param data This object contains the dimension as well as the order of the degree of freedom this polynomial should be used in.
    *  @param k_0 This is a parameter of  the underlying Moebius transform of HSIE.
    */
  HSIEPolynomial(std::vector<ComplexNumber> in_a, ComplexNumber k0);

  static bool matricesLoaded;
  static dealii::FullMatrix<ComplexNumber> D;
  static dealii::FullMatrix<ComplexNumber> I;

  /*
    *  @brief Applies the operator D to the object.
    *
    *  @details To compute the way dofs couple, we need to apply the operator D to them. See for example equation 18 in "High order Curl-conforming Hardy spce infinite elements for exterior Maxwell problems".
    * 
    */
  HSIEPolynomial applyD();

  /*
    *  @brief Applies the operator I to the object.
    *
    *  @details To compute the way dofs couple, we need to apply the operator I to them. See for example equation 30 in "High order Curl-conforming Hardy spce infinite elements for exterior Maxwell problems".
    * 
    */
  HSIEPolynomial applyI();

  /*
    *  @brief Multiplies this object with the one passed in as a parameter.
    *
    *  @details factor Mutates this object instead of returning a new one.
    *  @param The polynomial to multiply by.
    */
  void multiplyBy(ComplexNumber factor);

  /*
    *  @brief Multiply with a double. Simply multiplies the pactors \f$a_i\f$ by that factor.
    *
    *  @details  Mutates this object instead of returning a new one.
    *  @param factor The constant factor to multiply by.
    */
  void multiplyBy(double factor);

  /*
    *  @brief Applies the operator \f$T_+\f$ to this object.
    *
    *  @details This is an operator required to compute the entries of the stiffness matrix. See equation 8 in "High order Curl-conforming Hardy spce infinite elements for exterior Maxwell problems" for a definition.
    *  @param u_0 The boundary value this object extends.
    *  @return Stores the result in place.
    */
  void applyTplus(ComplexNumber u_0);

  /*
    *  @brief Applies the operator \f$T_-\f$ to this object.
    *
    *  @details This is an operator required to compute the entries of the stiffness matrix. See equation 8 in "High order Curl-conforming Hardy spce infinite elements for exterior Maxwell problems" for a definition.
    *  @param u_0 The boundary value this object extends.
    *  @return Stores the result in place.
    */
  void applyTminus(ComplexNumber u_0);

  /*
    *  @brief Applies the derivative to this polynomial and stores the result in place.
    *
    *  @return Stores the result in place.
    */
  void applyDerivative();

  /*
    *  @brief Computes the sum of this polynomial and the input parameter.
    *  @details Simple add operation. Stores the result in place.
    *  @param b the polynomial to add onto the one represented by this object.
    *  @return Stores the result in place.
    */
  void add(HSIEPolynomial b);
};
