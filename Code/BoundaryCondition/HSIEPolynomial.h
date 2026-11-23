#pragma once

/**
 * @file HSIEPolynomial.h
 * @author Pascal Kraft (kraft.pascal@gmail.com)
 * @brief Contains the implementation of a Hardy polynomial which is required for the Hardy Space infinite elements.
 * @version 0.1
 * @date 2022-03-22
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <vector>
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
  std::vector<ComplexNumber> a; ///< Coefficients of the polynomial. a[i] belongs to the monomial x^i.
  std::vector<ComplexNumber> da; ///< Coefficients of the derivative, da[i] = (i+1) a[i+1]. Kept up to date by update_derivative().
  /** @brief Parameter kappa_0 of the Moebius transform of the Hardy space method. Only required by applyD() and applyI(). */
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
     */
  void update_derivative();

  /**
     *  @brief Prepares the Tensors D and I that are required for some of the computations.
     *
     *  @details For the defnition of D see the publication on "High order Curl-conforming Hardy spce infinite elements for exterior Maxwell problems" equation 21. D has tri-diagonal shape and represents the derivative for the Laplace-Moebius transformed shape of a function.
     * The matrix I is the inverse of D and also gets computed in this function. These matrices are required in many places and never change. They, therefore, are only computed once and made available statically.
     * The operator D (and I in turn) can be applied to polynomials of any degree. The computation of I, however gets more expensive the larger the maximal degree of the polynomials becomes. We therefore provide the maximal value of the dimension of polynomials.
     * 
     *  @param dim Number of rows and columns of D and I. Polynomials with up to this many coefficients can be handled.
     *  @param k_0 This is a parameter of HSIE and also impacts D (and I).
     */
  static void computeDandI(unsigned int dim,ComplexNumber k_0 );

  /**
    *  @brief Computes a polynomial that is required for the shape functions of HSIE. 
    *
    *  @details See "High order Curl-conforming Hardy spce infinite elements for exterior Maxwell problems" equation 44.
    *  This is the function that carries the boundary value: \f$\frac{1}{i \kappa_0} T_-(1, 0)\f$ (see applyTminus()).
    * 
    *  @param k0 This is a parameter of  the underlying Moebius transform of HSIE.
    *  @return An object of type HSIEPolynomial that fulfills the equation mentioned above.
    */
  static HSIEPolynomial PsiMinusOne(ComplexNumber k0);
  
  /**
    *  @brief Computes a polynomial that is required for the shape functions of HSIE. 
    *
    *  @details See "High order Curl-conforming Hardy spce infinite elements for exterior Maxwell problems" equation 44.
    *  For j >= 0 this is \f$\frac{1}{i \kappa_0} T_-(0, x^j)\f$ (see applyTminus()). For j = -1 it returns PsiMinusOne().
    * 
    *  @param j The order of the Polynomial, i.e. the element degree it is associated with.
    *  @param k0 This is a parameter of  the underlying Moebius transform of HSIE.
    *  @return An object of type HSIEPolynomial that fulfills the equation mentioned above.
    */
  static HSIEPolynomial PsiJ(int j, ComplexNumber k0);
  
  /**
     *  @brief Prepares an empty HSIE Polynomial
     *
     *  @return A polynomial with four coefficients that are all zero. Its value of k0 is zero.
     */
  static HSIEPolynomial ZeroPolynomial();

  /**
    *  @brief Computes a polynomial that is required for the shape functions of HSIE. 
    *
    *  @details See "High order Curl-conforming Hardy spce infinite elements for exterior Maxwell problems" equation 41.
    *  This is the function that carries the boundary value: \f$T_+(1, 0)\f$ (see applyTplus()).
    * 
    *  @param k0 This is a parameter of  the underlying Moebius transform of HSIE.
    *  @return An object of type HSIEPolynomial that fulfills the equation mentioned above.
    */
  static HSIEPolynomial PhiMinusOne(ComplexNumber k0);

  /**
    *  @brief Computes a polynomial that is required for the shape functions of HSIE. 
    *
    *  @details See "High order Curl-conforming Hardy spce infinite elements for exterior Maxwell problems" equation 41.
    *  For j >= 0 this is \f$T_+(0, x^j)\f$ (see applyTplus()). For j = -1 it returns PhiMinusOne().
    * 
    *  @param j The order of the Polynomial, i.e. the element degree it is associated with.
    *  @param k0 This is a parameter of  the underlying Moebius transform of HSIE.
    *  @return An object of type HSIEPolynomial that fulfills the equation mentioned above.
    */
  static HSIEPolynomial PhiJ(int j, ComplexNumber k0);

  /**
    *  @brief Constructs the monomial \f$x^{dim}\f$.
    *
    *  @param dim Degree of the monomial. The coefficient vector has dim + 1 entries.
    *  @param k0 This is a parameter of  the underlying Moebius transform of HSIE.
    */
  HSIEPolynomial(unsigned int dim ,ComplexNumber k0);

    /**
    *  @brief Constructor shape function polynomials.
    *
    *  @details Uses the data from the DofData object to initialize a monomial. For data.hsie_order >= 1 this is \f$x^{hsie\_order - 1}\f$, otherwise it is the constant 1.
    * 
    *  @param data The dof this polynomial should be used in. Only its member hsie_order is evaluated.
    *  @param k_0 This is a parameter of  the underlying Moebius transform of HSIE.
    */
  HSIEPolynomial(DofData& data, ComplexNumber k_0);

    /**
    *  @brief Constructs a polynomial from its coefficients.
    * 
    *  @param in_a The coefficients. in_a[i] belongs to the monomial \f$x^i\f$.
    *  @param k0 This is a parameter of  the underlying Moebius transform of HSIE.
    */
  HSIEPolynomial(std::vector<ComplexNumber> in_a, ComplexNumber k0);

  /** @brief True once computeDandI() has filled D and I. */
  static bool matricesLoaded;
  /** @brief Matrix of the operator D with respect to the monomial basis (see computeDandI()). Shared by all objects. */
  static dealii::FullMatrix<ComplexNumber> D;
  /** @brief Inverse of the matrix D. */
  static dealii::FullMatrix<ComplexNumber> I;

  /**
    *  @brief Applies the operator D to the object.
    *
    *  @details To compute the way dofs couple, we need to apply the operator D to them. See for example equation 18 in "High order Curl-conforming Hardy spce infinite elements for exterior Maxwell problems".
    *  The matrices D and I are computed first if they are missing or too small. This object is not changed.
    * 
    *  @return The polynomial whose coefficient vector is D times the coefficient vector of this object.
    */
  HSIEPolynomial applyD();

  /**
    *  @brief Applies the operator I to the object.
    *
    *  @details To compute the way dofs couple, we need to apply the operator I to them. See for example equation 30 in "High order Curl-conforming Hardy spce infinite elements for exterior Maxwell problems".
    *  The matrices D and I are computed first if they are missing or too small. This object is not changed.
    * 
    *  @return The polynomial whose coefficient vector is I times the coefficient vector of this object.
    */
  HSIEPolynomial applyI();

  /**
    *  @brief Multiplies all coefficients \f$a_i\f$ by a complex factor.
    *
    *  @details Mutates this object instead of returning a new one.
    *  @param factor The constant factor to multiply by.
    */
  void multiplyBy(ComplexNumber factor);

  /**
    *  @brief Multiply with a double. Simply multiplies the factors \f$a_i\f$ by that factor.
    *
    *  @details  Mutates this object instead of returning a new one.
    *  @param factor The constant factor to multiply by.
    */
  void multiplyBy(double factor);

  /**
    *  @brief Applies the operator \f$T_+\f$ to this object.
    *
    *  @details This is an operator required to compute the entries of the stiffness matrix. See equation 8 in "High order Curl-conforming Hardy spce infinite elements for exterior Maxwell problems" for a definition.
    *  If U is the polynomial stored in this object, it is replaced by \f$T_+(u_0, U)(x) = \frac{1}{2}\left(u_0 + (x + 1) U(x)\right)\f$. The result is stored in place and has one coefficient more.
    *  @param u_0 The boundary value this object extends.
    */
  void applyTplus(ComplexNumber u_0);

  /**
    *  @brief Applies the operator \f$T_-\f$ to this object.
    *
    *  @details This is an operator required to compute the entries of the stiffness matrix. See equation 8 in "High order Curl-conforming Hardy spce infinite elements for exterior Maxwell problems" for a definition.
    *  If U is the polynomial stored in this object, it is replaced by \f$T_-(u_0, U)(x) = \frac{1}{2}\left(u_0 + (x - 1) U(x)\right)\f$. The result is stored in place and has one coefficient more.
    *  @param u_0 The boundary value this object extends.
    */
  void applyTminus(ComplexNumber u_0);

  /**
    *  @brief Applies the derivative to this polynomial and stores the result in place.
    */
  void applyDerivative();

  /**
    *  @brief Computes the sum of this polynomial and the input parameter.
    *  @details Simple add operation. Stores the result in place.
    *  @param b the polynomial to add onto the one represented by this object.
    */
  void add(HSIEPolynomial b);
};
