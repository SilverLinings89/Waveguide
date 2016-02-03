
#ifndef STATICFUNCTIONS_H_
#define STATICFUNCTIONS_H_

using namespace dealii;

/**
 * For given vectors \f$\boldsymbol{a},\boldsymbol{b} \in \mathbb{R}^3\f$, this function calculates the following crossproduct:
 * \f[\boldsymbol{a} \ times \boldsymbol{b} = \begin{pmatrix} a_2 b_3 - a_3b_2 \\ a_3b_1 - a_1b_3 \\ a_1b_2 - a_2b_1\end{pmatrix}\f]
 */
inline Tensor<1, 3 , double> crossproduct(Tensor<1, 3, double> a, Tensor<1, 3, double> b);


inline double dotproduct(Tensor<1, 3, double> a, Tensor<1, 3, double> b);


static Point<3> Triangulation_Stretch_X (const Point<3> &p);


static Point<3> Triangulation_Stretch_Y (const Point<3> &p);


static Point<3> Triangulation_Stretch_Z (const Point<3> &p);


static double Solution (Point<3, double> p ,const unsigned int component);


inline bool file_exists (const std::string& name);


#endif /* STATICFUNCTIONS_H_ */
