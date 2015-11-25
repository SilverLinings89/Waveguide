
#ifndef STATICFUNCTIONS_H_
#define STATICFUNCTIONS_H_

using namespace dealii;

/**
 * This function is very useful. It takes a standard filename and generates from it a structure with parsed variable values via a ParameterReader-ParameterHandler workflow. This encapsulates all the effort and makes the parameters easily available in the code. The function is used in several constructors since it spares the effort of passing too many arguments and improves the structural code quality by relying more heavily on the input file.
 * \return The return value is a structure of the type Parameters that contains either the default values or the values in the input file. Values from the file are dominant if both are present.
 */
static Parameters GetParameters();

/**
 * This function calculates the distance (norm of the difference) of the projection of two vectors in the \f$ xy\f$-plane. A formal description would be
 * \f[ d(\boldsymbol(x), \boldsymbol(y)) = ||\left( \boldsymbol{x} - \boldsymbol{y} \right) \cdot \begin{pmatrix}1\\1\\0\end{pmatrix}||_2 \f]
 */
static double Distance2D (Point<3> position, Point<3> to = Point<3>());

/**
 * For given vectors \f$\boldsymbol{a},\boldsymbol{b} \in \mathbb{R}^3\f$, this function calculates the following crossproduct:
 * \f[\boldsymbol{a} \ times \boldsymbol{b} = \begin{pmatrix} a_2 b_3 - a_3b_2 \\ a_3b_1 - a_1b_3 \\ a_1b_2 - a_2b_1\end{pmatrix}\f]
 */
inline Tensor<1, 3 , double> crossproduct(Tensor<1, 3, double> a, Tensor<1, 3, double> b);


inline double dotproduct(Tensor<1, 3, double> a, Tensor<1, 3, double> b);


template<int dim> static void mesh_info(const Triangulation<dim> &tria, const std::string &filename);


static Point<3> Triangulation_Stretch_X (const Point<3> &p);


static Point<3> Triangulation_Stretch_Y (const Point<3> &p);


static Point<3> Triangulation_Stretch_Z (const Point<3> &p);


static Point<3> Triangulation_Stretch_Real (const Point<3> &p);


static bool System_Coordinate_in_Waveguide(Point<3> p);


static double TEMode00 (Point<3, double> p ,const unsigned int component);


static double Solution (Point<3, double> p ,const unsigned int component);


inline bool file_exists (const std::string& name);


#endif /* STATICFUNCTIONS_H_ */
