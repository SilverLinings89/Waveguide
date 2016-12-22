#ifndef InhomogenousTransformationRectangleFlag
#define InhomogenousTransformationRectangleFlag


/**
 * \class InhomogenousTransformationRectangle
 * \brief In this case we regard a rectangular waveguide and the effects on the material tensor by the space transformation and the boundary condition PML may overlap (hence inhomogenous space transformation)
 *
 * If this kind of boundary condition works stably we will also be able to  deal with more general settings (which might for example incorporate angles in between the output and input connector.
 * \author Pascal Kraft
 * \date 28.11.2016
 */
class InhomogenousTransformationRectangular : public SpaceTransformation {


};

#endif
 
