#ifndef InhomogenousTransformationCircularFlag
#define InhomogenousTransformationCircularFlag


/**
 * \class InhomogenousTransformationCircular
 * \brief In this case we regard a tubular waveguide and the effects on the material tensor by the space transformation and the boundary condition PML may overlap (hence inhomogenous space transformation)
 *
 * The usage of a coordinate transformation which is identity on the domain containing our PML is a strong restriction however it ensures lower errors since the quality of the PML is harder to estimate otherwise. Also it limits us in how we model the waveguide essentially forcing us to have no bent between the wavguides-connectors.
 * \author Pascal Kraft
 * \date 28.11.2016
 */
class InhomogenousTransformationCircular : public SpaceTransformation {


};

#endif InhomogenousTransformationCircularFlag
 
