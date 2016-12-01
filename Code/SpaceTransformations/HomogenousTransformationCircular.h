#ifndef HomogenousTransformationCircularFlag
#define HomogenousTransformationCircularFlag


/**
 * \class HomogenousTransformationCircular
 * \brief For this transformation we try to achieve a situation in which tensorial material properties from the coordinate transformation and PML-regions dont overlap.
 *
 * The usage of a coordinate transformation which is identity on the domain containing our PML is a strong restriction however it ensures lower errors since the quality of the PML is harder to estimate otherwise. Also it limits us in how we model the waveguide essentially forcing us to have no bent between the wavguides-connectors.
 * \author Pascal Kraft
 * \date 28.11.2016
 */
class HomogenousTransformationCircular : public SpaceTransformation {


};

#endif HomogenousTransformationCircularFlag
