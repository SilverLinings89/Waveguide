#ifndef DualProblemTransformationWrapperFlag
#define DualProblemTransformationWrapperFlag


/**
 * \class DualProblemTransformationWrapper
 * \brief If we do an adjoint Computation, we need a SpaceTransformation, which has the same properties as the primal one but measures in transformed coordinates. This Wrapper contains the space transformation of the primal version but maps input parameters to their dual equivalent.
 *
 * Essentialy this class enables us to write a waveguide class which is unaware of its being primal or dual. Using this wrapper makes us compute the solution of the inverse order shape parametrization.
 * \author Pascal Kraft
 * \date 1.12.2016
 */
class DualProblemTransformationWrapper : public SpaceTransformation {


};

#endif DualProblemTransformationWrapperFlag
