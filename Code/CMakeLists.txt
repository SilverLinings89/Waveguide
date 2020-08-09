INCLUDE_DIRECTORIES(Core, Helpers, MeshGenerators, OptimizationAlgorithm, OptimizationStrategies, SpaceTransformations, HSIEPreconditioner)

ADD_LIBRARY(Code
        Core/Sector.cpp
        Core/SolutionWeight.cpp
        Core/NumericProblem.cpp
        Core/Simulation.cpp
        Core/GlobalObjects.cpp
        Core/Types.h
        Helpers/GeometryManager.cpp
        Helpers/ExactSolution.cpp
        Helpers/ModeManager.cpp
        Helpers/ParameterReader.cpp
        Helpers/PointSourceField.cpp
        Helpers/Parameters.cpp
        Helpers/PointVal.cpp
        Helpers/QuadratureFormulaCircle.cpp
        Helpers/staticfunctions.cpp
        Helpers/ShapeDescription.cpp
        MeshGenerators/SquareMeshGenerator.cpp
        SpaceTransformations/DualProblemTransformationWrapper.cpp
        SpaceTransformations/HomogenousTransformationRectangular.cpp
        SpaceTransformations/InhomogenousTransformationRectangular.cpp
        SpaceTransformations/SpaceTransformation.cpp
        HSIEPreconditioner/HSIEPolynomial.cpp
        HSIEPreconditioner/HSIEPreconditioner.cpp
        HSIEPreconditioner/HSIESurface.cpp
        HSIEPreconditioner/LaguerreFunction.cpp
        HSIEPreconditioner/JacobianForCell.cpp
        Hierarchy/HierarchicalProblem.cpp 
        Hierarchy/HierarchicalProblem.h 
        Hierarchy/NonLocalProblem.cpp 
        Hierarchy/NonLocalProblem.h 
        Hierarchy/LocalProblem.cpp 
        Hierarchy/LocalProblem.h
        Hierarchy/MPICommunicator.cpp
        Hierarchy/MPICommunicator.h
        Hierarchy/DofIndexData.h
        Hierarchy/DofIndexData.cpp
        Hierarchy/LevelDofIndexData.h
        Hierarchy/LevelDofIndexData.cpp
        )

DEAL_II_SETUP_TARGET(Code)
	
