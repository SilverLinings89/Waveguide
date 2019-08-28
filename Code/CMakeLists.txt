INCLUDE_DIRECTORIES(Core, Helpers, MeshGenerators, OptimizationAlgorithm, OptimizationStrategies, SpaceTransformations, HSIEPreconditioner)

ADD_LIBRARY(Code
        Core/PreconditionerSweeping.cpp
        Core/Sector.cpp
        Core/SolutionWeight.cpp
        Core/NumericProblem.cpp
        Core/AuxiliaryProblem.cpp
        Core/MainProblem.cpp
        Core/Simulation.cpp
        Core/DOFManager.cpp
        Helpers/GeometryManager.cpp
        Helpers/ExactSolution.cpp
        Helpers/ModeManager.cpp
        Helpers/ParameterReader.cpp
        Helpers/Parameters.cpp
        Helpers/PointVal.cpp
        Helpers/QuadratureFormulaCircle.cpp
        Helpers/staticfunctions.cpp
        Helpers/ShapeDescription.cpp
        MeshGenerators/SquareMeshGenerator.cpp
        OptimizationAlgorithm/Optimization1D.cpp
        OptimizationAlgorithm/OptimizationAlgorithm.cpp
        OptimizationAlgorithm/OptimizationCG.cpp
        OptimizationAlgorithm/OptimizationSteepestDescent.cpp
        OptimizationStrategies/AdjointOptimization.cpp
        OptimizationStrategies/FDOptimization.cpp
        OptimizationStrategies/Optimization.cpp
        SpaceTransformations/DualProblemTransformationWrapper.cpp
        SpaceTransformations/HomogenousTransformationRectangular.cpp
        SpaceTransformations/InhomogenousTransformationRectangular.cpp
        SpaceTransformations/SpaceTransformation.cpp
        HSIEPreconditioner/HSIEPreconditioner.cpp
        HSIEPreconditioner/HSIESurface.cpp
        )

DEAL_II_SETUP_TARGET(Code)
	
