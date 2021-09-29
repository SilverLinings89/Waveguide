INCLUDE_DIRECTORIES(Core, Helpers, MeshGenerators, OptimizationAlgorithm, OptimizationStrategies, SpaceTransformations, HSIEPreconditioner)

ADD_LIBRARY(Code
        Core/Sector.cpp
        Core/FEDomain.cpp
        Core/SolutionWeight.cpp
        Core/InnerDomain.cpp
        Core/Types.h
        Runners/Simulation.cpp
        Runners/ParameterSweep.cpp
        Runners/SingleCoreRun.cpp
        Runners/SweepingRun.cpp
        Helpers/ParameterReader.cpp
        Helpers/PointSourceField.cpp
        Helpers/Parameters.cpp
        Helpers/PointVal.cpp
        Helpers/QuadratureFormulaCircle.cpp
        Helpers/staticfunctions.cpp
        Helpers/ShapeDescription.cpp
        Solutions/ExactSolution.cpp
        Solutions/ExactSolutionRamped.cpp
        Solutions/ExactSolutionConjugate.cpp
        MeshGenerators/SquareMeshGenerator.cpp
        SpaceTransformations/DualProblemTransformationWrapper.cpp
        SpaceTransformations/HomogenousTransformationRectangular.cpp
        SpaceTransformations/InhomogenousTransformationRectangular.cpp
        SpaceTransformations/SpaceTransformation.cpp
        BoundaryCondition/JacobianForCell.cpp
        BoundaryCondition/DofData.cpp
        BoundaryCondition/BoundaryCondition.cpp
        BoundaryCondition/PMLSurface.cpp
        BoundaryCondition/PMLMeshTransformation.cpp
        BoundaryCondition/HSIEPolynomial.cpp
        BoundaryCondition/HSIESurface.cpp
        BoundaryCondition/NeighborSurface.cpp
        BoundaryCondition/EmptySurface.cpp
        BoundaryCondition/DirichletSurface.cpp
        BoundaryCondition/LaguerreFunction.cpp
        BoundaryCondition/JacobianForCell.cpp
        Hierarchy/HierarchicalProblem.cpp  
        Hierarchy/NonLocalProblem.cpp  
        Hierarchy/LocalProblem.cpp
        Hierarchy/MPICommunicator.cpp
        Hierarchy/DofIndexData.cpp
        Hierarchy/LevelDofIndexData.cpp
        ModalComputations/RectangularMode.cpp
        GlobalObjects/GeometryManager.cpp
        GlobalObjects/TimerManager.cpp
        GlobalObjects/GlobalObjects.cpp
        GlobalObjects/ModeManager.cpp
        GlobalObjects/OutputManager.cpp
        )

DEAL_II_SETUP_TARGET(Code)
	
# Get the current working branch
execute_process(
    COMMAND git rev-parse --abbrev-ref HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_BRANCH
    OUTPUT_STRIP_TRAILING_WHITESPACE)

# Get the latest commit hash
execute_process(
    COMMAND git rev-parse HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_COMMIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE)

target_compile_definitions(Code PRIVATE
    "-DGIT_COMMIT_HASH=\"${GIT_COMMIT_HASH}\"")
