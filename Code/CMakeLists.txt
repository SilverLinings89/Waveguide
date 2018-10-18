INCLUDE_DIRECTORIES(Core, Helpers, MeshGenerators, OptimizationAlgorithm, OptimizationStrategies, SpaceTransformations)

ADD_LIBRARY(Code
	Core/PreconditionerSweeping.cpp
	Core/Sector.cpp
	Core/SolutionWeight.cpp
	Core/Waveguide.cpp
	Helpers/ExactSolution.cpp
	Helpers/ModeManager.cpp
	Helpers/ParameterReader.cpp
	Helpers/Parameters.cpp
	Helpers/PointVal.cpp
	Helpers/QuadratureFormulaCircle.cpp
	Helpers/staticfunctions.cpp
	Helpers/ShapeDescription.cpp
	MeshGenerators/MeshGenerator.cpp
	MeshGenerators/RoundMeshGenerator.cpp
	MeshGenerators/SquareMeshGenerator.cpp
	OptimizationAlgorithm/Optimization1D.cpp
	OptimizationAlgorithm/OptimizationAlgorithm.cpp
	OptimizationAlgorithm/OptimizationCG.cpp
	OptimizationAlgorithm/OptimizationSteepestDescent.cpp
	OptimizationStrategies/AdjointOptimization.cpp
	OptimizationStrategies/FDOptimization.cpp
	OptimizationStrategies/Optimization.cpp
	SpaceTransformations/DualProblemTransformationWrapper.cpp
	SpaceTransformations/HomogenousTransformationCircular.cpp
	SpaceTransformations/HomogenousTransformationRectangular.cpp
	SpaceTransformations/InhomogenousTransformationCircular.cpp
	SpaceTransformations/InhomogenousTransformationRectangular.cpp
	SpaceTransformations/SpaceTransformation.cpp
	HSIE/FaceSurfaceComparator.cpp
	HSIE/FaceSurfaceComparatorZ.cpp
	HSIE/HSIE_Preconditioner_Base.cpp
	HSIE/HSIEDofType.cpp
	)
	
DEAL_II_SETUP_TARGET(Code)
	
