INCLUDE_DIRECTORIES(
	${CMAKE_SOURCE_DIR}/Code/Core, 
	${CMAKE_SOURCE_DIR}/Code/Helpers, 
	${CMAKE_SOURCE_DIR}/Code/MeshGenerators, 
	${CMAKE_SOURCE_DIR}/Code/OptimizationAlgorithm, 
	${CMAKE_SOURCE_DIR}/Code/OptimizationStrategies, 
	${CMAKE_SOURCE_DIR}/Code/SpaceTransformations
)

ADD_EXECUTABLE(main main.cpp)

DEAL_II_SETUP_TARGET(main)

TARGET_LINK_LIBRARIES(main Code)
