#pragma once

#include "../Code/BoundaryCondition/LaguerreFunction.h"
#include "../third_party/googletest/googletest/include/gtest/gtest.h"
#include "../Code/BoundaryCondition/BoundaryCondition.h"
#include "../Code/BoundaryCondition/HSIESurface.h"
#include "../Code/BoundaryCondition/HSIEPolynomial.h"
#include "../Code/BoundaryCondition/PMLSurface.h"
#include "../Code/Core/Types.h"
#include "../Code/Helpers/staticfunctions.h"
#include <deal.II/grid/grid_generator.h>
#include <deal.II/base/tensor.h>
#include <deal.II/grid/grid_tools.h>
#include "./TestHelperFunctions.h"

class PMLCubeFixture: public ::testing::Test {
public:
 
protected:
  void SetUp() override {
    setup_GlobalParams_for_tests(5,5,5,BoundaryConditionType::PML,1,1,1);
    Geometry.initialize();
  }
};
