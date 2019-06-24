/*
 * Simulation.cpp
 *
 *  Created on: Jun 24, 2019
 *      Author: kraft
 */

#include "Simulation.h"

Simulation::Simulation() {
  // TODO Auto-generated constructor stub
}

Simulation::~Simulation() {
  // TODO Auto-generated destructor stub
}

void Simulation::Run() {
  LoadParameters();
  PrepareGeometry();
  PrepareTransformedGeometry();
  InitializeMainProblem();
  InitializeAuxiliaryProblem();
}

void Simulation::LoadParameters() {}

void Simulation::PrepareGeometry() { geometry.initialize(parameters); }

void Simulation::PrepareTransformedGeometry() {}

void Simulation::InitializeMainProblem() {}

void Simulation::InitializeAuxiliaryProblem() {}
