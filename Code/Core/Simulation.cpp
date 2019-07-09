/*
 * Simulation.cpp
 *
 *  Created on: Jun 24, 2019
 *      Author: kraft
 */

#include "Simulation.h"
#include "../Helpers/staticfunctions.h"
#include <deal.II/base/mpi.h>
#include <deal.II/base/logstream.h>

Simulation::Simulation() = default;

Simulation::~Simulation() = default;

void Simulation::Run() {
    CreateOutputDirectory();
    LoadParameters();
    PrepareGeometry();
    PrepareTransformedGeometry();
    InitializeMainProblem();
    InitializeAuxiliaryProblem();
}

void Simulation::LoadParameters() {

}

void Simulation::CreateOutputDirectory() {
    char *pPath;
    pPath = getenv("WORK");
    bool seperate_solutions = (pPath != nullptr);
    if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) {
        dealii::deallog.depth_console(10);
    } else {
        deallog.depth_console(0);
    }
    int i = 0;
    bool dir_exists = true;
    while (dir_exists) {
        std::stringstream out;
        if (seperate_solutions) {
            out << pPath << "/";
        }
        out << "Solutions/run";
        out << i;
        solutionpath = out.str();
        struct stat myStat;
        const char *myDir = solutionpath.c_str();
        if ((stat(myDir, &myStat) == 0) &&
            (((myStat.st_mode) & S_IFMT) == S_IFDIR)) {
            i++;
        } else {
            dir_exists = false;
        }
    }
    i = Utilities::MPI::max(i, MPI_COMM_WORLD);
    std::stringstream out;
    if (seperate_solutions) {
        out << pPath << "/";
    }
    out << "Solutions/run";

    out << i;
    solutionpath = out.str();
    mkdir(solutionpath.c_str(), ACCESSPERMS);

    log_stream.open(
            solutionpath + "/main" +
            std::to_string(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)) +
            ".log",
            std::ios::binary);

    int create_link = symlink(solutionpath.c_str(), "./latest");
    if (create_link == 0) {
        deallog << "Symlink latest created." << std::endl;
    } else {
        deallog << "Symlink latest creation failed." << std::endl;
    }

    deallog.attach(log_stream);
}

void Simulation::PrepareGeometry() { geometry.initialize(parameters); }

void Simulation::PrepareTransformedGeometry() {}

void Simulation::InitializeMainProblem() {}

void Simulation::InitializeAuxiliaryProblem() {}
