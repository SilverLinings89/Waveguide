#pragma once

/**
 * @file GlobalObjects.h
 * @author Pascal Kraft
 * @brief Contains the declaration of some global objects that contain the parameter values as well as some values derived from them, like the geometry and information about other processes.
 * @version 0.1
 * @date 2022-03-22
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "../Helpers/Parameters.h"
#include "GeometryManager.h"
#include "../Hierarchy/MPICommunicator.h"
#include "ModeManager.h"
#include "OutputManager.h"
#include "TimerManager.h"
#include "../SpaceTransformations/SpaceTransformation.h"

/** @brief All parameters of the run (run file, case file and overrides) and values derived from them, for example the block indices of this process. */
extern Parameters GlobalParams; 
/** @brief Coordinate ranges of the local and the global domain and the geometry and dof data of all levels. */
extern GeometryManager Geometry;
/** @brief MPI communicators of the levels of the sweeping hierarchy and the rank of this process in each of them. */
extern MPICommunicator GlobalMPI;
/** @brief Analytic modes of a circular waveguide. Legacy object: it is loaded at startup but its modes are not used. */
extern ModeManager GlobalModeManager;
/** @brief Output folder of this run, file names in it and the log file of this process. */
extern OutputManager GlobalOutputManager;
/** @brief Timers of the sections of the code (assembly, solve, output), one set per level. */
extern TimerManager GlobalTimerManager;
/** @brief The coordinate transformation between the computational (math) and the physical (phys) coordinates. The derived type is selected by the parameter transformation_type. */
extern SpaceTransformation * GlobalSpaceTransformation;

/**
 * @brief Reads the parameters and initializes all global objects declared in this file.
 *
 * The order is: parameters, level communicators, geometry (all levels), mode manager, output folder, timers and
 * the space transformation. The program exits if the override data cannot be parsed.
 *
 * @param run_file Name of the parameter file that describes the run (solver, parallelization, output).
 * @param case_file Name of the parameter file that describes the case (geometry, material, signal).
 * @param override_data Parameter values that replace those of the two files, as key-value pairs separated by ";", for example "pml_order=4;pml_sigma_max=10". Empty for no overrides.
 */
void initialize_global_variables(const std::string run_file,const std::string case_file, std::string override_data = "");
