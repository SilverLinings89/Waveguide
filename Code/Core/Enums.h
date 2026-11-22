#pragma once

/**
 * @file Enums.h
 * @author Pascal Kraft
 * @brief All the enums used in this project.
 * @version 0.1
 * @date 2022-03-22
 * 
 * @copyright Copyright (c) 2022
 * 
 */

/**
 * @brief Coordinate direction in which a level of the sweeping preconditioner sweeps.
 *
 * The value is the index of the coordinate. The highest level always sweeps in Z. With two levels, level 1 sweeps in Y. With three levels, level 2 sweeps in Y and level 1 in X (see get_sweeping_direction_for_level()).
 */
enum SweepingDirection {
  X = 0, Y = 1, Z = 2
};

/**
 * @brief Types of degrees of freedom on a Hardy space infinite element surface (see DofData and HSIESurface).
 *
 * EDGE: Nedelec dof on an edge of the surface mesh. It is also a dof of the inner domain.
 * SURFACE: Nedelec dof in the interior of a surface cell. It is also a dof of the inner domain.
 * RAY: dof on an infinite ray that starts at a vertex of the surface mesh.
 * IFFa: dof on the infinite face above a surface edge, built from a Nedelec shape function on the edge.
 * IFFb: dof on the infinite face above a surface edge, built from a nodal (Q) shape function on the edge.
 * SEGMENTa: dof of the infinite segment above a surface cell, built from a Nedelec shape function in the cell.
 * SEGMENTb: dof of the infinite segment above a surface cell, built from a nodal (Q) shape function in the cell.
 */
enum DofType {
  EDGE, SURFACE, RAY, IFFa, IFFb, SEGMENTa, SEGMENTb
};

/**
 * @brief The six directions along the coordinate axes.
 *
 * The values are the boundary ids of the surfaces of a cuboid whose outward normal points in that direction.
 */
enum Direction {
  MinusX = 0, PlusX = 1, MinusY = 2, PlusY = 3, MinusZ = 4, PlusZ = 5
};

/** @brief Shape of the cross-section of the waveguide connectors. Not used in the current code. */
enum ConnectorType {
  Circle, Rectangle
};

/**
 * @brief The method used to truncate the domain at surfaces with absorbing boundary conditions.
 *
 * PML: perfectly matched layer. HSIE: Hardy space infinite elements.
 */
enum BoundaryConditionType {
  PML, HSIE
};

/** @brief Region of the cross-section over which the signal is evaluated. Not used in the current code. */
enum Evaluation_Domain {
  CIRCLE_CLOSE, CIRCLE_MAX, RECTANGLE_INNER
};

/**
 * @brief What is located at one of the six surfaces of the local domain on a sweeping level.
 *
 * OPEN_SURFACE: the tangential trace is set to zero (EmptySurface). Used on the lower surfaces in the sweeping directions of higher levels, where a neighbor exists that is not part of the problem on this level.
 * NEIGHBOR_SURFACE: interface to another process. The dofs on the lower surfaces (boundary ids 0, 2, 4) of this type are owned by the neighbor.
 * ABC_SURFACE: absorbing boundary condition (PML or HSIE).
 * DIRICHLET_SURFACE: Dirichlet values (DirichletSurface), used to prescribe the input signal.
 */
enum SurfaceType {
  OPEN_SURFACE, NEIGHBOR_SURFACE, ABC_SURFACE, DIRICHLET_SURFACE
};

/** @brief Quantity used to rate a waveguide shape. Not used in the current code. */
enum Evaluation_Metric {
  FUNDAMENTAL_MODE_EXCITATION, POYNTING_TYPE_ENERGY
};

/** @brief Names of predefined reference geometries. Not used in the current code. */
enum SpecialCase {
  none,
  reference_bond_nr_0,
  reference_bond_nr_1,
  reference_bond_nr_2,
  reference_bond_nr_40,
  reference_bond_nr_41,
  reference_bond_nr_42,
  reference_bond_nr_43,
  reference_bond_nr_44,
  reference_bond_nr_45,
  reference_bond_nr_46,
  reference_bond_nr_47,
  reference_bond_nr_48,
  reference_bond_nr_49,
  reference_bond_nr_50,
  reference_bond_nr_51,
  reference_bond_nr_52,
  reference_bond_nr_53,
  reference_bond_nr_54,
  reference_bond_nr_55,
  reference_bond_nr_56,
  reference_bond_nr_57,
  reference_bond_nr_58,
  reference_bond_nr_59,
  reference_bond_nr_60,
  reference_bond_nr_61,
  reference_bond_nr_62,
  reference_bond_nr_63,
  reference_bond_nr_64,
  reference_bond_nr_65,
  reference_bond_nr_66,
  reference_bond_nr_67,
  reference_bond_nr_68,
  reference_bond_nr_69,
  reference_bond_nr_70,
  reference_bond_nr_71,
  reference_bond_nr_72
};

/**
 * @brief How the shape gradient is computed. Not used in the current code.
 *
 * FD: finite differences. Adjoint: adjoint method.
 */
enum OptimizationSchema {
  FD, Adjoint
};

/**
 * @brief The iterative solver used on the sweeping levels above level 0 (see NonLocalProblem).
 *
 * GMRES: flexible GMRES with right preconditioning. MINRES: minimal residual method. BICGS: BiCGStab. TFQMR: transpose-free QMR. PCONLY: Richardson iteration, i.e. repeated application of the preconditioner. S_CG: conjugate gradients.
 */
enum SolverOptions {
  GMRES, MINRES, BICGS, TFQMR, PCONLY, S_CG
};

/** @brief Variants of the sweeping preconditioner. Not used in the current code. */
enum PreconditionerOptions {
  Sweeping, FastSweeping, HSIESweeping, HSIEFastSweeping
};

/**
 * @brief Method used to compute a step of the shape optimization. Not used in the current code.
 *
 * Steepest: steepest descent. BFGS: BFGS method.
 */
enum SteppingMethod {
  Steepest, BFGS
};

/**
 * @brief Selects the space transformation that describes the shape of the waveguide (see GlobalSpaceTransformation).
 *
 * WavegeuideTransformationType: WaveguideTransformation. AngleWaveguideTransformationType: AngleWaveguideTransformation. BendTransformationType: BendTransformation. PredefinedShapeTransformationType: PredefinedShapeTransformation.
 */
enum TransformationType {
  WavegeuideTransformationType, AngleWaveguideTransformationType, BendTransformationType, PredefinedShapeTransformationType
};
