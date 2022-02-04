#pragma once

enum SweepingDirection {
  X = 0, Y = 1, Z = 2
};

enum DofType {
  EDGE, SURFACE, RAY, IFFa, IFFb, SEGMENTa, SEGMENTb
};

enum Direction {
  MinusX = 0, PlusX = 1, MinusY = 2, PlusY = 3, MinusZ = 4, PlusZ = 5
};

enum ConnectorType {
  Circle, Rectangle
};

enum BoundaryConditionType {
  PML, HSIE
};

enum Evaluation_Domain {
  CIRCLE_CLOSE, CIRCLE_MAX, RECTANGLE_INNER
};

enum SurfaceType {
  OPEN_SURFACE, NEIGHBOR_SURFACE, ABC_SURFACE, DIRICHLET_SURFACE
};

enum Evaluation_Metric {
  FUNDAMENTAL_MODE_EXCITATION, POYNTING_TYPE_ENERGY
};

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

enum OptimizationSchema {
  FD, Adjoint
};

enum SolverOptions {
  GMRES, MINRES, UMFPACK
};

enum PreconditionerOptions {
  Sweeping, FastSweeping, HSIESweeping, HSIEFastSweeping
};

enum SteppingMethod {
  Steepest, CG, LineSearch
};

enum TransformationType {
  InhomogenousWavegeuideTransformationType, AngleWaveguideTransformationType, BendTransformationType
};
