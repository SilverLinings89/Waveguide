
#ifndef ParametersFlag
#define ParametersFlag

#include <mpi.h>
#include <string>
#include "ShapeDescription.h"
#include "Enums.h"

/**
 * \class Parameters
 * \brief This structure contains all information contained in the input file
 * and some values that can simply be computed from it.
 *
 * In the application, static Variable of this type makes the input parameters
 * available globally. \author: Pascal Kraft \date: 28.11.2016
 */

const static std::string PrecOptionNames[] = {
    "Sweeping", "FastSweeping", "HSIESweeping", "HSIEFastSweeping"};

struct Parameters {
  bool O_O_G_HistoryLive;

  bool O_O_G_HistoryShapes;

  bool O_O_G_History;

  bool O_O_V_T_TransformationWeightsAll;

  bool O_O_V_T_TransformationWeightsFirst;

  bool O_O_V_T_TransformationWeightsLast;

  bool O_O_V_S_SolutionAll;

  bool O_O_V_S_SolutionFirst;

  bool O_O_V_S_SolutionLast;

  bool O_C_D_ConvergenceAll;

  bool O_C_D_ConvergenceFirst;

  bool O_C_D_ConvergenceLast;

  bool O_C_P_ConvergenceAll;

  bool O_C_P_ConvergenceFirst;

  bool O_C_P_ConvergenceLast;

  bool O_G_Summary;

  bool O_G_Log;

  ConnectorType M_C_Shape;

  double M_C_Dim1In;

  double M_C_Dim2In;

  double M_C_Dim1Out;

  double M_C_Dim2Out;

  double M_R_XLength;

  double M_R_YLength;

  double M_R_ZLength;

  double M_W_Delta;

  double M_W_epsilonin;

  double M_W_epsilonout;

  int M_W_Sectors;

  double M_W_Lambda;

  bool Sc_Homogeneity;

  OptimizationSchema Sc_Schema;

  int Sc_OptimizationSteps;

  SteppingMethod Sc_SteppingMethod;

  // StepWidth Sc_StepWidth;

  int So_ElementOrder;

  int So_RestartSteps;

  int So_TotalSteps;

  double So_Precision;

  bool C_AllOne;

  double C_Mu;

  double C_Epsilon;

  double C_Pi;

  double C_c;

  double C_f0;

  double C_k0;

  double C_omega;

  int R_Global;

  int R_Local;

  int R_Interior;

  double SectorThickness;

  unsigned int MPI_Rank;

  unsigned int NumberProcesses;

  // unsigned int MPI_Size;

  double Phys_V;
  double Phys_SpotRadius;

  double StepWidth;

  bool M_PC_Use;

  int M_PC_Case;

  ShapeDescription sd;

  unsigned int Blocks_in_z_direction;

  unsigned int Blocks_in_x_direction;

  unsigned int Blocks_in_y_direction;

  unsigned int Index_in_x_direction;

  unsigned int Index_in_y_direction;

  unsigned int Index_in_z_direction;

  int current_run_number;

  unsigned int
      Coupling_Interface_Z_Block_Index;  // The jump interface is at the lower
                                         // z-surface of the block with this
                                         // Index_in_z_direction

  unsigned int HSIE_SWEEPING_LEVEL =
      1;  // 1 means normal sweeping, 2 means hierarchical sweeping with depth
          // 1, 3 means hierarchical sweeping with depth 2.
};

#endif
