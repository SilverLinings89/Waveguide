
#ifndef ParametersFlag
#define ParametersFlag

#include <string>
#include <mpi.h>
#include "ShapeDescription.h"
/**
 * \class Parameters
 * \brief This structure contains all information contained in the input file and some values that can simply be computed from it.
 *
 * In the application, static Variable of this type makes the input parameters available globally.
 * \author: Pascal Kraft
 * \date: 28.11.2016
 */

enum ConnectorType{
    Circle, Rectangle
};

enum BoundaryConditionType{
    PML, HSIE
};

enum SpecialCase{
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

enum OptimizationSchema{
    FD, Adjoint
};

enum SolverOptions {
    GMRES, MINRES, UMFPACK
};

const static std::string PrecOptionNames[] = {"Sweeping","FastSweeping", "HSIESweeping", "HSIEFastSweeping"};

enum PreconditionerOptions {
  Sweeping,FastSweeping,HSIESweeping,HSIEFastSweeping
};

enum SteppingMethod {
    Steepest, CG, LineSearch
};

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
    
    bool O_C_D_ConvergenceAll ;
    
    bool O_C_D_ConvergenceFirst ;
    
    bool O_C_D_ConvergenceLast ;
    
    bool O_C_P_ConvergenceAll ;
    
    bool O_C_P_ConvergenceFirst ;
    
    bool O_C_P_ConvergenceLast ;
    
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
    
    BoundaryConditionType M_BC_Type;
    
    double M_BC_Zminus;
    
    double M_BC_Zplus;
    
    double M_BC_XMinus;
    
    double M_BC_XPlus;
    
    double M_BC_YMinus;
    
    double M_BC_YPlus;
    
    double M_BC_KappaXMax;
    
    double M_BC_KappaZMax;
    
    double M_BC_KappaYMax;
    
    double M_BC_SigmaXMax;
    
    double M_BC_SigmaZMax;
    
    double M_BC_SigmaYMax;
    
    double M_BC_DampeningExponent;
    
    bool Sc_Homogeneity;
    
    OptimizationSchema Sc_Schema;
    
    int Sc_OptimizationSteps;
    
    SteppingMethod Sc_SteppingMethod;

    // StepWidth Sc_StepWidth;

    SolverOptions So_Solver;
    
    PreconditionerOptions So_Preconditioner;

    double So_PreconditionerDampening;

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
    
    double LayerThickness;
    
    double SectorThickness;

    unsigned int	MPI_Rank;

    MPI_Comm	MPIC_World;

    bool PMLLayer;

    double LayersPerSector;

    int NumberProcesses;

    bool Head = false;

    double SystemLength;

    double Maximum_Z;
    double Minimum_Z;
    // unsigned int MPI_Size;

    double Phys_V;
    double Phys_SpotRadius;

    double StepWidth;

    bool M_PC_Use;

    int M_PC_Case;

    ShapeDescription sd;

};

#endif
