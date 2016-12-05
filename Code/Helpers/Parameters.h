
#ifndef ParametersFlag
#define ParametersFlag

/**
 * \class Parameters
 * \brief This structure contains all information contained in the input file and some values that can simply be computed from it.
 *
 * In the application, static Variable of this type makes the input parameters available globally.
 * \author: Pascal Kraft
 * \date: 28.11.2016
 */

enum ConnectorType{
    circle, rectangle
};

enum BoundaryConditionType{
    PML, HSIE
};

enum OptimizationSchema{
    FD, Adjoint
};

enum SolverOptions {
    GMRES, MINRES, UMFPACK
};

enum PreconditionerOptions {
Sweeping,Amesos_Lapack,Amesos_Scalapack,Amesos_Klu,Amesos_Umfpack,Amesos_Pardiso,Amesos_Taucs,Amesos_Superlu,Amesos_Superludist,Amesos_Dscpack,Amesos_Mumps
}

struct Parameters {
    
    bool O_O_G_HistoryLive;
    
    bool O_O_G_HistoryShapes;
    
    bool O_O_G_HistoryLive;
    
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
    
    int M_BC_Zplus;
    
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
    
    SolverOptions So_Solver;
    
    int So_RestartSteps;
    
    int So_TotalSteps;
    
    double So_Precision;
    
    bool C_AllOne;
    
    double C_Mu;
    
    double C_Epsilon;
    
    double C_Pi;
    
    int R_Global;
    
    int R_Local;
    
    int R_Interior;
    
    
    
	unsigned int	MPI_Rank;

	MPI_Comm	MPI_Communicator;

	unsigned int MPI_Size;

};

#endif
