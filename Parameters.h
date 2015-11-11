/**
 * Die Parameters-Struktur
 * Diese Struktur enthält alle Variablen, die auch im Input-File zur Verfügung stehen, und noch ein paar mehr. Sie wird befüllt, wenn das File gelesen wird und statisch verfügbar gemacht.
 * Dies reduziert den Aufwand das Objekt immer wieder hin und her zureichen und ist insofern sinnvoll als dass sowieso die Werte konstant sind, weil es sich bei allen um System-Parameter handelt.
 * @author: Pascal Kraft
 * @date: 07.09.2015
 */
#ifndef ParametersFlag
#define ParametersFlag

struct Parameters {
	bool			PRM_O_Grid, PRM_O_Dofs, PRM_O_ActiveCells, PRM_O_VerboseOutput;
	std::string		PRM_M_C_TypeIn, PRM_M_C_TypeOut;
	double			PRM_M_C_RadiusIn, PRM_M_C_RadiusOut;
	int				PRM_M_R_XLength, PRM_M_R_YLength, PRM_M_R_ZLength;
	double			PRM_M_W_Delta, PRM_M_W_EpsilonIn, PRM_M_W_EpsilonOut, PRM_M_W_Lambda;
	std::string		PRM_M_BC_Type;
	double			PRM_M_BC_XYin, PRM_M_BC_XYout, PRM_M_BC_Mantle, PRM_M_BC_KappaXMax, PRM_M_BC_KappaYMax, PRM_M_BC_KappaZMax, PRM_M_BC_SigmaXMax, PRM_M_BC_SigmaYMax, PRM_M_BC_SigmaZMax;
	int				PRM_M_BC_M;
	std::string 	PRM_D_Refinement;
	int 			PRM_D_XY, PRM_D_Z;
	int 			PRM_S_Steps;
	int				PRM_S_GMRESSteps, PRM_S_PreconditionerBlockCount;
	int				PRM_A_Threads;
	double			PRM_S_Precision;
	std::string 	PRM_S_Solver, PRM_S_Preconditioner;
	int 			PRM_M_W_Sectors;
	double 			PRM_C_PI, PRM_C_Eps0, PRM_C_Mu0, PRM_C_c, PRM_C_f0, PRM_C_omega;
	int				PRM_Op_MaxCases;
	double			PRM_Op_InitialStepWidth;
	double			PRM_M_C_TiltIn,PRM_M_C_TiltOut ;
	double			PRM_C_k0, PRM_C_lambda;
};

#endif
