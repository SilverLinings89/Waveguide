/**
 * Die Waveguide-Structure-Klasse
 * In dieser Klasse werden die Form-Parameter gespeichert. Sie hat ein Array von Membern vom Typ Sector, die jeweils einen Sektor darstellen.
 *
 * Funktion: TransformationTensor
 * Diese Methode reskaliert die z-Komponente der Eingabe auf das Intervall [0,1] im betroffenen Sektor. Fragt man also zum Beispiel die z-Komponente 0,3 an, die genau in der Mitte eines Sektors wird, übersetzt diese Methode dies zu 0,5. Danach ruft sie die Transformation_Tensor Methode des richtigen Sektors auf. Dort erfolgt die eigentliche Berechnung.
 *
 * Funktion: Estimate_and_initialize()
 * Diese Methode schätzt für vorgegebenes r_0, r_1, m_0, m_1, v_0 und v_1 und eine Anzahl von Sektoren eine sinnvolle Startbelegung der Freiheitsgrade auf Basis eines Polynoms ähnlich dem in der Arbeit.
 *
 * Funktion: getQi()
 * diese Methode funktioniert äquivalent zu TransformationTensor nur für die Berechnung der Q_i. Es sind nur Q_1 und Q_2 implementiert, weil diese die einzigen sind, die auf dem Rand des Rechengebiets einen Wert haben (Q_3 ist dort 1). Q_3 ist zudem ein aufwändigerer Term und nicht erforderlich beim aktuellen Stand.
 *
 * Funktion: getDof()
 * Fordert den Wert eines Form-Freiheitsgrades an. Diese sind primär nach z-Koordinate und dann nach Alphabet geordnet, also m_1, r_1,v_1, m_2, r_2, v_2 ...
 *
 * Funktion: setDof()
 * Setzt den Wert eines Freiheitsgrades und prüft dabei die Zulässigkeit (kann keine Randwerte anpassen, weil das Konstanten des Systems sind.)
 * @author: Pascal Kraft
 * @date: 07.09.2015
 */
#ifndef WaveguideStructureFlag
#define WaveguideStructureFlag

#include <math.h>
#include <vector>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/point.h>

#include "../Helpers/Parameters.h"
#include "Sector.h"

using namespace dealii;

/**
 * \class WaveguideStructure
 * \brief Not sure if I still need this.
 * During the whole optimization-process only one object of this type is used and it continually gets update to reflect the most recent update information.
 *
 * \author Pascal Kraft
 * \date 23.11.2015
 */
class WaveguideStructure {
	public:

		/**
		 * This member contains all the Sectors who, as a sum, form the complete Waveguide. These Sectors are a partition of the simulated domain.
		 */
		std::vector<Sector> case_sectors;
		/**
		 * The material-property \f$\epsilon_r\f$ has a different value inside and outside of the waveguides core. This variable stores its value inside the core.
		 */
		const double epsilon_K;
		/**
		 *  The material-property \f$\epsilon_r\f$ has a different value inside and outside of the waveguides core. This variable stores its value outside the core.
		 */
		const double epsilon_M;
		/**
		 * Since the computational domain is split into subdomains (called sectors), it is important to keep track of the amount of subdomains. This member stores the number of Sectors the computational domain has been split into.
		 */
		const int sectors;

		/**
		 * This value is initialized with the value Delta from the input-file.
		 */
		const double deltaY;

		/**
		 * The radius of the input waveguide connector. This variable gets initialized with the value from the input-file.
		 */
		const double r_0;

		/**
		 * The radius of the output waveguide connector. This variable gets initialized with the value from the input-file.
		 */
		const double r_1;

		/**
		 * The tilt of the input waveguide connector towards the \f$ z \f$-axis. This variable gets initialized with the value from the input-file.
		 */
		const double v_0;

		/**
		 * The tilt of the output waveguide connector towards the \f$ z \f$-axis. This variable gets initialized with the value from the input-file.
		 */
		const double v_1;

		/**
		 * The distance of the waveguide-center from the \f$z\f$-axis in the \f$xy\f$-plane at the input-end of the computational domain.
		 */
		const double m_0;
		/**
		 * The distance of the waveguide-center from the \f$z\f$-axis in the \f$xy\f$-plane at the output-end of the computational domain.
		 */
		const double m_1;

		/**
		 * A parameters-structure which upon construction contains all values from the input file and hides the persistence-layer associated with it.
		 */
		const Parameters parameters;

		/**
		 * This member stores the highest signal quality achieved with any shape in this optimization-run.
		 */
		double highest;

		/**
		 * This member stores the lowest signal quality achieved with any shape in this optimization-run.
		 */
		double lowest;

		/**
		 * Since accessing the Parameters-file is an IO-operation and therefore needlessly slow, this constructor takes an argument of the ParameterHandler Type, which has the data preloaded from another context. It then proceeds to initialize all the constants to their value, which is derived from the ParameterHandler.
		 */
		WaveguideStructure (const Parameters &);

		/**
		 * Destructor of this class.
		 */
		~WaveguideStructure();
                
		/**
		 * This member encapsulates the main functionality of objects of this class: It calculates the most important structural information required in this code - the Material-Tensor. In fact, this function determines, in which sector the passed position lies and calls the appropriate function on that Sector-object.
		 * \param in_x \f$x\f$-coordinate of the position for which to calculate the transformation tensor.
		 * \param in_y \f$y\f$-coordinate of the position for which to calculate the transformation tensor.
		 * \param in_z \f$z\f$-coordinate of the position for which to calculate the transformation tensor.
		 */
		Tensor<2,3, double> TransformationTensor (double in_x, double in_y, double in_z);

		/**
		 * At the beginning (before the first solution of a system) only the boundary conditions for the shape of the waveguide are known. Therefore the values for the degrees of freedom need to be estimated. This function sets all variables to appropiate values and estimates an appropriate shape based on averages and a polynomial interpolation of the boundary conditions on the shape.
		 */
		void 	estimate_and_initialize();

		/**
		 * In order to be able to estimate the shape-parameters in the interior of the domain, a function for the interpolation of waveguide-centers distance to the \f$z\f$-axis is required.
		 * \param z The \f$z\f$-coordinate for which to estimate \f$m\f$.
		 */
		double 	estimate_m(double z);

		/**
		 * This function is used to interpolate the tilt of the Waveguide in the interior of the computational domain based on the boundary (connector) data provided in the input file. For the interpolation, a third order polynomial is used which satisfies the condition of having the derivative zero at both, the input and output connector and yielding a shape, that continuously connects the two interfaces.
		 * \param z The \f$z\f$-coordinate for which to interpolate \f$v\f$.
		 */
		double 	estimate_v(double z);

		/**
		 * This member calculates the value of Q1 for a provided \f$z\f$-coordinate. This value is used in the transformation of the solution-vector in transformed coordinates (solution of the system-matrix) to real coordinates (physical field).
		 * \param z The value of Q1 is independent of \f$x\f$ and \f$y\f$. Therefore only a \f$z\f$-coordinate is provided in a call to the function.
		 */
		double 	getQ1 ( double z);

		/**
		 * This member calculates the value of Q2 for a provided \f$z\f$-coordinate. This value is used in the transformation of the solution-vector in transformed coordinates (solution of the system-matrix) to real coordinates (physical field).
		 * \param z The value of Q2 is independent of \f$x\f$ and \f$y\f$. Therefore only a \f$z\f$-coordinate is provided in a call to the function.
		 */
		double 	getQ2 ( double z);

		/**
		 * This is a getter for the values of degrees of freedom. A getter-setter interface was introduced since the values are estimated automatically during the optimization and non-physical systems should be excluded from the domain of possible cases.
		 * \param dof The index of the degree of freedom to be retrieved from the structure of the modelled waveguide.
		 * \return This function returns the value of the requested degree of freedom. Should this dof not exist, 0 will be returnd.
		 */
		double 	get_dof (int dof, bool free);

		/**
		 * This function sets the value of the dof provided to the given value. It is important to consider, that some dofs are non-writable (i.e. the values of the degrees of freedom on the boundary, like the radius of the input-connector cannot be changed).
		 * \param dof The index of the parameter to be changed.
		 * \param value The value, the dof should be set to.
		 */
		void	set_dof (int dof , double value, bool free );

		/**
		 * Using this method unifies the usage of coordinates. This function takes a global \f$z\f$ coordinate (in the computational domain) and returns both a Sector-Index and an internal \f$z\f$ coordinate indicating which sector this coordinate belongs to and how far along in the sector it is located.
		 * \param double in_z global system \f$z\f$ coordinate for the transformation.
		 */
		std::pair<int, double> Z_to_Sector_and_local_z(double in_z);

		/**
		 * Returns the complete length of the computational domain.
		 */
		double System_Length();

		/**
		 * Returns the length of one sector
		 */
		double Sector_Length();

		/**
		 * Returns the length of one layer
		 */
		double Layer_Length();

		/**
		 * Returns the radius for a system-coordinate;
		 */
		double get_r(double in_z);

		/**
		 * Returns the shift for a system-coordinate;
		 */
		double get_m(double in_z);

		/**
		 * Returns the tilt for a system-coordinate;
		 */
		double get_v(double in_z);

		/**
		 * This Method writes a comprehensive description of the current structure to the console.
		 */
		void WriteConfigurationToConsole();

		/**
		 * For a given value \f$ z \f$ this function returns the Layer in which points with this \f$ z \f$-coordinate would be located.
		 */
		int Z_to_Layer(double z);

		/**
		 * This vector of values saves the initial configuration
		 */
		Vector<double> InitialDofs;

		/**
		 * This vector of values saves the initial configuration
		 */
		double InitialQuality;

		/**
		 * Console output of the current Waveguide Structure.
		 */
		void Print();
};

#endif
