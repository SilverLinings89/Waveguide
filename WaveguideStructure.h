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

#include "Parameters.h"
#include "Sector.h"

using namespace dealii;

class WaveguideStructure {
	public:
		std::vector<Sector> case_sectors;
		const double epsilon_K, epsilon_M;
		const int sectors;
		const double sector_z_length;
		const double z_min, z_max;
		const double deltaY;
		const double r_0, r_1;
		const double v_0, v_1;
		const double m_0, m_1;
		const Parameters parameters;
		double highest, lowest;

		WaveguideStructure (Parameters &);
		Tensor<2,3, double> TransformationTensor (double in_x, double in_y, double in_z);
		void 	run() ;
		void 	estimate_and_initialize();
		double 	m(double);
		double 	v(double);
		double 	getQ1 ( double);
		double 	getQ2 ( double);
		double 	get_dof (int );
		void	set_dof (int , double );

};

#endif
