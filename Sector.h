/**
 * Die Sector-Klasse
 * Diese Klasse implementiert einen einzigen Sektor. Er hat seine Freiheitsgrade auf dem linken und rechten Rand und bietet die Materialeigenschaften als Ergebnis.
 *
 * Funktion: TransformationTensorInternal
 * Sollte nur von WaveGuideStructure gerufen werden und liefert den TransformationsTensor am angegebenen Punkt.
 *
 * Funktion: set_properties
 * Setzt einen unbeschränkten Formfreiheitsgrad auf einen Wert.
 *
 * Funktion: set_properties_force
 * Setzt auch beschränkte Freiheitsgrade auf einen Wert. Die Methode wird verwendet um für Sektoren, die am Rand liegen, die entsprechenden Werte einmalig zu setzen. In der Methode ohne force können diese Freiheitsgrade nicht manipuliert werden.
 *
 * Funktion: getQi
 * Diese Methode berechnet den Wert für das entsprechende Q an der gegebenen z-Komponente. Q3 ist nicht implementiert und Q2 und Q1 hängen jeweils nur von z ab.
 *
 * @author: Pascal Kraft
 * @date: 07.09.2015
 */
#ifndef SectorFlag
#define SectorFlag

#include <deal.II/base/tensor.h>

using namespace dealii;

class Sector {
	public:
		const bool left;
		const bool right;
		const bool boundary;
		double r_0, r_1, v_0, v_1, m_0, m_1;
		const double z_0, z_1;
		Tensor<2,3, double> TransformationTensorInternal (double in_x, double in_y, double in_z);
		Sector(bool, bool, double, double);
		void set_properties(double , double , double , double, double, double);
		void set_properties_force(double , double , double , double, double, double);
		double getQ1( double);
		double getQ2( double);
		double getQ3( double);
};

#endif
