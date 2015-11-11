/**
 * Die Optimization-Klasse
 * Diese Klasse verwaltet ein Wellenleiter-Objekt und ein WellenleiterStruktur-Objekt. Ihre run() Methode beschreibt den Ablauf des Optimierungs-Verfahrens. Dabei werden regelmäßig die beiden Objekte bearbeitet.
 * In ihrem Konstruktor fordert sie ein Parameters-Objekt, das alle Informationen aus dem Input-File enthält, sowie eine Referenz auf den Wellenleiter für die Berechnung der Lösungen sowie eine Referenz auf eine Wellenleiter-Form, die die Materialtensoren liefert und die Form-Parameter verwaltet.
 * @author: Pascal Kraft
 * @date: 07.09.2015
 */

#ifndef OptimizationFlag
#define OptimizationFlag

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>

#include "WaveguideStructure.h"
#include "Waveguide.h"
#include "Parameters.h"

using namespace dealii;

class Optimization {
	public:
		const int dofs; // (sectors +1) *3 -6
		const Parameters System_Parameters;
		Waveguide<SparseMatrix<double>, Vector<double> > &waveguide;
		WaveguideStructure &structure;

		Optimization( Parameters , Waveguide<SparseMatrix<double>, Vector<double> >  & , WaveguideStructure &);
		void run();

};

#endif
