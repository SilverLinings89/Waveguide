/**
 * Die Parameter-Reader Klasse
 * Diese Klasse liest das Input-File und Parsed dessen Werte. Wenn die Werte fehlerhaft sind, werden Standard-Werte verwendet. Für einige Werte ist auch ein Bereich vorgegeben (zum Beispiel kann die Länge des Rechengebiets nicht negativ sein.)
 * Der Vorteil dieses Mechanismus ist, dass nicht für jeden Durchlauf das Programm kompiliert werden muss, sondern lediglich ein Input-File (mit einer GUI) bearbeitet wird.
 * Mit geschickter Implementierung (Templating) kann man es auch erreichen, dass man zwischen Lösern und Vorkonditionierern aus verschiedenen Bibliotheken umschalten kann - also Beispielsweise ein Programm hat, das entweder mit Trilinos-Datentypen und Lösern arbeitet oder mit den PETSc-Äquivalenten.
 * @author: Pascal Kraft
 * @date: 07.09.2015
 */
#ifndef ParameterReaderFlag
#define ParameterReaderFlag

#include <deal.II/base/parameter_handler.h>

using namespace dealii;

class ParameterReader : public Subscriptor
{
public:

	ParameterReader			(ParameterHandler &prmhandler);
	void read_parameters	(const std::string inputfile);
	void declare_parameters	();

private:
	ParameterHandler &prm;

};

#endif
