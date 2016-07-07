/*
 * GradientTable.h
 *
 *  Created on: 03.02.2016
 *      Author: ae14
 */

#ifndef GRADIENTTABLE_H_
#define GRADIENTTABLE_H_

class GradientTable {

private:

	dealii::Vector<double> steps;
	dealii::Vector<double> qualities;
	dealii::Vector<double> grad_step;
	dealii::Vector<double> ref_configuration;
	dealii::Vector<double> last_configuration;

	double final_quality;
	double initial_quality;
	double last_quality;
public:
	GradientTable( unsigned int in_step , dealii::Vector<double> in_configuration, double in_quality, dealii::Vector<double> in_last_configuration, double in_last_quality);

    ~GradientTable ();
        
	const int ndofs;

	const int nfreedofs;

	const unsigned int GlobalStep;

	void SetInitialQuality(double in_quality);

	void AddComputationResult(int in_component, double in_step, double in_quality);

	void AddFullStepResult(dealii::Vector<double> in_step, double in_quality);

	void PrintFullLine();

	void PrintTable();

	void WriteTableToFile(std::string in_filename);
};

#endif /* GRADIENTTABLE_H_ */
