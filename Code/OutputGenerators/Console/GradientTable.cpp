/*
 * GradientTable.cpp
 *
 *  Created on: 03.02.2016
 *      Author: ae14
 */

#include "GradientTable.h"
#include <sys/ioctl.h>
#include <stdio.h>
#include <unistd.h>
#include <iomanip>

GradientTable::GradientTable(unsigned int in_step , dealii::Vector<double> in_configuration, double in_quality, dealii::Vector<double> in_last_configuration, double in_last_quality):
  ndofs(100),
  nfreedofs(100),
  GlobalStep(in_step)
{
	final_quality = 0.0;
	initial_quality = in_quality;
	last_quality = in_last_quality;
	steps.reinit(nfreedofs);
	qualities.reinit(nfreedofs);
	last_configuration.reinit(nfreedofs);
	ref_configuration.reinit(nfreedofs);
	grad_step.reinit(nfreedofs);
	for(int i = 0 ; i < nfreedofs; i++) {
		ref_configuration[i] = in_configuration[i];
		last_configuration[i] = in_last_configuration[i];
	}
}

GradientTable::~GradientTable() {

}

void GradientTable::AddComputationResult(int in_component, double in_step, double in_quality){
	steps[in_component] = in_step;
	qualities[in_component] = in_quality;
}

void GradientTable::AddFullStepResult(dealii::Vector<double> in_step, double in_quality) {
	for(int i = 0 ; i < nfreedofs; i++) {
		grad_step[i] = in_step[i];
	}
	final_quality = in_quality;
}

void GradientTable::PrintFullLine() {
	struct winsize w;
	ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
	for ( int i = 0; i<w.ws_col; i++) {
		std::cout << "-";
	}
	std::cout << std::endl;
}

void GradientTable::PrintTable() {
	PrintFullLine();
	std::cout << "Gradient Table for Step: " <<  GlobalStep << std::endl;
	std::cout << std::setw(14) << "-" << "|";
	for(int i =0; i< nfreedofs; i++) {
		if(i%3 ==0) {
			std::cout << "m_" << std::setw(10) << i/3 +1;
		}
		if(i%3 ==1) {
			std::cout << "r_" << std::setw(10) << i/3 +1;
		}
		if(i%3 ==2) {
			std::cout << "v_" << std::setw(10) << i/3 +1;
		}
		std::cout << "|";
	}
	std::cout << "Quality   " << "|"<<std::endl;

	PrintFullLine();

	std::cout<< "Initial Config|";
	for(int i =0; i<nfreedofs; i++) {
		std::cout<< std::setw(12) << ref_configuration[i];
		std::cout<<"|";
	}
	std::cout << std::setw(10)<< initial_quality;
	std::cout << "|"<<std::endl;

	std::cout<< std::setw(14)<< "Last Config";
	std::cout<< "|";
	for(int i =0; i<nfreedofs; i++) {
		std::cout<< std::setw(12) << last_configuration[i];
		std::cout<<"|";
	}
	std::cout << std::setw(10)<< last_quality;
	std::cout << "|"<<std::endl;

	std::cout<< std::setw(14)<< "Delta";
	std::cout<< "|";
	for(int i =0; i<nfreedofs; i++) {
		std::cout<< std::setw(12) << steps[i];
		std::cout<<"|";
	}
	std::cout << std::setw(10)<< " ";
	std::cout << "|"<<std::endl;

	std::cout<< std::setw(14)<< "Quality";
	std::cout<< "|";
	for(int i =0; i<nfreedofs; i++) {
		std::cout<< std::setw(12) << qualities[i];
		std::cout<<"|";

	}
	std::cout << std::setw(10)<< " ";
	std::cout << "|"<<std::endl;

	std::cout<< std::setw(14)<< "Gradient Step";
	std::cout<< "|";
	for(int i =0; i<nfreedofs; i++) {
		std::cout<< std::setw(12) << grad_step[i];
		std::cout<<"|";
	}
	std::cout << std::setw(10)<< " ";
	std::cout << "|"<<std::endl;

	std::cout<< std::setw(14)<< "New Config";
	std::cout<< "|";
	for(int i =0; i<nfreedofs; i++) {
		std::cout<< std::setw(12) << last_configuration[i]+ grad_step[i];
		std::cout<<"|";
	}
	std::cout << std::setw(10)<< final_quality;
	std::cout << "|"<<std::endl;

	PrintFullLine();
}

void GradientTable::WriteTableToFile(std::string ){

}

