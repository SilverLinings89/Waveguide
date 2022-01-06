#include "ConvergenceOutputGenerator.h" 
#include "../../GlobalObjects/GlobalObjects.h"
#include "../../Helpers/staticfunctions.h"

ConvergenceOutputGenerator::ConvergenceOutputGenerator()
{
    title = "";
}

ConvergenceOutputGenerator::~ConvergenceOutputGenerator()
{ }   

void ConvergenceOutputGenerator::set_title(std::string in_title) {
    title = in_title;
}

void ConvergenceOutputGenerator::push_values(double in_x, double in_y) {
    x.push_back(in_x);
    y.push_back(in_y);
}
  
void ConvergenceOutputGenerator::write_gnuplot_file() {
    fname = GlobalOutputManager.get_numbered_filename("Convergence", GlobalParams.MPI_Rank, "plt");
    ofname = GlobalOutputManager.get_numbered_filename("Convergence", GlobalParams.MPI_Rank, "png");
    std::ofstream out(fname);
    out << "reset" << std::endl << "set term png" << std::endl << "set output \"" << ofname <<"\"" << std::endl;
    out << "set title \"" << title << "\"" << std::endl;
    out << "show title" << std::endl;
    out << "array xvals[" << x.size() << "]" << std::endl;
    for(unsigned int val = 0; val < x.size(); val++) {
        out << "xvals["<< val + 1 << "] = " << x[val] << std::endl;
    }
    out << "array yvals[" << y.size() << "]" << std::endl;
    for(unsigned int val = 0; val < y.size(); val++) {
        out << "yvals["<< val + 1 << "] = " << y[val] << std::endl;
    }
    out << "set logscale y" << std::endl << "set logscale x" << std::endl << "set xlabel '" << x_label << "'" << std::endl << "set ylabel '" << y_label<< "'" << std::endl <<  "set format y \"%.1e\"" << std::endl;
    out << "plot sample [i=1:" << x.size() +1<< "] '+' using (xvals[i]):(yvals[i]) with linespoints" << std::endl;
    out.close();

}

void ConvergenceOutputGenerator::run_gnuplot() {
    write_gnuplot_file();
    std::string command = "gnuplot " + fname;
    exec(command.c_str());
}
