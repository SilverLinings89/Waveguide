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

void ConvergenceOutputGenerator::push_values(double in_x, double in_y_num, double in_y_theo) {
    x.push_back(in_x);
    y_numerical.push_back(in_y_num);
    y_theoretical.push_back(in_y_theo);
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
    out << "array yvalsnum[" << y_numerical.size() << "]" << std::endl;
    for(unsigned int val = 0; val < y_numerical.size(); val++) {
        out << "yvalsnum["<< val + 1 << "] = " << y_numerical[val] << std::endl;
    }
    out << "array yvalstheo[" << y_numerical.size() << "]" << std::endl;
    for(unsigned int val = 0; val < y_theoretical.size(); val++) {
        out << "yvalstheo["<< val + 1 << "] = " << y_theoretical[val] << std::endl;
    }
    out << "set logscale y" << std::endl << "set logscale x" << std::endl << "set xlabel '" << x_label << "'" << std::endl << "set ylabel '" << y_label<< "'" << std::endl <<  "set format y \"%.1e\"" << std::endl;
    out << "plot sample [i=1:" << x.size() +1<< "] '+' using (xvals[i]):(yvalsnum[i]) with linespoints title \"numerical error\", [j=1:" << x.size() +1<< "] '+' using (xvals[j]):(yvalstheo[j]) with linespoints title \"theoretical error\"" << std::endl;
    out.close();

}

void ConvergenceOutputGenerator::run_gnuplot() {
    write_gnuplot_file();
    std::string command = "gnuplot " + fname;
    exec(command.c_str());
}
