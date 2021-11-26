#include "ResidualOutputGenerator.h" 
#include "../../GlobalObjects/GlobalObjects.h"
#include "../../Helpers/staticfunctions.h"

ResidualOutputGenerator::ResidualOutputGenerator():
    name(""),
    print_to_console(false),
    level(0) {

}   

ResidualOutputGenerator::ResidualOutputGenerator(std::string in_name, std::string plot_title, bool in_print_to_console, unsigned int in_level):
    name(in_name),
    print_to_console(in_print_to_console),
    level(in_level) {
    fname = GlobalOutputManager.get_numbered_filename(name, GlobalParams.MPI_Rank, "plt");
    ofname = GlobalOutputManager.get_numbered_filename(name, GlobalParams.MPI_Rank, "png");
    for(unsigned int i = 0; i < GlobalParams.HSIE_SWEEPING_LEVEL - level; i++) {
        spacing += "\t";
    }
}
    
void ResidualOutputGenerator::push_value(double value) {
    if(data_series.size() > 0) {
        data_series[data_series.size() - 1].values.push_back(value);
    }
}

void ResidualOutputGenerator::write_residual_statement_to_console() {
    if (print_to_console) {
        if(data_series.size() > 0) {
            if(data_series[data_series.size() - 1].values.size() > 1) {
                const unsigned step = data_series[data_series.size() - 1].values.size();
                const double residual = data_series[data_series.size() - 1].values[step - 1];
                const double fraction = residual / data_series[data_series.size() - 1].values[step - 2];
                std::cout << spacing << "Residual in step " << step << " : " << residual << " Factor: " << fraction << std::endl;
            } else {
                if(data_series[data_series.size() - 1].values.size() == 1) {
                    std::cout << spacing << "Residual in step 1 : " << std::to_string(data_series[data_series.size() - 1].values[0]) << std::endl;
                } else {
                    std::cout << "No residuals passed so far. Error!" << std::endl;
                }
            }
        } else {
            std::cout << "No residual series yet. Error!" << std::endl;
        }
    }
}

void ResidualOutputGenerator::close_current_series() {
    data_series[data_series.size() - 1].is_closed = true;
}

void ResidualOutputGenerator::new_series(std::string in_name) {
    DataSeries new_series;
    new_series.is_closed = false;
    new_series.values = std::vector<double>();
    new_series.name = in_name;
    data_series.push_back(new_series);
}

void ResidualOutputGenerator::write_gnuplot_file() {
    if(!generated_script) {
        std::ofstream out(fname);
        out << "reset" << std::endl << "set term png" << std::endl << "set output \"" << ofname <<"\"" << std::endl;
        for(unsigned int i = 0; i < data_series.size(); i++) {
            out << "array series" << i << "[" << data_series[i].values.size() << "]" << std::endl;
            for(unsigned int val = 0; val < data_series[i].values.size(); val++) {
                out << "series" << i << "["<< val + 1 << "] = " << data_series[i].values[val] << std::endl;
            }
        }
        out << "set logscale y" << std::endl << "set xlabel 'Step'" << std::endl << "set ylabel 'Residual'" << std::endl <<  "set format y \"%.1e\"" << std::endl;
        out << "plot ";
        for(unsigned int i = 0; i < data_series.size(); i++) {
            out << "series" << i << " with linespoints title\"" << data_series[i].name << "\"";
            if(i != data_series.size() -1) {
                out << ", ";
            } else {
                out << std::endl;
            }
        }
        generated_script = true;
        out.close();
    }
}

void ResidualOutputGenerator::run_gnuplot() {
    if(!generated_plot) {
        write_gnuplot_file();
        std::string command = "gnuplot " + fname;
        exec(command.c_str());
        generated_plot = true;
    }
}
