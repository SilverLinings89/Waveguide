#include "./OutputManager.h"
#include "../GlobalObjects/GlobalObjects.h"
#include "../Helpers/staticfunctions.h"
#include <iostream>
#include <fstream>

OutputManager::OutputManager() { }

void OutputManager::initialize() {
    char *pPath;
    pPath = getenv("WORK");
    bool seperate_solutions = (pPath != NULL);
    int i = 0;
    bool dir_exists = true;
    while (dir_exists) {
        std::stringstream out;
        if (seperate_solutions) {
            out << pPath << "/";
        }
        out << "../Solutions/run";
        out << i;
        output_folder_path = out.str();
        struct stat myStat;
        const char *myDir = output_folder_path.c_str();
        if ((stat(myDir, &myStat) == 0) && (((myStat.st_mode) & S_IFMT) == S_IFDIR)) {
            i++;
        } else {
            dir_exists = false;
        }
    }
    i = dealii::Utilities::MPI::max(i, MPI_COMM_WORLD);
    std::stringstream out;
    if (seperate_solutions) {
        out << pPath << "/";
    }
    out << "../Solutions/run" << i;
    output_folder_path = out.str();
    if(GlobalParams.MPI_Rank == 0) {
        std::cout << "Solution path: " << output_folder_path << std::endl;
    }
    mkdir(output_folder_path.c_str(), ACCESSPERMS);
    log_stream.open(output_folder_path + "/main" + std::to_string(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)) + ".log", std::ios::binary);
    write_run_description();
    print_info("Git status:", exec("git rev-parse HEAD"));
}

OutputManager::~OutputManager() {
    log_stream.close();
}

std::string OutputManager::get_full_filename(std::string filename) {
    return output_folder_path + "/" + filename;
}

void OutputManager::write_run_description() {
    std::string filename = get_full_filename("run_description.txt");
    std::ofstream out(filename);
    out << "n_procs\t" << GlobalParams.NumberProcesses << std::endl;
    out << "trunc_method\t" << ((GlobalParams.BoundaryCondition == BoundaryConditionType::HSIE)? "HSIE" : "PML") << std::endl;
    out << "in_signal\t" << (GlobalParams.use_tapered_input_signal ? "Taper" : "Dirichlet") << std::endl;
    out << "input_dirichlet_zero" << (GlobalParams.prescribe_0_on_input_side ? "true" : "false") << std::endl;
    out.close();
}

std::string OutputManager::get_numbered_filename(std::string filename, unsigned int number, std::string extension) {
    return get_full_filename(filename) + std::to_string(number) + '.' + extension;
}

void OutputManager::write_log_ling(std::string in_line) {
    log_stream << in_line << std::endl;
}