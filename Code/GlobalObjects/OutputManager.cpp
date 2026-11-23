#include <string>
#include "./OutputManager.h"
#include "../GlobalObjects/GlobalObjects.h"
#include "../Helpers/staticfunctions.h"
#include <iostream>
#include <fstream>

OutputManager::OutputManager() { }

void OutputManager::initialize() {
    // The base folder for outputs is, in this order of precedence:
    //  1. the environment variable WAVEGUIDE_OUTPUT_DIR,
    //  2. $WORK/Solutions (HPC workspaces),
    //  3. the folder Solutions in the repository.
    std::string base_path = repository_path("Solutions");
    const char * work_path = getenv("WORK");
    const char * output_path = getenv("WAVEGUIDE_OUTPUT_DIR");
    if (work_path != NULL) {
        base_path = std::string(work_path) + "/Solutions";
    }
    if (output_path != NULL) {
        base_path = std::string(output_path);
    }
    if(GlobalParams.MPI_Rank == 0) {
        mkdir(base_path.c_str(), ACCESSPERMS);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    int i = 0;
    bool dir_exists = true;
    while (dir_exists) {
        output_folder_path = base_path + "/run" + std::to_string(i);
        struct stat myStat;
        const char *myDir = output_folder_path.c_str();
        if ((stat(myDir, &myStat) == 0) && (((myStat.st_mode) & S_IFMT) == S_IFDIR)) {
            i++;
        } else {
            dir_exists = false;
        }
    }
    i = dealii::Utilities::MPI::max(i, MPI_COMM_WORLD);
    output_folder_path = base_path + "/run" + std::to_string(i);
    if(GlobalParams.MPI_Rank == 0) {
        print_info("OutputManager::initialize", "Solution path: " + output_folder_path);
        mkdir(output_folder_path.c_str(), ACCESSPERMS);
        std::string git_commit_hash = exec(("git -C \"" + repository_path("") + "\" rev-parse HEAD 2>/dev/null").c_str());
        git_commit_hash.erase(std::remove(git_commit_hash.begin(), git_commit_hash.end(), '\n'), git_commit_hash.end());
        write_run_description(git_commit_hash);
        print_info("OutputManager::initialize","Git status: " + git_commit_hash);
    }
    // All processes wait until the output folder exists before they open their log files.
    MPI_Barrier(MPI_COMM_WORLD);
    log_stream.open(output_folder_path + "/main" + std::to_string(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)) + ".log", std::ios::binary);
}

OutputManager::~OutputManager() {
    log_stream.close();
}

std::string OutputManager::get_full_filename(std::string filename) {
    return output_folder_path + "/" + filename;
}

void OutputManager::write_run_description(std::string git_commit_hash) {
    std::string filename = get_full_filename("run_description.txt");
    std::ofstream out(filename);
    out << "Number of processes: \t" << GlobalParams.NumberProcesses << std::endl;
    out << "Sweeping level: " << GlobalParams.Sweeping_Level << std::endl;
    out << "Truncation Method: " << ((GlobalParams.BoundaryCondition == BoundaryConditionType::HSIE)? "HSIE" : "PML") << std::endl;
    out << "Signal input method: " << (GlobalParams.use_tapered_input_signal ? "Taper" : "Dirichlet") << std::endl;
    out << "Set 0 on input interface: " << (GlobalParams.prescribe_0_on_input_side ? "true" : "false") << std::endl;
    out << "Use predefined shape: " << (GlobalParams.Use_Predefined_Shape ? "true" : "false") << std::endl;
    if(GlobalParams.Use_Predefined_Shape) {
        out << "Predefined Shape Number: " << GlobalParams.Number_of_Predefined_Shape << std::endl;
    }
    out << "Block Counts: [" << GlobalParams.Blocks_in_x_direction << "x" << GlobalParams.Blocks_in_y_direction << "x" << GlobalParams.Blocks_in_z_direction << "]" << std::endl;
    out << "Global cell count x: " << GlobalParams.Blocks_in_x_direction * GlobalParams.Cells_in_x << std::endl;
    out << "Global cell count y: " << GlobalParams.Blocks_in_y_direction * GlobalParams.Cells_in_y << std::endl;
    out << "Global cell count z: " << GlobalParams.Blocks_in_z_direction * GlobalParams.Cells_in_z << std::endl;
    out << "Number of PML cell layers: " << GlobalParams.PML_N_Layers << std::endl;
    out << "Use relative convergence limiter: " << (GlobalParams.use_relative_convergence_criterion ? "true" : "false") << std::endl;
    if(GlobalParams.use_relative_convergence_criterion) {
        out << "Relative convergence limit: " << GlobalParams.relative_convergence_criterion << std::endl;
    }
    out << "Global x range: " << Geometry.global_x_range.first << " to " << Geometry.global_x_range.second <<std::endl;
    out << "Global y range: " << Geometry.global_y_range.first << " to " << Geometry.global_y_range.second <<std::endl;
    out << "Global z range: " << Geometry.global_z_range.first << " to " << Geometry.global_z_range.second <<std::endl;
    out << "Git commit hash: " << git_commit_hash << std::endl;
    out.close();
}

std::string OutputManager::get_numbered_filename(std::string filename, unsigned int number, std::string extension) {
    return get_full_filename(filename) + std::to_string(number) + '.' + extension;
}

void OutputManager::write_log_ling(std::string in_line) {
    log_stream << in_line << std::endl;
}