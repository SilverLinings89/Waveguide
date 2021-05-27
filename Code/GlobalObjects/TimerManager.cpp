#include "TimerManager.h"
#include <deal.II/base/timer.h>
#include "GlobalObjects.h"
#include <iostream>
#include <fstream>

TimerManager::TimerManager() { }

void TimerManager::initialize() {
    level_count = 1;
    if(GlobalParams.Blocks_in_z_direction != 1) {
        level_count ++;
    }
    if(GlobalParams.Blocks_in_y_direction != 1) {
        level_count ++;
    }
    if(GlobalParams.Blocks_in_x_direction != 1) {
        level_count ++;
    }
    for(unsigned int i =0 ; i < level_count; i++) {
        std::string fname = GlobalOutputManager.get_numbered_filename("timer_output", i, "dat");
        filenames.push_back(fname);
        std::ofstream * filestream = new std::ofstream(fname);
        filestreams.push_back(filestream);
        timer_outputs.emplace_back(GlobalMPI.communicators_by_level[i], *(filestreams[i]), dealii::TimerOutput::summary, dealii::TimerOutput::cpu_and_wall_times);
    }
}

void TimerManager::switch_context(std::string context, unsigned int level) {
    dealii::TimerOutput::Scope next_section(timer_outputs[level], context);
}

void TimerManager::write_output() {
    for(unsigned int i = 0; i < level_count; i++) {
        timer_outputs[i].print_summary();
        timer_outputs[i].print_wall_time_statistics(GlobalMPI.communicators_by_level[i]);
    }
}
