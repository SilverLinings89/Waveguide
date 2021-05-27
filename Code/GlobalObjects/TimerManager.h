#pragma once

#include <deal.II/base/timer.h>
#include <array>

class TimerManager {
    public: 
    std::vector<dealii::TimerOutput> timer_outputs;
    std::vector<std::string> filenames;
    std::vector<std::ofstream *> filestreams;
    unsigned int level_count;

    TimerManager();

    void initialize();
    void switch_context(std::string context, unsigned int level);
    void write_output();

};