#pragma once

/**
 * @file TimerManager.h
 * @author your name (you@domain.com)
 * @brief Implementation of a handler for multiple timers with names that can gernerate output.
 * @version 0.1
 * @date 2022-03-22
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#include <deal.II/base/timer.h>
#include <array>

/**
 * @brief A class that stores timers for later output.
 * It uses sections to compute all times of similar type, like all solve calls on a certain level or all assembly work.
 * The object computes timing individually for every level.
 * 
 */
class TimerManager {
    public: 
    std::vector<dealii::TimerOutput> timer_outputs;
    std::vector<std::string> filenames;
    std::vector<std::ofstream *> filestreams;
    unsigned int level_count;

    TimerManager();

    /**
     * @brief Preoares the internal datastructures.
     * 
     */
    void initialize();

    /**
     * @brief After this point, the timers will count towards the new section.
     * 
     * @param context Name of the section to switch to.
     * @param level The level we are currently on.
     */
    void switch_context(std::string context, unsigned int level);

    /**
     * @brief Writes an output file containing all the timer information about all levels and sections.
     * 
     */
    void write_output();

    /**
     * @brief End contribution to the current context on the provided level.
     * 
     * @param level The HSIE sweeping level whose timing measurements we want to switch to another context. If we get done with assembly work on level two and want to switch to solving, we would call leave_context(2) followed by enter_context("solve", 2).
     */
    void leave_context(unsigned int level);
};