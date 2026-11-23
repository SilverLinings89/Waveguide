#pragma once

/**
 * @file TimerManager.h
 * @author Pascal Kraft
 * @brief Implementation of a handler for multiple timers with names that can gernerate output.
 * @version 0.1
 * @date 2022-03-22
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#include <fstream>
#include <string>
#include <vector>
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
    /** @brief One timer per level, indexed by level. Each uses the communicator of its level and measures CPU and wall time. */
    std::vector<dealii::TimerOutput> timer_outputs;
    /** @brief Names of the output files of the timers, indexed by level. The name is "timer_output" followed by the level and ".dat". */
    std::vector<std::string> filenames;
    /** @brief Output streams of the files in filenames. Owned by this object and deleted in finalize(). */
    std::vector<std::ofstream *> filestreams;
    /** @brief Number of levels with a timer: 1 plus the number of directions with more than one block. */
    unsigned int level_count;

    /** @brief Does nothing. The object is set up in initialize(). */
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

    /**
     * @brief Destroys all timers and closes the output files. Has to be called before MPI is finalized.
     * 
     */
    void finalize();
};