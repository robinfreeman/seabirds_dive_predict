#!/bin/bash
# Author: Luke Swaby (lds20@ic.ac.uk)
# Script: run_GPS.sh
# Desc: Processes and plots GPS data extracted from logger
# Arguments: none
# Date: Apr 2021

echo '\n\nExtracting GPS data...\n\n'
Rscript extract_GPS.R

echo '\n\nCombining GPS data...\n\n'
Rscript combine_GPS.R

echo '\n\nPlotting GPS data...\n\n'
Rscript plot_GPS.R