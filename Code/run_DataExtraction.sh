#!/bin/bash
# Author: Luke Swaby (lds20@ic.ac.uk)
# Script: run_GPS.sh
# Desc: Processes and plots GPS data extracted from logger
# Arguments: none
# Date: Apr 2021

Rscript extract_GPS.R

Rscript extract_depth.R

Rscript plot_depth.R

Rscript extract_GLS.R

Rscript depth_to_immersion.R