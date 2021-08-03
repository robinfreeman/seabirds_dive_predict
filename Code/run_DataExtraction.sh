#!/bin/bash
# Author: Luke Swaby (lds20@ic.ac.uk)
# Script: run_GPS.sh
# Desc: Processes and plots GPS data extracted from logger
# Arguments: none
# Date: Apr 2021

Rscript extract_depth.R

Rscript extract_GPS.R

#Rscript plot_depth.R

Rscript extract_GLS.R

Rscript depth_to_immersion.R

# Create training data sets for deep learning
for i in $(seq 2 2 10) 
do 
	outpth=../Data/ACC${i}_reduced_all_dives.csv
	python3 ../Code/write_training_dsets.py -t ACC -o $outpth -w $i --reduce
done


for i in $(seq 60 120 1200) 
do 
	outpth=../Data/IMM${i}_reduced_all_dives.csv
	python3 ../Code/write_training_dsets.py -t IMM -o $outpth -w $i --reduce
done


# train models

python3 AWCK.py -i ../Data/ -t ACC
python3 AWCK.py -i ../Data/ -t ACC