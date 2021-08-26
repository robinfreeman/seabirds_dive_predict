#!/bin/bash
# Author: Luke Swaby (lds20@ic.ac.uk)
# Script: RUN_PROJECT.sh
# Desc: Runs all code scripts to compile project
# Arguments: none
# Date: Aug 2021

Rscript extract_GPS.R

Rscript extract_GLS.R

Rscript depth_to_IMM.R

# Create training data sets for deep learning
for i in $(seq 2 2 10) 
do 
	outpth=../Data/ACC${i}_reduced_all_dives.csv
	python3 write_rolling_dsets.py -d ACC -i ../Data/BIOT_DGBP/ -o $outpth -w $i -r 25 --reduce
done


for i in $(seq 60 120 540) 
do 
	outpth=../Data/IMM${i}_reduced_all_dives.csv
	python3 write_rolling_dsets.py -d IMM -o $outpth -w $i -r 6 --reduce
done


# Train models
python3 build_train_xvalidate_ANN.py -t ACC -i ../Data/ -o ../Results/ -e 50
python3 build_train_xvalidate_ANN.py -t IMM -i ../Data/ -o ../Results/ -e 50

# Run jupyter notebook to generate plots (requires runipy command line tool)
runipy plots.ipynb

# Additional diluted ACC analysis

for x in 1 2 4
do
	Rscript --vanilla ACC_sumstats.R $x
	
	python3 ACC_sumstats.py -w $x
done
