# Author: Robin Freeman (robin.freeman@ioz.ac.uk) Luke Swaby (lds20@ic.ac.uk)
# Script: extract_GPS.R
# Desc: Creates csv files containing only those rows of the parent files with GPS data
# Date: Apr 2021

library(data.table)

# Get all CSV files (containing GPS/ACC data)
files = list.files("../Data/BIOT_DGBP/BIOT_DGBP/", pattern = "1.csv", full.names = T)

# For each file, extract just location data and resave as *_loc.csv
for (f in files) {
  ts_data = fread(file = f)
  # How many rows
  nrow(ts_data)
  
  ts_data_loc = subset(ts_data, !is.na(`location-lat`))
  
  nrow(ts_data_loc)
  write.csv(ts_data_loc, gsub(".csv", "_loc.csv", f))
}