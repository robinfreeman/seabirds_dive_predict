# Author: Robin Freeman (robin.freeman@ioz.ac.uk) Luke Swaby (lds20@ic.ac.uk)
# Script: combine_GPS.R
# Desc: Load GPS data and combine into a single data frame
# Date: Apr 2021

library(geosphere)
library(data.table)

# Get GPS data, merge into single data frame
gps_files = list.files("../Data/BIOT_DGBP/BIOT_DGBP/", pattern = "*_loc.csv", full.names = T)
data_list = list()

for (i in 1:length(gps_files)) {
  f = gps_files[i]
  cat(sprintf("Processing file: %s\n", f))
  ts_data_loc = fread(file = f)
  
  # convert date and time to readable format
  datetime_s = paste(ts_data_loc$Date, ts_data_loc$Time)
  ts_data_loc$datetime = as.POSIXct(datetime_s, format = "%d/%m/%Y %H:%M:%S.000")
  
  # work out max dist from dg and total distance
  ts_data_loc$dist_to_dg_m = distHaversine(c(72.41111, -7.31333), cbind(ts_data_loc$`location-lon`, ts_data_loc$`location-lat`))
  ts_data_loc$dist_to_dg_km = ts_data_loc$dist_to_dg_m/1000
  
  # Initialise cols
  ts_data_loc$dist_moved_m = 0 
  ts_data_loc$time_diff_s = 0
  
  n = nrow(ts_data_loc)
  
  # Calculate distance moved and time taken between each data point
  ts_data_loc$dist_moved_m[-1] = distHaversine(cbind(ts_data_loc$`location-lon`[-n], ts_data_loc$`location-lat`[-n]), cbind(ts_data_loc$`location-lon`[-1], ts_data_loc$`location-lat`[-1]))
  ts_data_loc$time_diff_s[-1] = as.numeric(ts_data_loc$datetime[-1] - ts_data_loc$datetime[-n])
  
  # Calculate speed
  ts_data_loc$calc_sp_ms = ts_data_loc$dist_moved_m/ts_data_loc$time_diff_s
  
  data_list[[i]] = ts_data_loc
}

gps_data_df = rbindlist(data_list)

save(gps_data_df, file = "../Data/BIOT_DGBP/gps_data_df.RData")
write.csv(gps_data_df, "../Data/BIOT_DGBP/all_gps_data.csv")