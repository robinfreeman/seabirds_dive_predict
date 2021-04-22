# Author: Robin Freeman (robin.freeman@ioz.ac.uk) Luke Swaby (lds20@ic.ac.uk)
# Script: extract_pressure.R
# Desc: Creates csv files containing only those rows of the parent files with pressure data
# Date: Apr 2021

library(geosphere)
library(data.table)
library(ggplot2)
library(gridExtra)

#################### Extract pressure data ####################

# Get all Pressure files (containing Pressure/activity data)
files = list.files("../Data/BIOT_DGBP/BIOT_DGBP/", pattern = "1.csv", full.names = T)
data_list = list()

# For each file, extract pressure data and resave as *_p.csv
for (i in 1:length(files)) {
  
  f = files[i]
  cat(sprintf("Processing file: %s\n", f))
  
  # Read in file and subset rows containing pressure data
  ts_data = fread(file = f)
  ts_data_p = ts_data[!is.na(Pressure)]
  
  # convert date and time to readable format
  datetime_s = paste(ts_data_p$Date, ts_data_p$Time)
  ts_data_p$datetime = as.POSIXct(datetime_s, format = "%d/%m/%Y %H:%M:%S.000")
  
  # work out max dist from dg and total distance
  ts_data_p$dist_to_dg_m = distHaversine(c(72.41111, -7.31333), cbind(ts_data_p$`location-lon`, ts_data_p$`location-lat`))
  ts_data_p$dist_to_dg_km = ts_data_p$dist_to_dg_m/1000
  
  # Initialise cols
  ts_data_p$dist_moved_m = ts_data_p$time_diff_s = NA
  
  # Note indicies of GPS rows
  gps_idx = which(!is.na(ts_data_p$`location-lon`))
  n = length(gps_idx) 
  
  # Initialise vecs
  dist.temp = time.diff.temp = rep(0, n) 
  
  # Subset cols
  lon.temp = ts_data_p$`location-lon`[gps_idx]
  lat.temp = ts_data_p$`location-lat`[gps_idx]
  datetime.temp = ts_data_p$datetime[gps_idx]
  
  # Calculate distance moved and time taken between each data point
  dist.temp[-1] = distHaversine(cbind(lon.temp[-n], lat.temp[-n]), cbind(lon.temp[-1], lat.temp[-1]))
  time.diff.temp[-1] = as.numeric(datetime.temp[-1] - datetime.temp[-n])
  
  # Load into df
  ts_data_p$dist_moved_m[gps_idx] = dist.temp
  ts_data_p$time_diff_s[gps_idx] = time.diff.temp
  
  # Calculate speed
  ts_data_p$calc_sp_ms = ts_data_p$dist_moved_m/ts_data_p$time_diff_s
  
  # Write data frame to out file and add to data_list
  fwrite(ts_data_p, gsub(".csv", "_p.csv", f)) # write out file
  data_list[[i]] = ts_data_p
}

p_data_df = rbindlist(data_list)

fwrite(p_data_df, file = "../Data/BIOT_DGBP/all_p_data.csv")

#################### Calculate Altitude ####################

# Function to calculate altutude from pressure and temperature
calc_height = function(Pressure_0, Pressure, Temp) {
  rel_p = (Pressure_0/Pressure)
  temp_K = Temp + 273.15
  lapse_rate = 0.0065
  
  h = ( ( rel_p^(1/5.257) - 1) * (temp_K) )/lapse_rate
  return(h)
}

# Add altitude col
Pressure_at_sea_level = 1013.25
p_data_df$altitude = calc_height(Pressure_at_sea_level, p_data_df$Pressure, p_data_df$`Temp. (?C)`)

#################### Plot GPS Altitude and Pressure for each bird ####################

birds = unique(p_data_df$TagID)

for (i in 1:length(birds)) {
  this_bird = birds[i]
  this_data = subset(p_data_df, TagID == this_bird)
  
  g1 = ggplot(this_data, aes(x = datetime, y = `height-above-msl`)) +  
    geom_point() 
  g2 = ggplot(this_data, aes(x = datetime, y = Pressure))  +  
    geom_point() + geom_hline(yintercept=Pressure_at_sea_level, color="blue", linetype="dashed")
  g3 = ggplot(this_data, aes(x = datetime, y = altitude))  +  
    geom_point() + geom_hline(yintercept=0, color="blue", linetype="dashed")
  
  g = grid.arrange(g1, g2, g3, ncol=1)
  
  ggsave(g, file=paste0('../Data/BIOT_DGBP/', this_bird, "_alt_plot.png"), width = 9, height = 9)
}