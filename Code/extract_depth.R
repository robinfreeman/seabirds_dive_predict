# Author: Robin Freeman (robin.freeman@ioz.ac.uk) Luke Swaby (lds20@ic.ac.uk)
# Script: extract_depth.R
# Desc: Creates csv files containing only those rows of the parent files with depth data
# Date: Apr 2021

## Imports
suppressMessages(library(data.table))
suppressMessages(library(dplyr))
suppressMessages(library(geosphere))

## Global variables
threshold = 0.03

# Get all Pressure files (containing Pressure/activity data)
cat('\nExtracting pressure data...\n\n')
files = list.files("../Data/BIOT_DGBP/BIOT_DGBP/", pattern = "1.csv", full.names = TRUE)
data_list = list()
sum_stats = list()

# Function to extract statistical mode
getmode <- function(v) {
  uniqv <- unique(na.omit(v))
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

# For each file, extract pressure data and resave as *_p.csv
for (i in 1:length(files)) {
  
  f = files[i]
  cat(sprintf("Processing file: %s\n", f))
  
  cat("Reading in file...")
  
  # Read in file and subset rows containing pressure data
  ts_data = fread(file = f)
  ts_data_d = ts_data[!is.na(Depth)]
  this_bird = unique(ts_data_d$TagID)
  
  ################ CALCULATING DISTANCE DATA #####################
  cat("\rCalculating distances travelled...")
  
  # Determine nest coordinates (as then most common GPS)
  nest_coords = c(getmode(ts_data_d$`location-lon`), getmode(ts_data_d$`location-lat`))
  
  # convert date and time to readable format
  datetime_s = paste(ts_data_d$Date, ts_data_d$Time)
  ts_data_d$datetime = as.POSIXct(datetime_s, format = "%d/%m/%Y %H:%M:%S.000")
  
  # work out max dist from dg and total distance
  ts_data_d$dist_to_dg_m = distHaversine(nest_coords, cbind(ts_data_d$`location-lon`, ts_data_d$`location-lat`))
  ts_data_d$dist_to_dg_km = ts_data_d$dist_to_dg_m/1000
  
  # Initialise cols
  ts_data_d$dist_moved_m = ts_data_d$time_diff_s = NA
  
  # Note number of GPS rows and their indexes
  gps_idx = which(!is.na(ts_data_d$`location-lon`))
  n = length(gps_idx) 
  
  # Initialise vecs
  dist.temp = time.diff.temp = rep(0, n) 
  
  # Subset cols
  lon.temp = ts_data_d$`location-lon`[gps_idx]
  lat.temp = ts_data_d$`location-lat`[gps_idx]
  datetime.temp = ts_data_d$datetime[gps_idx]
  
  # Calculate distance moved and time taken between each data point
  dist.temp[-1] = distHaversine(cbind(lon.temp[-n], lat.temp[-n]), cbind(lon.temp[-1], lat.temp[-1]))
  time.diff.temp[-1] = as.numeric(datetime.temp[-1] - datetime.temp[-n])
  
  # Load into df
  ts_data_d$dist_moved_m[gps_idx] = dist.temp
  ts_data_d$time_diff_s[gps_idx] = time.diff.temp
  
  # Calculate speed
  ts_data_d$calc_sp_ms = ts_data_d$dist_moved_m/ts_data_d$time_diff_s
  
  ################ INTERPOLATE GPS #####################
  cat("\rInterpolating GPS...")
  diffs = diff(gps_idx)
  # 1. Find gps_indexes between which to interpolate
  gaps = which(diffs>60) 
  starts = gps_idx[gaps]
  # 2. Determine number of interpolations to be made between each
  steps = floor(diffs[gaps]/30)
  # 3. Find hypothetical indexes to interpolate at
  int_ix = apply(cbind(starts,steps), 1, function(v) seq(v[1], v[1]+v[2]*30, 30))
  # 4. Add these indexes to gps_index (maybe a new vector)
  gps_idx = sort(unique(c(unlist(int_ix), gps_idx)))
  # 5. Subset data for location interpolation
  loc_data_int = ts_data_d[gps_idx]
  # 6. Interpolate GPS rows
  loc_cols_int = zoo::na.approx(loc_data_int[,c("location-lat", "location-lon")])
  # 7 Push back into main df
  ts_data_d[gps_idx, c("location-lat", "location-lon")] = data.frame(loc_cols_int)
  
  ################ ADD ACCELERATION COL #####################
  #ts_data_a = ts_data[!is.na(X)]
  #ts_data_a$Acceleration = sqrt(ts_data_a$X^2 + ts_data_a$Y^2 + ts_data_a$Z^2) # magnitude of acceleration
  #ts_data_a$Mean_acceleration = NA
  #wndow = 0.5*30*25
  #gps_idx = which(!is.na(ts_data_a$`location-lat`))
  #mean_acc = sapply(gps_idx, function(i) mean(ts_data_a$Acceleration[(i-wndow):(i+wndow)]))
  #mean_depth[is.na(mean_depth)] = 0
  #ts_data_a$Mean_acceleration[gps_idx] = mean_acc
  ######################################################
  
  ######## REMOVE BACKGROUND NOISE ########## 
  cat("\rTransforming data...")
  k = 30
  
  # take rolling median as baseline for each window
  offset = zoo::rollapply(ts_data_d$Depth, width=k, by=k, FUN=median)
  offset = rep(offset, each=k)
  
  # match lengths
  dif = length(ts_data_d$Depth) - length(offset)
  offset = c(offset, rep(tail(offset, 1), dif))
  
  # Zero-offset data
  new_series = ts_data_d$Depth - offset
  new_series[new_series<0] = 0  # negative depth meaningless
  new_series[(length(new_series)-dif):length(new_series)] = 0  # no dives as device removed
  ts_data_d$Depth_mod = new_series

  ############ MAX/MEAN DEPTH ############
  cat("\rAdding cols containing max/mean depth around GPS records...")
  
  # New cols for dive profile assignment
  ts_data_d$Dive = ts_data_d$Max_depth_m = ts_data_d$Mean_depth_m = NA
  
  wndow = 15
  
  # Assign dive profiles based on deepest/mean depth within 15s of GPS record
  deepest = sapply(gps_idx, function(i) max(ts_data_d$Depth_mod[(i-wndow):(i+wndow)]))
  deepest[is.na(deepest)] = 0 # last value will be NA as there aren't 30 rows past it
  
  mean_depth = sapply(gps_idx, function(i) mean(ts_data_d$Depth_mod[(i-wndow):(i+wndow)]))
  mean_depth[is.na(mean_depth)] = 0 # last value will be NA as there aren't 30 rows past it
  
  dives = sapply(deepest, function(x) x>threshold)
  
  # Load into new cols
  ts_data_d$Max_depth_m[gps_idx] = deepest
  ts_data_d$Mean_depth_m[gps_idx] = mean_depth
  ts_data_d$Dive[gps_idx] = dives # these signify whether a single depth value exceeds threshold in window around GPS record
  
  ################## WRITE FILES ###################
  cat("\rWriting out files...")
  # Write depth data frame to out file and add to data_list
  fwrite(ts_data_d, gsub(".csv", "_dep.csv", f)) # write out file
  data_list[[i]] = ts_data_d
  
  # Append transformed depth col to original data
  ts_data$Depth_mod = NA
  dep_idx = which(!is.na(ts_data$Depth))
  ts_data$Depth_mod[dep_idx] = new_series
  fwrite(ts_data, file=f)
  
  ################# SUMMARY STATS ##################
  cat("\rCalculating summary stats...")
  tot_time = as.numeric(diff(range(ts_data_d$datetime)))
  max_dist = max(ts_data_d$dist_to_dg_km, na.rm = TRUE)
  max_depth = max(ts_data_d$Depth)
  mean_depth = mean(ts_data_d$Depth)
  div = sum(ts_data_d$Depth>threshold)
  non_div = nrow(ts_data_d)-div
  
  sum_stats[[this_bird]] = round(c(tot_time, max_dist, max_depth, 
                             mean_depth, div, non_div), 2)
  
  cat("\rDone!\n")
}

# Summary stats
cat("\nWriting summary stats...")
sumz = as.data.frame(t(data.frame(sum_stats)))
colnames(sumz) = c('Time Tracked (days)', 'Max Distance Travelled (km)', 'Max Depth (m)', 
                   'Mean Depth (m)', 'Dives (Depth>0.5)', 'Non-dives (Depth<=0.5)')
sumz = cbind(BirdID = rownames(sumz), sumz)
write.csv(sumz, file = '../Data/summary_stats.csv', row.names = FALSE)

# Save all depth data in one file
cat("\nWriting depth data file...")
d_data_df = rbindlist(data_list)
fwrite(d_data_df, file = "../Data/BIOT_DGBP/all_d_data.csv")
cat("\rDONE!")
