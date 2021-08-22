# Author: Robin Freeman (robin.freeman@ioz.ac.uk) Luke Swaby (lds20@ic.ac.uk)
# Script: extract_GPS.R
# Desc: Extracts and cleans raw data from AxyTrek GPS logger
# Date: Aug 2021

## Imports
suppressPackageStartupMessages(library(data.table))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(geosphere))
suppressPackageStartupMessages(library(sp))
suppressPackageStartupMessages(library(sf))
suppressPackageStartupMessages(library(stringr))

## Global variables
threshold = 0.1

chagos = read_sf("../Data/BIOT_DGBP/Chagos_v6_land_simple.shp")
diego.garcia = st_coordinates(tail(chagos$geometry, 1))[, c('X', 'Y')]
colnames(diego.garcia) = c('lon', 'lat')

# Get all Pressure files (containing Pressure/activity data)
cat('\nExtracting data...\n\n')
files = list.files("../Data/BIOT_DGBP/BIOT_DGBP/", pattern = "1.csv", full.names = TRUE)
gps_data_list = list()
depth_data_list = list()
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
  
  # Read in file and subset rows containing depth data
  ts_data = fread(file = f)
  
  # Index
  ts_data$ix = 1:nrow(ts_data)
  ts_data$TagID = str_remove(ts_data$TagID, "_gv[0-9]+_?[0-9]+") # reomove GLS tag
  this_bird = unique(ts_data$TagID)  # extract bird ID

  ########## TRIM DATA BEFORE FIRST DEPARTURE AND AFTER LAST RETURN ###########
  ts_data_loc = ts_data[!is.na(`location-lon`)]
  
  home.or.away = point.in.polygon(ts_data_loc$`location-lon`, ts_data_loc$`location-lat`, 
                                  diego.garcia[,'lon'], diego.garcia[,'lat'], 
                                  mode.checked=FALSE)
  
  start = ts_data_loc$ix[min(which(home.or.away == 0))]  # ix of first departure
  finish = ts_data_loc$ix[max(which(home.or.away == 0))]  # ix of last return
  
  # Trim and subset
  ts_data = ts_data[ix > start & ix < finish]
  ts_data$ix = 1:nrow(ts_data)  # reindex
  ts_data_d = ts_data[!is.na(Depth)]
  ts_data_d = ts_data_d[Depth<10]  # remove outlier

  # convert date and time to readable format for smaller dsets
  cat("\rConverting times to readable format...")
  datetime_s = paste(ts_data_d$Date, ts_data_d$Time)
  ts_data_d$datetime = as.POSIXct(datetime_s, format = "%d/%m/%Y %H:%M:%S", tz = "GMT")
  ts_data_d = ts_data_d %>% select(-c(Date, Time))  # drop obsolete cols
  
  ########################## INTERPOLATE GPS IN DEPTH #########################
  cat("\rInterpolating GPS coords in depth data...")
  
  # Note number of GPS rows and their indexes
  gps_idx = which(!is.na(ts_data_d$`location-lon`))
  diffs = diff(gps_idx)
  
  # Find gps_indexes between which to interpolate and number of interpolations to be made between each
  gaps = which(diffs>60) 
  starts = gps_idx[gaps]
  steps = floor(diffs[gaps]/30)
  
  # Find hypothetical indicies to interpolate at and add these indexes to gps_index
  int_ix = apply(cbind(starts,steps), 1, function(v) seq(v[1], v[1]+v[2]*30, 30))
  gps_idx = sort(unique(c(unlist(int_ix), gps_idx)))
  
  # Interpolate GPS rows
  loc_data_int = ts_data_d[gps_idx]
  loc_cols_int = zoo::na.approx(loc_data_int[,c("location-lat", "location-lon")])
  
  # Push back into main df and re-extract GPS
  ts_data_d[gps_idx, c("location-lat", "location-lon")] = data.frame(loc_cols_int)
  ts_data_loc = ts_data_d[!is.na(`location-lon`)]
  
  ######################### REMOVE BACKGROUND NOISE ###########################
  
  cat("\rTransforming data...")
  k = 30  # window width for smoothing
  
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
  
  # Append transformed depth col to original data
  ts_data$Depth_mod = NA
  dep_idx = which(!is.na(ts_data$Depth))
  ts_data$Depth_mod[dep_idx] = new_series

  ############################# MAX/MEAN DEPTH #################################
  cat("\rAdding cols containing max/mean depth around GPS records in depth data...")
  
  ts_data_d$Max_depth_m = ts_data_d$Mean_depth_m = NA  # New cols for dive profile assignment

  ## Midpoints
  mid.points = round(zoo::rollmean(gps_idx, 2))
  wdws = c(1, mid.points, nrow(ts_data_d))
  
  deepest = sapply(1:(length(wdws)-1), function(i) max(ts_data_d$Depth_mod[wdws[i]:wdws[i+1]]))
  mean_depth = sapply(1:(length(wdws)-1), function(i) mean(ts_data_d$Depth_mod[wdws[i]:wdws[i+1]]))

  # Load into new cols
  ts_data_d$Max_depth_m[gps_idx] = deepest
  ts_data_d$Mean_depth_m[gps_idx] = mean_depth
  ts_data_loc$Max_depth_m = deepest
  ts_data_loc$Mean_depth_m = mean_depth
  
  ################################# GPS STATS #################################
  cat("\rCalculating distances travelled...")
  
  # Determine nest coordinates (as the most common GPS)
  nest_coords = c(getmode(ts_data_loc$`location-lon`), getmode(ts_data_loc$`location-lat`))
  
  # work out dists from nest
  ts_data_loc$dist_to_dg_m = distHaversine(nest_coords, cbind(ts_data_loc$`location-lon`, ts_data_loc$`location-lat`))
  ts_data_loc$dist_to_dg_km = ts_data_loc$dist_to_dg_m/1000
  
  # Initialise cols
  ts_data_loc$dist_moved_m = ts_data_loc$time_diff_s = 0

  n = nrow(ts_data_loc)
  
  # Calculate distance moved and time taken between each data point
  ts_data_loc$dist_moved_m[-1] = distHaversine(cbind(ts_data_loc$`location-lon`[-n], ts_data_loc$`location-lat`[-n]), cbind(ts_data_loc$`location-lon`[-1], ts_data_loc$`location-lat`[-1]))
  ts_data_loc$time_diff_s[-1] = diff(ts_data_loc$datetime)
  
  # Calculate speed
  ts_data_loc$calc_sp_ms = ts_data_loc$dist_moved_m/ts_data_loc$time_diff_s
  
  ############################## WRITE FILES ###################################
  cat("\rWriting GPS file...")
  fwrite(ts_data_loc, gsub(".csv", "_loc.csv", f)) 
  gps_data_list[[i]] = ts_data_loc
  
  cat("\rWriting depth file...")
  # Write depth data frame to out file and add to data_list
  fwrite(ts_data_d, gsub(".csv", "_dep.csv", f)) # write out file
  depth_data_list[[i]] = ts_data_d
  
  cat("\rWriting ACC file...")
  fwrite(ts_data %>% select(ix, X, Y, Z, Depth_mod), file = paste0('../Data/BIOT_DGBP/ACC_', this_bird, '.csv'))
  
  cat("\rCalculating summary stats...")
  tot_obs = nrow(ts_data)
  tot_time = as.numeric(ts_data_d$datetime[nrow(ts_data_d)]-ts_data_d$datetime[1])
  tot_dist = sum(ts_data_loc$dist_moved_m)/1000
  max_dist = max(ts_data_loc$dist_to_dg_km, na.rm = TRUE)
  max_depth = max(ts_data_d$Depth)
  div = sum(deepest>threshold)  # in terms of dive locations instead of individual dives
  non_div = length(deepest)-div
  
  sum_stats[[this_bird]] = round(c(tot_obs, tot_time, tot_dist, max_dist, 
                                   max_depth, div, non_div), 2)
  
  cat("\rDone!\n")
}

# Summary stats
cat("\nWriting summary stats...")
sumz = as.data.frame(t(data.frame(sum_stats)))
colnames(sumz) = c('Total observations', 'Time Tracked (days)', 'Total Distance Travelled (km)', 
                   'Max Distance Travelled from Colony (km)', 'Max Depth (m)', 'Dives', 'Non-dives')
sumz = cbind(TagID = rownames(sumz), sumz)
write.csv(sumz, file = '../Data/BIOT_DGBP/summary_stats.csv', row.names = FALSE)

# Save all GPS data in one file
cat("\rWriting depth data file...")
gps_data_df = rbindlist(gps_data_list)
save(gps_data_df, file = "../Data/BIOT_DGBP/gps_data_df.RData")
fwrite(gps_data_df, "../Data/BIOT_DGBP/all_gps_data.csv")

# Save all depth data in one file
cat("\rWriting depth data file...")
d_data_df = rbindlist(depth_data_list)
fwrite(d_data_df, file = "../Data/BIOT_DGBP/all_d_data.csv")

cat("\rDONE!")
