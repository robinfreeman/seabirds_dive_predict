# Author: Robin Freeman (robin.freeman@ioz.ac.uk) Luke Swaby (lds20@ic.ac.uk)
# Script: extract_depth.R
# Desc: Creates csv files containing only those rows of the parent files with depth data
# Date: Apr 2021

## Imports
suppressMessages(library(data.table))
suppressMessages(library(dplyr))
suppressMessages(library(geosphere))
suppressMessages(library(sp))
suppressMessages(library(sf))

## Global variables
threshold = 0.03

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
  this_bird = unique(ts_data$TagID)  # extract bird ID

  ################### TRIM DATA BEFORE FIRST DEPARTURE AND AFTER LAST RETURN #######
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
  ts_data_loc = ts_data[!is.na(`location-lon`)]
  
  # TODO: REINDEX HERE (I.E. START FROM 1 AGAIN)
  
  # convert date and time to readable format for smaller dsets
  cat("\rConverting times to readable format...")

  datetime_s = paste(ts_data_loc$Date, ts_data_loc$Time)
  ts_data_loc$datetime = as.POSIXct(datetime_s, format = "%d/%m/%Y %H:%M:%S", tz = "GMT")
  ts_data_loc = ts_data_loc %>% select(-c(Date, Time))  # drop obsolete cols
  
  datetime_s = paste(ts_data_d$Date, ts_data_d$Time)
  ts_data_d$datetime = as.POSIXct(datetime_s, format = "%d/%m/%Y %H:%M:%S", tz = "GMT")
  ts_data_d = ts_data_d %>% select(-c(Date, Time))  # drop obsolete cols
  
  ################ INTERPOLATE GPS IN DEPTH #####################
  # Note number of GPS rows and their indexes
  cat("\rInterpolating GPS coords in depth data...")
  
  gps_idx = which(!is.na(ts_data_d$`location-lon`))
  
  #gps_idx = which(!is.na(ts_data_d$`location-lon`))

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
  
  ######## REMOVE BACKGROUND NOISE ########## 
  
  #ts_data_d$Depth_mod = c(abs(diff(ts_data_d$Depth)), 0)
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
  
  # Append transformed depth col to original data
  ts_data$Depth_mod = NA
  dep_idx = which(!is.na(ts_data$Depth))
  ts_data$Depth_mod[dep_idx] = new_series

  ############ MAX/MEAN DEPTH ############
  cat("\rAdding cols containing max/mean depth around GPS records in depth data...")
  
  # New cols for dive profile assignment
  ts_data_d$Max_depth_m = ts_data_d$Mean_depth_m = NA
  #ts_data_d$Dive = NA    # can just analyse depth_mod col later
  
  ## Midpoints
  mid.points = round(zoo::rollmean(gps_idx, 2))
  wdws = c(1, mid.points, nrow(ts_data_d))
  
  deepest = sapply(1:(length(wdws)-1), function(i) max(ts_data_d$Depth_mod[wdws[i]:wdws[i+1]]))
  mean_depth = sapply(1:(length(wdws)-1), function(i) mean(ts_data_d$Depth_mod[wdws[i]:wdws[i+1]]))

  #dives = sapply(deepest, function(x) x>threshold)
  
  # Load into new cols
  ts_data_d$Max_depth_m[gps_idx] = deepest
  ts_data_d$Mean_depth_m[gps_idx] = mean_depth
  #ts_data_d$Dive[gps_idx] = dives # these signify whether a single depth value exceeds threshold in window around GPS record
  
  ################ GPS STATS #################

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
  
  ################## WRITE FILES ###################
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
  mean_depth = mean(ts_data_d$Depth)
  div = sum(ts_data_d$Depth_mod>threshold)
  non_div = nrow(ts_data_d)-div
  
  # Total time
  # Total dist travelled
  # Dives
  
  sum_stats[[this_bird]] = round(c(tot_obs, tot_time, tot_dist, max_dist, 
                                   max_depth, mean_depth, div, non_div), 2)
  
  cat("\rDone!\n")
}

# Summary stats
cat("\nWriting summary stats...")
sumz = as.data.frame(t(data.frame(sum_stats)))
colnames(sumz) = c('Total observations', 'Time Tracked (days)', 'Total Distance Travelled (km)', 'Max Distance Travelled (km)', 'Max Depth (m)', 
                   'Mean Depth (m)', 'Dives', 'Non-dives')
sumz = cbind(BirdID = rownames(sumz), sumz)
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
