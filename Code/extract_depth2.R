# Author: Robin Freeman (robin.freeman@ioz.ac.uk) Luke Swaby (lds20@ic.ac.uk)
# Script: extract_depth.R
# Desc: Creates csv files containing only those rows of the parent files with depth data
# Date: Apr 2021

suppressMessages(library(geosphere))
suppressMessages(library(data.table))
suppressMessages(library(ggplot2))
suppressMessages(library(gridExtra))
suppressMessages(library(dplyr))

suppressMessages(library(mapview))
suppressMessages(library(leaflet))
suppressMessages(library(webshot))
#webshot::install_phantomjs()


#################### Extract pressure data ####################

cat('\nExtracting pressure data...\n\n')

# Get all Pressure files (containing Pressure/activity data)
files = list.files("../Data/BIOT_DGBP/BIOT_DGBP/", pattern = "1.csv", full.names = TRUE)
data_list = list()

# For each file, extract pressure data and resave as *_p.csv
for (i in 1:length(files)) {
  
  f = files[i]
  cat(sprintf("Processing file: %s\n", f))
  
  cat("Reading in file...")
  
  # Read in file and subset rows containing pressure data
  ts_data = fread(file = f)
  ts_data_d = ts_data[!is.na(Depth)]
  this_bird = unique(ts_data_d$TagID)
  
  cat("\rCalculating distances travelled...")
  
  # convert date and time to readable format
  datetime_s = paste(ts_data_d$Date, ts_data_d$Time)
  ts_data_d$datetime = as.POSIXct(datetime_s, format = "%d/%m/%Y %H:%M:%S.000")
  
  # work out max dist from dg and total distance
  ts_data_d$dist_to_dg_m = distHaversine(c(72.41111, -7.31333), cbind(ts_data_d$`location-lon`, ts_data_d$`location-lat`))
  ts_data_d$dist_to_dg_km = ts_data_d$dist_to_dg_m/1000
  
  # Initialise cols
  ts_data_d$dist_moved_m = ts_data_d$time_diff_s = NA
  
  # Note number of GPS rows and their indexes
  gps_idx = which(!is.na(ts_data_d$`location-lon`))
  
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
  ######################################################
  
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
  
  # Plot depth time series
  cat("\rPlotting depth time series...")
  threshold = 0.5
  #g1 = ggplot(ts_data_d, aes(x = datetime, y = `height-msl`)) +  
  #  geom_point() 
  g = ggplot(ts_data_d, aes(x = datetime, y = Depth))  +  
    geom_point() + 
    geom_hline(yintercept=threshold, color="blue", linetype="dashed") #+
    #geom_hline(yintercept=0.4, color="red", linetype="dashed") + 
    #geom_hline(yintercept=0.3, color="green", linetype="dashed") 
    
  #g = grid.arrange(g1, g2, ncol=1)
  
  #ggsave(g, file=paste0('../Plots/', this_bird, "_alt_plot.png"), width = 9, height = 9)
  
  ### PLOT DEPTH OVER GPS ###
  
  cat("\rPlotting depth over GPS...")
  
  # New cols for dive profile assignment
  ts_data_d$Dive = ts_data_d$Max_depth_m = ts_data_d$Mean_depth_m = NA
  
  wndow = 15
  
  # Assign dive profiles based on deepest depth within 30s buffer of location
  deepest = sapply(gps_idx, function(i) max(ts_data_d$Depth[(i-wndow):(i+wndow)]))
  deepest[is.na(deepest)] = 0 # last value will be NA as there aren't 30 rows past it
  
  mean_depth = sapply(gps_idx, function(i) mean(ts_data_d$Depth[(i-wndow):(i+wndow)]))
  mean_depth[is.na(mean_depth)] = 0 # last value will be NA as there aren't 30 rows past it
  
  dives = sapply(deepest, function(x) x>threshold)
  
  # Load into new cols
  ts_data_d$Max_depth_m[gps_idx] = deepest
  ts_data_d$Mean_depth_m[gps_idx] = mean_depth
  ts_data_d$Dive[gps_idx] = dives
  
  # Write data frame to out file and add to data_list
  fwrite(ts_data_d, gsub(".csv", "_dep.csv", f)) # write out file
  data_list[[i]] = ts_data_d
  
  # Filter GPS coordinates of dives
  loc_data <- ts_data_d[gps_idx]
  lox <- loc_data %>% 
    filter(Dive == TRUE) %>%
    select(`location-lat`, `location-lon`)
  
  # Plot GPS track with mean depth overlaid
  
  #birdIcon <- makeIcon(iconUrl = '../Images/imgbin_computer-icons-project-symbol-png.png', iconWidth = 20, iconHeight = 20)
  #birdIcon <- makeIcon(iconUrl = 'http://www.pngall.com/wp-content/uploads/2017/05/Map-Marker-Free-Download-PNG.png')
  pal <- colorNumeric(palette = "YlOrRd", domain = loc_data$Mean_depth_m, reverse = TRUE)
  
  m <- leaflet(data = loc_data) %>% 
    #addTiles() %>% 
    addProviderTiles('Esri.WorldImagery') %>%
    addCircleMarkers(lng = loc_data$`location-lon`, 
                     lat = loc_data$`location-lat`, 
                     color = ~pal(Mean_depth_m), 
                     radius = 1) %>% 
    addPolylines(lng = loc_data$`location-lon`, 
                 lat = loc_data$`location-lat`, 
                 color = "black",
                 weight = 2,
                 opacity = .4) %>%
    addLegend(position = "bottomright", pal = pal, values = loc_data$Mean_depth_m, title = "Depth", 
              labFormat = labelFormat(transform = function(x) sort(x, decreasing = TRUE))) %>%
    #addMarkers(lng = lox$`location-lon`, lat = lox$`location-lat`, icon = birdIcon)
    addMarkers(lng = lox$`location-lon`, lat = lox$`location-lat`)
  m
  
  
  #pal <- colorNumeric(palette = "YlOrRd", domain = loc_data_INTERPOLATED$Mean_depth_m, reverse = TRUE)
  #m_INTERPOLATED <- leaflet(data = loc_data_INTERPOLATED) %>% 
  #  addProviderTiles('Esri.WorldImagery') %>%
  #  addCircleMarkers(lng = loc_data_INTERPOLATED$`location-lon`, 
  #                   lat = loc_data_INTERPOLATED$`location-lat`, 
  #                   color = ~pal(Mean_depth_m), 
  #                   radius = 1) %>% 
  #  addPolylines(lng = loc_data_INTERPOLATED$`location-lon`, 
  #               lat = loc_data_INTERPOLATED$`location-lat`, 
  #               color = "black",
  #               weight = 2,
  #               opacity = .4) %>%
  #  addLegend(position = "bottomright", pal = pal, values = loc_data_INTERPOLATED$Mean_depth_m, title = "Depth", 
  #            labFormat = labelFormat(transform = function(x) sort(x, decreasing = TRUE))) %>%
  #  #addMarkers(lng = lox$`location-lon`, lat = lox$`location-lat`, icon = birdIcon)
  #  addMarkers(lng = lox_INTERPOLATED$`location-lon`, lat = lox_INTERPOLATED$`location-lat`)
  
  #boxplot(loc_data_INTERPOLATED$Mean_depth_m)
  #boxplot(loc_data_INTERPOLATED$Mean_depth_m[-which(loc_data_INTERPOLATED$Mean_depth_m>0.6)])
  #boxplot(loc_data_INT_no_outliers$Mean_depth_m)
  
  #loc_data_INT_no_outliers = loc_data_INTERPOLATED
  #loc_data_INT_no_outliers$Mean_depth_m[which(loc_data_INTERPOLATED$Mean_depth_m>0.6)] = loc_data_INT_no_outliers$Mean_depth_m[which(loc_data_INTERPOLATED$Mean_depth_m>0.6)]/2
  #pal <- colorNumeric(palette = "YlOrRd", domain = loc_data_INT_no_outliers$Mean_depth_m, reverse = TRUE)
  #m_INTERPOLATED_no_outliers <- leaflet(data = loc_data_INT_no_outliers) %>% 
    #addTiles() %>% 
  #  addProviderTiles('Esri.WorldImagery') %>%
  #  addCircleMarkers(lng = loc_data_INT_no_outliers$`location-lon`, 
  #                   lat = loc_data_INT_no_outliers$`location-lat`, 
  #                   color = ~pal(Mean_depth_m), 
  #                   radius = 1) %>% 
  #  addPolylines(lng = loc_data_INT_no_outliers$`location-lon`, 
  #               lat = loc_data_INT_no_outliers$`location-lat`, 
  #               color = "black",
  #               weight = 2,
  #               opacity = .4) %>%
  #  addLegend(position = "bottomright", pal = pal, values = loc_data_INT_no_outliers$Mean_depth_m, title = "Depth", 
  #            labFormat = labelFormat(transform = function(x) sort(x, decreasing = TRUE))) %>%
  #  #addMarkers(lng = lox$`location-lon`, lat = lox$`location-lat`, icon = birdIcon)
  #  addMarkers(lng = lox_INTERPOLATED$`location-lon`, lat = lox_INTERPOLATED$`location-lat`)
  
  #addAwesomeMarkers(m_INTERPOLATED_no_outliers, lng = lox_INTERPOLATED$`location-lon`, 
  #                  lat = lox_INTERPOLATED$`location-lat`, icon = "arrow-up")
  
  #mapshot(m, file = sprintf("../Plots/depth_mean_%s.png", this_bird))
  
  cat("\rDone!\n")
}

cat("\nWriting out file...")
d_data_df = rbindlist(data_list)
fwrite(d_data_df, file = "../Data/BIOT_DGBP/all_d_data.csv")
cat("\rDONE!")
