rm# Author: Luke Swaby (lds20@ic.ac.uk)
# Script: plot_depth.R
# Desc: Plots depth data
# Date: May 2021

## Imports
suppressMessages(library(data.table))
suppressMessages(library(mapview))
suppressMessages(library(leaflet))
suppressMessages(library(webshot))
suppressMessages(library(ggplot2))
suppressMessages(library(gridExtra))
suppressMessages(library(dplyr))

cat('\nPLOTTING DEPTH DATA...\n\n')

# Get all Pressure files (containing Pressure/activity data)
files = list.files("../Data/BIOT_DGBP/BIOT_DGBP/", pattern = "_dep.csv", full.names = TRUE)

for (i in 1:length(files)) {
  
  f = files[i]
  cat(sprintf("Processing file: %s\n", f))
  
  cat("Reading in file...")
  

  # Read in file and subset rows containing pressure data
  ts_data_d = fread(file = f)
  this_bird = unique(ts_data_d$TagID)
  
  ######### PLOT DEPTH TIME-SERIES ###########
  cat("\rPlotting depth time series...")
  
  g1 = ggplot(ts_data_d, aes(x = datetime, y = Depth))  +  
    geom_point(size=0.2) + 
    ggtitle("Depth time-series (raw)")
  
  g2 = ggplot(ts_data_d, aes(x = datetime, y = Depth_mod))  +  
    geom_point(size = 0.2) + 
    ggtitle("Depth time-series (transformed)") 
  
  g = grid.arrange(g1, g2, ncol=1)
  
  ggsave(g, file=paste0('../Plots/', this_bird, "_alt_plot_COMP.png"), width = 9, height = 9)
  
  ###### PLOT GPS TRACK WITH MEAN DEPTH OVERLAID ##########
  cat("\rPlotting GPS with depth overlaid...")
  
  # Filter GPS coordinates of dives
  gps_idx = which(!is.na(ts_data_d$`location-lon`))
  
  loc_data <- ts_data_d[gps_idx]
  lox <- loc_data %>% 
    filter(Dive == TRUE) %>%
    select(`location-lat`, `location-lon`)
  
  # Plot GPS track with mean depth overlaid
  pal <- colorNumeric(palette = "YlOrRd", domain = loc_data$Mean_depth_m, reverse = TRUE)
  
  m <- leaflet(data = loc_data) %>% 
    #addTiles() %>% 
    #addProviderTiles('OpenTopoMap') %>%
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
    addMarkers(lng = lox$`location-lon`, lat = lox$`location-lat`)
  
  mapshot(m, file = sprintf("../Plots/depth_mean_%s.png", this_bird))
  
  cat("\rDone!\n")
}
