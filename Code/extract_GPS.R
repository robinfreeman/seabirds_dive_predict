# Author: Robin Freeman (robin.freeman@ioz.ac.uk) Luke Swaby (lds20@ic.ac.uk)
# Script: extract_GPS.R
# Desc: Creates csv files containing only those rows of the parent files with GPS data
# Date: Apr 2021

## Imports ##

suppressMessages(library(geosphere))
suppressMessages(library(data.table))
suppressMessages(library(rgeos))
suppressMessages(library(rgdal)) # requires sp, will use proj.4 if installed
suppressMessages(library(maptools))
suppressMessages(library(ggplot2))
suppressMessages(library(plyr))
suppressMessages(library(kableExtra))
suppressMessages(library(rnaturalearth))
suppressMessages(library(ggspatial))
suppressMessages(library(sf))


## Script ##

#################### Extract GPS data ####################

cat('\nExtracting GPS data...\n\n')

# Get all CSV files (containing GPS/ACC data)
files = list.files("../Data/BIOT_DGBP/BIOT_DGBP/", pattern = "_dep.csv", full.names = TRUE)
data_list = list()

# Function to extract statistical mode
getmode <- function(v) {
  uniqv <- unique(na.omit(v))
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

# For each file, extract just location data and resave as *_loc.csv
for (i in 1:length(files)) {
  
  file = files[i]
  cat(sprintf("Processing file: %s\n", file))
  
  # Read in file and subset rows containing GPS data
  cat("Loading data...")
  ts_data = fread(file = file)
  #save(ts_data, file = gsub(".csv", ".RData", file)) # save RDA file for quicker loading later
  ts_data_loc = ts_data[!is.na(`location-lat`)]
  
  # Convert date and time to readable format
  datetime_s = paste(ts_data_loc$Date, ts_data_loc$Time)
  ts_data_loc$datetime = as.POSIXct(datetime_s, format = "%d/%m/%Y %H:%M:%OS", tz = "GMT")
  
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
  ts_data_loc$time_diff_s[-1] = as.numeric(ts_data_loc$datetime[-1] - ts_data_loc$datetime[-n])
  
  # Calculate speed
  ts_data_loc$calc_sp_ms = ts_data_loc$dist_moved_m/ts_data_loc$time_diff_s
  
  # Write data frame to out file and add to data_list
  cat("\rWriting out file...")
  fwrite(ts_data_loc, gsub(".csv", "_loc.csv", file)) 
  data_list[[i]] = ts_data_loc
  
  cat("\rDone\n")
  
}

gps_data_df = rbindlist(data_list)

# Save data
save(gps_data_df, file = "../Data/BIOT_DGBP/gps_data_df.RData")
fwrite(gps_data_df, "../Data/BIOT_DGBP/all_gps_data.csv")

#################### Plots ####################

cat('\nPlotting GPS data...')

# Change birds IDs to numeric factors
#gps_data_df$TagID = factor(as.numeric(factor(gps_data_df$TagID)))
gps_data_df$TagID = as.factor(gps_data_df$TagID)

# Convert datetime strings back to datetimes
#gps_data_df$datetime = as.POSIXct(gps_data_df$datetime)

# Get EEZ shapefile
eez = readOGR(dsn="../Data/BIOT_DGBP/", layer="eez_noholes", verbose = FALSE)

eez@data$id = rownames(eez@data)
eez.points = fortify(eez, region="id")
eez.df = join(eez.points, eez@data, by="id")

# Get chagos shapefile
chagos = readOGR(dsn="../Data/BIOT_DGBP/", layer="Chagos_v6_land_simple", verbose = FALSE)

chagos@data$id = rownames(chagos@data)
chagos.points = fortify(chagos, region="id")
chagos.df = join(chagos.points, chagos@data, by="id")
fwrite(chagos.df, "../Data/BIOT_DGBP/chagos_df.csv")

chagos.df_land = subset(chagos.df, DEPTHLABEL == "land")

## GPS plot
g <- ggplot(eez.df, aes(x = long, y = lat, alpha = 0.01)) + 
  geom_polygon() +
  #geom_polygon(data = chagos.df, aes(x = long, y = lat, color = "darkgreen")) + 
  geom_point(data = gps_data_df, aes(x = `location-lon`, y = `location-lat`, group = TagID, color = TagID), size=2) +
  geom_path(data = gps_data_df, aes(x = `location-lon`, y = `location-lat`, group = TagID), color = "black", size=1) +
  coord_equal() +
  facet_wrap(~TagID)  + 
  theme_bw() + 
  theme(panel.background = element_rect(fill = "lightblue"), legend.position = "none", text = element_text(size=10))

#print(g)
ggsave(g, file="../Plots/chagos_bp_redfoot_map.png", width = 9, height = 9)



###################################################
world <- ne_countries(scale = "medium", returnclass = "sf")

birds = unique(gps_data_df$TagID)
chagos = st_read("../Data/BIOT_DGBP/Chagos_v6_land_simple.shp")
eez = st_read("../Data/BIOT_DGBP/eez_noholes.shp")

g <- ggplot(data = world) + 
      #annotation_map_tile(type = "osm") +
      geom_sf(data = eez,  alpha=0.3, linetype = "dashed") +  # MPA
      geom_sf(data = chagos, colour='gray25', fill = 'gray25', size=0.5) +
      annotation_scale(location = "bl", width_hint = 0.5) + # scale bar
      xlab("Longitude") + ylab("Latitude") + 
      ggtitle("Exclusive Economic Zone of Chagos") +
      #geom_path(data = gps_data_df, aes(x = `location-lon`, y = `location-lat`), color = TagID, size=0.5) +
      geom_path(data = gps_data_df, aes(x = `location-lon`, y = `location-lat`, colour=TagID), alpha=.6) +
      scale_fill_manual(values=rainbow(length(birds))) +
      annotate(geom = "text", x = 72, y = -4, label = "Exclusive Economic Zone", fontface = "italic", color = "grey22", size = 5, alpha=.5) 

ggsave(g, file="../Plots/all_birds_GPS_plot.png", width = 9, height = 9)
###################################################


cat('\rSaving GPS plots for each bird...')

birds = as.vector(unique(gps_data_df$TagID))
for (i in 1:length(birds)) {
  this_bird = birds[i]
  g = ggplot(eez.df, aes(x = long, y = lat, alpha = 0.01)) + 
    geom_polygon() +
    #geom_polygon(data = chagos.df, aes(x = long, y = lat, color = "darkgreen")) + 
    geom_point(data = gps_data_df[TagID==this_bird], aes(x = `location-lon`, y = `location-lat`, group = TagID, color = TagID), size=0.5) +
    geom_path(data = gps_data_df[TagID==this_bird], aes(x = `location-lon`, y = `location-lat`, group = TagID), color = "black", size=0.5) +
    coord_equal() +
    #geom_text(aes(x = -Inf, y = -Inf, label = max(dist_to_dg_m)), hjust   = -0.1, vjust   = -1) + 
    theme_bw() + 
    theme(panel.background = element_rect(fill = "lightblue"), legend.position = "none")
  #print(g)
  ggsave(g, file=paste0('../Plots/', this_bird, "_plot.png"), width = 9, height = 9)
}

## Speed plot
cat('\rPlotting speed data...')

g = ggplot(subset(gps_data_df, calc_sp_ms < 20), aes(x=calc_sp_ms))+ 
      geom_histogram(binwidth=0.05) + 
      xlab("Speed (m/s)") + ylab("Frequency") + 
      facet_wrap(~TagID) + 
      theme(text = element_text(size=10), axis.text.x = element_text(angle=90, hjust=1))

ggsave(g, file="../Plots/chagos_bp_redfoot_speed_plots.png", width = 9, height = 9)

cat('\rWriting summary table...\n')

## Summary table
smry = ddply(gps_data_df, .(TagID), dplyr::summarise,
             `Max Dist (km)` = max(dist_to_dg_km),
             `Total Dist (km)` = sum(dist_moved_m)/1000,
             `Number of fixes` = length(dist_to_dg_km),
             `Duration` = tail(datetime, 1) - head(datetime, 1),
             `Mean Speed (m/s)` = mean(calc_sp_ms, na.rm = T)
)

invisible(kable(smry, format = "html", digits=2))