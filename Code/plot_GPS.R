# Author: Robin Freeman (robin.freeman@ioz.ac.uk) Luke Swaby (lds20@ic.ac.uk)
# Script: plot_GPS.R
# Desc: Load GPS data and plot with EEZ
# Date: Apr 2021

# Imports 
require('rgeos')
require("rgdal") # requires sp, will use proj.4 if installed
require("maptools")
require("ggplot2")
require("plyr")

# Load data
gps_data_df = read.csv("../Data/BIOT_DGBP/all_gps_data.csv", row.names = 1, as.is=T)

# Change birds IDs to numeric factors
gps_data_df$TagID = factor(as.numeric(factor(gps_data_df$TagID)))

# Convert datetime strings back to datetimes
gps_data_df$datetime = as.POSIXct(gps_data_df$datetime)

# Get EEZ shapefile
eez = readOGR(dsn="../Data/BIOT_DGBP/", layer="eez_noholes")

eez@data$id = rownames(eez@data)
eez.points = fortify(eez, region="id")
eez.df = join(eez.points, eez@data, by="id")

# Get chagos shapefile
chagos = readOGR(dsn="../Data/BIOT_DGBP/", layer="Chagos_v6_land_simple")

chagos@data$id = rownames(chagos@data)
chagos.points = fortify(chagos, region="id")
chagos.df = join(chagos.points, chagos@data, by="id")
write.csv(chagos.df, "../Data/BIOT_DGBP/chagos_df.csv")

chagos.df_land = subset(chagos.df, DEPTHLABEL == "land")

library(ggplot2)

## GPS PLOT ##

g <- ggplot(eez.df, aes(x = long, y = lat, alpha = 0.01)) + 
  geom_polygon() +
  #geom_polygon(data = chagos.df, aes(x = long, y = lat, color = "darkgreen")) + 
  geom_point(data = gps_data_df, aes(x = location.lon, y = location.lat, group = TagID, color = TagID), size=2) +
  geom_path(data = gps_data_df, aes(x = location.lon, y = location.lat, group = TagID), color = "black", size=1) +
  coord_equal() +
  facet_wrap(~TagID)  + 
  theme_bw() + 
  theme(panel.background = element_rect(fill = "lightblue"), legend.position = "none", text = element_text(size=10))

print(g)


## SPEED PLOT ##

birds = unique(gps_data_df$TagID)
for (i in 1:length(birds)) {
  this_bird = birds[i]
  g = ggplot(eez.df, aes(x = long, y = lat, alpha = 0.01)) + 
    geom_polygon() +
    #geom_polygon(data = chagos.df, aes(x = long, y = lat, color = "darkgreen")) + 
    geom_point(data = subset(gps_data_df, TagID == this_bird), aes(x = location.lon, y = location.lat, group = TagID, color = TagID), size=0.5) +
    geom_path(data = subset(gps_data_df, TagID == this_bird), aes(x = location.lon, y = location.lat, group = TagID), color = "black", size=0.5) +
    coord_equal() +
    #geom_text(aes(x = -Inf, y = -Inf, label = max(dist_to_dg_m)), hjust   = -0.1, vjust   = -1) + 
    theme_bw() + 
    theme(panel.background = element_rect(fill = "lightblue"), legend.position = "none")
  #print(g)
  ggsave(g, file=paste0('../Data/BIOT_DGBP/', this_bird, "_plot.png"), width = 9, height = 9)
}

ggplot(subset(gps_data_df, calc_sp_ms < 20), aes(x=calc_sp_ms))+ 
  geom_histogram(binwidth=0.05) + 
  xlab("Speed (m/s)") + ylab("Frequency") + 
  facet_wrap(~TagID) + 
  theme(text = element_text(size=10), axis.text.x = element_text(angle=90, hjust=1))

## SUMMARY TABLE ##

require(kableExtra)

smry = ddply(gps_data_df, .(TagID), dplyr::summarise,
             `Max Dist (km)` = max(dist_to_dg_km),
             `Total Dist (km)` = sum(dist_moved_m)/1000,
             `Number of fixes` = length(dist_to_dg_km),
             `Duration` = tail(datetime, 1) - head(datetime, 1),
             `Mean Speed (m/s)` = mean(calc_sp_ms, na.rm = T)
)

kable(smry, format = "html", digits=2)