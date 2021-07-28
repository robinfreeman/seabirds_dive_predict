# Author: Robin Freeman (robin.freeman@ioz.ac.uk) Luke Swaby (lds20@ic.ac.uk)
# Script: plot_GPS.R
# Desc: Load GPS data and plot with EEZ
# Date: Apr 2021

library("ggplot2")
theme_set(theme_bw())
library("sf")
library("rnaturalearth")
library("rnaturalearthdata")
library(data.table)
require(ggmap)
library(dplyr)
library(ggspatial)

# Read in world map
world <- ne_countries(scale = "medium", returnclass = "sf")

#esri_ocean <- paste0('https://services.arcgisonline.com/arcgis/rest/services/Ocean/World_Ocean_Base/MapServer/tile/${z}/${y}/${x}.jpeg')
#esri_ocean <- 'https://services.arcgisonline.com/arcgis/rest/services/Ocean/World_Ocean_Reference/MapServer/tile/${z}/${y}/${x}.jpeg'

# Read in shapefiles
chagos.simp = st_read("../Data/BIOT_DGBP/Chagos_v6_land_simple.shp")
chagos.land = st_read("../Data/BIOT_DGBP/Chagos_v6_land.shp")
chagos = st_read("../Data/BIOT_DGBP/Chagos_v6.shp")
eez = st_read("../Data/BIOT_DGBP/eez_noholes.shp")

# Load predictions from best window size
dive.preds = fread('../Results/IMM_360_xval_predictions.csv', stringsAsFactors = FALSE) 
dive.preds$BirdID = gsub('_gv[0-9]+', '_S1', dive.preds$BirdID)  #TODO: this can be deleted once project is rerun as you made change to filename save in depthtoimmersion to only include GPS tag 
colnames(dive.preds)[1] = 'TagID'

TagIDs = unique(dive.preds$TagID)

# Load depth data
dep.files = list.files("../Data/BIOT_DGBP/BIOT_DGBP/", pattern = paste0('(', paste(TagIDs, collapse = '|'),')', '_dep.csv'), full.names = TRUE)

for (f in dep.files){

  dep.data = fread(f, stringsAsFactors = FALSE)
  dep.data$TagID = str_remove(dep.data$TagID, "_gv[0-9]+_?[0-9]+")
  bird = unique(dep.data$TagID)
  
  # Subset data
  dive.preds.bird = dive.preds[TagID == bird]
  all.data = left_join(dep.data, dive.preds.bird, by=c('ix', 'TagID'))
  loc.data = all.data[!is.na(`location-lon`)]
  
  # for each gps _idx, if there is a prediction within its widndow, then include it AT the gps ix
  gps.idx = which(!is.na(all.data$`location-lon`))
  mid.points = round(zoo::rollmean(gps.idx, 2)) 
  wdws = c(1, mid.points, nrow(all.data))
  
  # Load predictions from window around each GPS point into gps row
  loc.data$Prediction = sapply(1:(length(wdws)-1), function(i){
    
    preds.tmp = unique(na.omit(all.data$Prediction[wdws[i]:wdws[i+1]]))
    
    if (length(preds.tmp) == 0){
      return(NA)
    } else if (length(preds.tmp) == 1){
      return(preds.tmp)
    } else {
      return(1)
    }
  })
  
  # Add dive cols
  loc.data$Dive = as.numeric(loc.data$Max_depth_m > 0.03)
  
  # Add confusion matrix col for plot
  loc.data$ConfMatrix = apply(loc.data %>% select('Dive', 'Prediction'), 1, function(v){
    if (is.na(v[1]) | is.na(v[2])){
      return(NA)
    } else if (v[1] == 1 & v[2] == 1){
      return("TP")
    } else if (v[1] == 1 & v[2] == 0){
      return("FN")
    } else if (v[1] == 0 & v[2] == 1){
      return("FP")
    } else if (v[1] == 0 & v[2] == 0){
      return("TN")
    } 
  })
  
  lox <- loc.data %>% 
    select(`location-lat`, `location-lon`, 'ConfMatrix') %>% 
    filter(ConfMatrix %in% c("FP", "TP", "FN"))
  
  g = ggplot(data = world) +
    #annotation_map_tile(type = esri_ocean) +
    #annotation_map_tile(type = "hikebike") +
    geom_sf(data = eez,  alpha=0.3, linetype = "dashed") +  # MPA
    geom_sf(data = chagos, fill = 'lightgray', colour = 'lightgray') +  # Chagos Land
    geom_sf(data = chagos.land, colour='gray25', fill = 'gray25', size=0.5) +  # Chagos Land
  
    annotation_scale(location = "bl", width_hint = 0.5) + # scale bar
    xlab("Longitude") + ylab("Latitude") + 
    ggtitle("GPS track with dives overlaid", subtitle = bird) +
    #annotate(geom = "text", x = -7, y = 72.5, label = "Gulf of Mexico", 
    #  fontface = "italic", color = "grey22", size = 6) +
    geom_path(data = loc.data, aes(x = `location-lon`, y = `location-lat`), colour='blue3', alpha=.5) + 
    #geom_point(data = lox[Dive == FALSE], aes(x = `location-lon`, y = `location-lat`), colour='red3', shape=17, size=.5) + 
    #geom_point(data = lox[Dive == TRUE], aes(x = `location-lon`, y = `location-lat`), colour='black', shape=2, size=2) + 
    #geom_point(data = lox[Dive == TRUE], aes(x = `location-lon`, y = `location-lat`), colour='green4', shape=17, size=3, alpha = 0.5) +
    geom_point(data = lox[ConfMatrix == "FP"], aes(x = `location-lon`, y = `location-lat`, colour='orange'), shape=1, size=3, alpha = 0.5, stroke=0.8) + 
    geom_point(data = lox[ConfMatrix == "TP"], aes(x = `location-lon`, y = `location-lat`, colour='green4'), shape=1, size=3, alpha = 0.8, stroke=0.8) + 
    geom_point(data = lox[ConfMatrix == "FN"], aes(x = `location-lon`, y = `location-lat`, colour='red'), shape=1, size=3, alpha = 0.8, stroke=0.8) + 
    coord_sf(xlim = range(loc.data$`location-lon`) + c(-0.5, 0.5), ylim = range(loc.data$`location-lat`) + c(-0.5, 0.5), expand = FALSE) +
    scale_color_identity(name = "Metric",
                         breaks = c("green4", "red", "orange"),
                         labels = c("True Positives", "False Negatives", "False Positives"),
                         guide = "legend") #+
    #theme(legend.position = c(0.87, 0.8), legend.background = element_rect(size=0.4, linetype="solid", 
    #                              colour ="black"))
     #theme(panel.background = element_rect(fill = "lightblue")) 
    #theme(panel.grid.major = element_line(color = gray(.65), linetype = "dotted", size = 0.5), panel.background = element_rect(fill = "lightblue1"))
  
  ggsave(g, file=paste0('../Plots/GLS_prediction_track_', bird, ".png"))
}
################################# ALL TRACKS ################################################
gps_data_df = fread("../Data/BIOT_DGBP/all_gps_data.csv")
birds = unique(gps_data_df$TagID)
chagos = st_read("../Data/BIOT_DGBP/Chagos_v6_land_simple.shp")
eez = st_read("../Data/BIOT_DGBP/eez_noholes.shp")
esri_ocean <- 'https://services.arcgisonline.com/arcgis/rest/services/Ocean/World_Ocean_Reference/MapServer/tile/${z}/${y}/${x}.jpeg'


g <- ggplot(data = world) + 
      #annotation_map_tile(type = "osm") +
      geom_sf(data = eez,  alpha=0.3, linetype = "dashed") +  # MPA
    geom_sf(data = chagos, fill = 'lightgray', colour = 'lightgray') + 
      geom_sf(data = chagos, colour='gray25', fill = 'gray25', size=0.5) +
      annotation_scale(location = "bl", width_hint = 0.5) + # scale bar
      xlab("Longitude") + ylab("Latitude") + 
      ggtitle("Exclusive Economic Zone of Chagos") +
      #geom_path(data = gps_data_df, aes(x = `location-lon`, y = `location-lat`), color = TagID, size=0.5) +
      geom_path(data = gps_data_df, aes(x = `location-lon`, y = `location-lat`, colour=TagID), alpha=.6) +
      scale_fill_manual(values=rainbow(length(birds))) +
      annotate(geom = "text", x = 72, y = -4, label = "Exclusive Economic Zone", fontface = "italic", color = "grey22", size = 5, alpha=.5) 
    
#################################################################################

### LEAFLET

pal <- colorNumeric(palette = "YlOrRd", domain = loc.data$Max_depth_m, reverse = TRUE)
  
leaflet(data = loc.data) %>% 
  #addTiles() %>% 
  #addProviderTiles('OpenTopoMap') %>%
  addProviderTiles('Esri.WorldImagery') %>%
  addCircleMarkers(lng = loc.data$`location-lon`, 
                   lat = loc.data$`location-lat`, 
                   color = ~pal(Max_depth_m), 
                   radius = 1) %>% 
  addPolylines(lng = loc.data$`location-lon`, 
               lat = loc.data$`location-lat`, 
               color = "black",
               weight = 2,
               opacity = .4) %>%
  addLegend(position = "bottomright", pal = pal, values = loc.data$Max_depth_m, title = "Depth", 
            labFormat = labelFormat(transform = function(x) sort(x, decreasing = TRUE))) %>%
  addCircleMarkers(lng = lox[Dive==T]$`location-lon`, lat = lox[Dive==T]$`location-lat`, color = 'green')






## GGMAP
cbbox <- make_bbox(lon = loc.data$`location-lon`, lat = loc.data$`location-lat`, f = .1) #from ggmap
sq_map <- get_map(location = cbbox, maptype = "satellite", source = "google")

ggmap(sq_map) + theme_minimal() +
  theme(legend.position = "none") +
  xlab("Longitude") + ylab("Latitude") + 
  ggtitle("World map", subtitle = "BirdID") +
  #annotate(geom = "text", x = -7, y = 72.5, label = "Gulf of Mexico", 
  #  fontface = "italic", color = "grey22", size = 6) +
  geom_path(data = loc.data, aes(x = `location-lon`, y = `location-lat`), colour='red3', linetype=2) + 
  #geom_point(data = lox[Dive == FALSE], aes(x = `location-lon`, y = `location-lat`), colour='red3', shape=17, size=.5) + 
  geom_point(data = lox[Dive == TRUE], aes(x = `location-lon`, y = `location-lat`), colour='green4', shape=17, size=2) + 


  ## google ##
register_google(key = "AIzaSyCAYLkDYh-GyfOKxmSDzW_c7H7AvrdI1qQ")
mapImageData <- get_googlemap(center = c(lon = mean(range(loc.data$`location-lon`)), lat = mean(range(loc.data$`location-lat`))),
                              zoom = 8,
                              #color = 'bw',
                              scale = 1,
                              maptype = "terrain")
ggmap(mapImageData) + 
  geom_point(data = loc.data, aes(x = `location-lon`, y = `location-lat`), colour = "red3",
             alpha = .1,
             size = .1)


## GGoceanmaps ###
library(ggOceanMaps)

dt <- data.frame(lon = c(-30, -30, 30, 30), lat = c(50, 80, 80, 50))

#lonrange=range(loc.data$`location-lon`) + c(-0.5, 0.5)
#latrange = range(loc.data$`location-lat`) + c(-0.5, 0.5)
#dt = data.frame(lon = rep(lonrange, each=2), lat = c(latrange[1], rep(latrange[2], 2), latrange[1]))


basemap(data = dt, bathymetry = TRUE) + 
  geom_polygon(data = transform_coord(dt), aes(x = lon, y = lat), color = "red", fill = NA)




### tmap ##
library(tmap)
hmm <- sf::st_as_sf(loc.data, coords = c("location-lon", "location-lon"))%>% 
  sf::st_set_crs(4326)


diego.garcia = data.frame(c(72.420231, 72.453459, 72.457886, 72.500819, 72.500642, 72.483256, 
                            72.483433, 72.507307, 72.429940, 72.409344, 72.420786, 72.343269), 
                          c(-7.219409, -7.239153, -7.260946, -7.293678, -7.314442, -7.335030, 
                            -7.358431, -7.380312, -7.457893, -7.437472, -7.343578, -7.270662))
colnames(diego.garcia) = c('lon', 'lat')



DT <- data.table(
                 place=c("Finland", "Canada", "Tanzania", "Bolivia", "France"),
                 longitude=c(27.472918, -90.476303, 34.679950, -65.691146, 4.533465),
                 latitude=c(63.293001, 54.239631, -2.855123, -13.795272, 48.603949))
# st_as_sf() ######
# sf version 0.2-7
DT_sf = st_as_sf(DT, coords = c("longitude", "latitude"), 
                 crs = 4326, relation_to_geometry = "field")
# sf version 0.3-4, 0.4-0
DT_sf = st_as_sf(DT, coords = c("longitude", "latitude"), 
                 crs = 4326, agr = "constant")
plot(DT_sf)




## DTA CLIPPING METHOD ###
files = list.files("../Data/BIOT_DGBP/BIOT_DGBP/", pattern = "dep.csv", full.names = TRUE)
f = files[9]

dep_data = fread(f)
loc.data = dep_data[!is.na(`location-lat`)]

chagos = read_sf("../Data/BIOT_DGBP/Chagos_v6_land_simple.shp")
diego.garcia = st_coordinates(tail(chagos$geometry, 1))[, c('X', 'Y')]
colnames(diego.garcia) = c('lon', 'lat')
home.or.away = point.in.polygon(loc.data$`location-lon`, loc.data$`location-lat`
                                , diego.garcia[,'lon'], diego.garcia[,'lat'], 
                                mode.checked=FALSE)

start = loc.data$datetime[min(which(home.or.away == 0))]  # time of first departure
finish = loc.data$datetime[max(which(home.or.away == 0))]  # time of last return


test.lon = head(loc.data[datetime > (start-1) & datetime < finish]$`location-lon`, 30)
test.lat = head(loc.data[datetime > (start-1) & datetime < finish]$`location-lat`, 30)
leaflet(data = loc.data) %>% 
addProviderTiles('Esri.WorldImagery') %>%
addPolylines(lng = test.lon, 
             lat = test.lat, 
             color = "red",
             weight = 2,
             opacity = .4) %>%
  addCircleMarkers(lng = test.lon[1], lat = test.lat[1], color = 'green') %>%
  addCircleMarkers(lng = test.lon[length(test.lon)], lat = test.lat[length(test.lat)], color = 'red')
#///////