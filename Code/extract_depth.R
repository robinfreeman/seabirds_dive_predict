# Author: Robin Freeman (robin.freeman@ioz.ac.uk) Luke Swaby (lds20@ic.ac.uk)
# Script: extract_depth.R
# Desc: Creates csv files containing only those rows of the parent files with depth data
# Date: Apr 2021

suppressMessages(library(geosphere))
suppressMessages(library(data.table))
suppressMessages(library(ggplot2))
suppressMessages(library(gridExtra))

#################### Extract pressure data ####################

cat('\nExtracting pressure data...\n\n')

# Get all Pressure files (containing Pressure/activity data)
files = list.files("../Data/BIOT_DGBP/BIOT_DGBP/", pattern = "1.csv", full.names = TRUE)
data_list = list()

# For each file, extract pressure data and resave as *_p.csv
for (i in 1:length(files)) {
  
  f = files[i]
  cat(sprintf("Processing file: %s\n", f))
  
  # Read in file and subset rows containing pressure data
  ts_data = fread(file = f)
  ts_data_d = ts_data[!is.na(Depth)]
  
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
  
  # Write data frame to out file and add to data_list
  fwrite(ts_data_d, gsub(".csv", "_dep.csv", f)) # write out file
  data_list[[i]] = ts_data_d
}

d_data_df = rbindlist(data_list)

fwrite(d_data_df, file = "../Data/BIOT_DGBP/all_d_data.csv")

#################### Calculate Altitude ####################

cat('\nCalculating altitude from pressure...')

# Function to calculate altutude from pressure and temperature
#calc_height = function(Pressure_0, Pressure, Temp) {
#  rel_p = (Pressure_0/Pressure)
#  temp_K = Temp + 273.15
#  lapse_rate = 0.0065
#  
#  h = ( ( rel_p^(1/5.257) - 1) * (temp_K) )/lapse_rate
#  return(h)
#}

# Add altitude col
#Pressure_at_sea_level = 1013.25
#d_data_df$altitude = calc_height(Pressure_at_sea_level, d_data_df$Pressure, d_data_df$`Temp. (?C)`)

#################### Plot GPS Altitude and Pressure for each bird ####################
cat('\rPlotting GPS altitude and depth data...')

birds = unique(d_data_df$TagID)

for (i in 1:length(birds)) {
  this_bird = birds[i]
  this_data = d_data_df[TagID == this_bird]
  
  g1 = ggplot(this_data, aes(x = datetime, y = `height-msl`)) +  
    geom_point() 
  g2 = ggplot(this_data, aes(x = datetime, y = Depth))  +  
    geom_point() #+ geom_hline(yintercept=Pressure_at_sea_level, color="blue", linetype="dashed")
  #g3 = ggplot(this_data, aes(x = datetime, y = altitude))  +  
  #  geom_point() + geom_hline(yintercept=0, color="blue", linetype="dashed")
  
  #g = grid.arrange(g1, g2, g3, ncol=1)
  g = grid.arrange(g1, g2, ncol=1)
  
  ggsave(g, file=paste0('../Data/BIOT_DGBP/', this_bird, "_alt_plot.png"), width = 9, height = 9)
}

#//////////////////////////////////////////////////////////////////////////////
options(rgl.useNULL = TRUE) # for Mac?
library(rgl)
library(RColorBrewer)
library(leaflet)
library(ggmap)
library(data.table)


d_data_df = fread(file = "../Data/BIOT_DGBP/all_d_data.csv")
birds = unique(d_data_df$TagID)

## PLOT 3D ##

idd <- 1
location <- d_data_df[!is.na(`location-lat`) & TagID == birds[idd]]
#set.seed(1)
#rows <- sample(1:nrow(location), 10000)
#sublocation <- location[rows,]
#nrow(location)

#options(rgl.printRglwidget = TRUE)
#plot3d(location$`location-lat`, location$`location-lon`, -location$Depth,
#       xlab = "Latitude", ylab = "Longitude", zlab = "Altitude",
#       col = brewer.pal(3, "Dark2"),size = 8)

## LEAFLET ##
pal <- colorNumeric(palette = "RdBu", domain = location$Depth)

m <- leaflet(data = location) %>% 
  #addTiles() %>% 
  addProviderTiles('Esri.WorldImagery') %>%
  addCircleMarkers(lng = location$`location-lon`, 
                   lat = location$`location-lat`, 
                   color = ~pal(Depth), 
                   radius = 1) %>% 
  addPolylines(lng = location$`location-lon`, 
               lat = location$`location-lat`, 
               color = "black",
               weight = 2,
               opacity = .4) %>%
  addLegend(position = "bottomright", pal = pal, values = location$Depth) %>%
  addMarkers(lng = lons, lat = lats)

m

m1 <- leaflet() %>% 
  addTiles() %>% 
  addPolylines(lng = location$`location-lon`, 
               lat = location$`location-lat`, 
               color = "#03F",
               weight = 5,
               opacity = 0.5)

m1  # Print the map

## GGMAP ##

register_google(key = "AIzaSyCAYLkDYh-GyfOKxmSDzW_c7H7AvrdI1qQ")
map = get_map(location = c(lon = 74, lat = -6.28275), zoom = 8, 
              maptype = 'roadmap', source = "google")

# 71.87243
library(grid)

ggmap(map) + geom_point(data = location, alpha = 0.25, 
                        aes(x = location$`location-lon`, y = location$`location-lat`, colour = location$Depth)) +
  labs(x = NULL, y = NULL) +
  scale_colour_gradient("Depth", high = "red") +
  #scale_size("Accuracy") + theme_classic() +
  theme(axis.line = element_blank(), #axis.text = element_blank(),
        axis.ticks = element_blank(),
        plot.margin= unit(c(3, 0, 0, 0),"mm"),
        legend.text = element_text(size = 6),
        legend.title = element_text(size = 8, face = "plain"),
        panel.background = element_rect(fill='#D6E7EF'))



#///////////////////////////////////////////////////////////////////////
## Import data ##
birds <- fread('../Data/BIOT_DGBP/all_gps_data.csv', header = TRUE)
nrow(birds)

## Plot GPS coords for each bird ##
ids <- unique(birds$TagID)
bird <- birds %>% filter(TagID == ids[2]) %>% select(`location-lat`, `location-lon`) %>% as_tibble()
locations_sf <- st_as_sf(birds, coords = c("location-lon", "location-lat"), crs=4326)
mapview(locations_sf)


## GOOGLE ##
register_google(key = "AIzaSyCAYLkDYh-GyfOKxmSDzW_c7H7AvrdI1qQ")
#bw_map <- get_googlemap(center = c(71.86435, -5.35088), zoom = 11)
bw_map <- get_googlemap(center = c(71.87243, -6.28275), zoom = 8)

ggmap(bw_map) +
  geom_point(data = birds, aes(x = lon, y = lat, col=id), cex=0.1, pch=3)



library(mapdeck)

set_token("pk.eyJ1IjoiY3l0b3MiLCJhIjoiSHBoWG5VQSJ9.hYFgp0rZwNLvbIff5puV8A")
mapdeck(style = mapdeck_style("dark")) %>%
  add_pointcloud(
    data = combined_data
    , lon = 'Longitude'
    , lat = 'Latitude'
    , layer_id = 'lpi'
    , tooltip = "Binomial"
    , palette = 'rdylbu'
    , fill_colour = "Taxa"
    , legend = TRUE
    , radius = 5
  )



#////////////////////////// SANDBOX /////////////////////////
g3 = ggplot(this_data_GPS, aes(x = datetime, y = Depth))  +  
  geom_point()


# this section should be run with this_data = subsetted depth data for a 
# given bird, and the addMarkers line of the leaflet plot is unhashed and 
# then run after this.
lats = c()
lons = c()
idx.tmp = which(this_data$Depth>2.2) # grab indexes of all points deeper than 2.2
for (i in 1:length(idx.tmp)){
  q = idx.tmp[i]
  tmp = this_data[(q-30):(q+30)] # grab section around to find nearest GPS point
  gps.tmp = which(!is.na(tmp$`location-lat`))
  if (length(gps.tmp)){
    row = gps.tmp[1]
    lats = c(lats, tmp[row,]$`location-lat`)
    lons = c(lons, tmp[row,]$`location-lon`)
  }
}

# WHERE THIS_DATA IS SUBSETTED DEPTH DATA FOR A SINGLE BIRD, ASSIGN DIVES
# LIKE SO:

# initialise new cols
this_data$Dive = this_data$Max_depth_m = NA

gps.idx = which(!is.na(this_data$`location-lat`))
#length(gps.idx)

threshold = 2.2

deepest = sapply(gps.idx, function(i) max(this_data$Depth[(i-30):(i+30)]))
deepest[is.na(deepest)] = 0 # last value will be NA as there aren't 30 rows past it
dives = sapply(deepest, function(x) x>threshold)

# Load into new cols
this_data$Max_depth_m[gps.idx] = deepest
this_data$Dive[gps.idx] = dives


min(sumsts)
max(sumsts)
mean(sumsts)


