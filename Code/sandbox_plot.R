
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


dd_data_df = fread(file = "../Data/BIOT_DGBP/all_d_data.csv")
birdis = unique(dd_data_df$TagID)

## PLOT 3D ##

idd <- 1
location <- dd_data_df[!is.na(`location-lat`) & TagID == birdis[idd]]
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
  addLegend(position = "bottomright", pal = pal, values = location$Depth) 
#%>% addMarkers(lng = lons, lat = lats)

m



#TODO: check how many secs between each GPS and make buffer half of it

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

#set_token("sk.eyJ1IjoibGRzd2FieSIsImEiOiJja28ycTRpY3Qwa3h5Mm5xOTRvbGs0aG0zIn0.aQaEpoi55TYn-iGzM9185Q")
set_token("pk.eyJ1IjoibGRzd2FieSIsImEiOiJja28ycHp4NXQwMnFuMndvMDNsMGk1YjljIn0.N_KnEPytCe44O1677FQrzw")
m1 <- mapdeck(data = loc_data2, style = mapdeck_style("dark")) %>%
  add_pointcloud(
    data = loc_data
    , lon = 'location-lon'
    , lat = 'location-lat'
    , layer_id = 'lpi'
    , tooltip = "Binomial"
    , palette = 'rdylbu'
    , fill_colour = "Depth"
    , legend = TRUE
    , radius = 5
  )
m1

mapshot(m1, file = '../Plots/mapdeck.png')



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




############## CREATING ACCELERATION DEEP LEARNING DATA SET #################
# METHOD 1
atm = proc.time()
idx_range = head(which(!is.na(ts_data$Depth)), -1)[-1] # removes first and last index of depth rows
tst = sapply(idx_range, function(ix, wdw=25, thrshold=0.5) {
  X = ts_data$X[(ix-wdw):(ix+wdw)]
  Y = ts_data$Y[(ix-wdw):(ix+wdw)]
  Z = ts_data$Z[(ix-wdw):(ix+wdw)]
  dive = ts_data$Depth[ix]>thrshold
  return(c(X,Y,Z,dive))
})
fwrite(tst, file = "../Data/centred_acc_train_data.csv")
proc.time() - atm



# METHOD 2
ts_data_e = ts_data[1:125000]
atm = proc.time()
wdw = 250
tst3 = sapply(1:(nrow(ts_data)-wdw), function(ix, thrshold=0.5) {
  X = ts_data$X[ix:(ix+wdw)]
  Y = ts_data$Y[ix:(ix+wdw)]
  Z = ts_data$Z[ix:(ix+wdw)]
  dive = any(na.omit(ts_data$Depth[ix:(ix+wdw)]) > thrshold) # do any dives occur in this window?
  return(c(X,Y,Z,dive))
})
tst3 = t(tst3)
proc.time() - atm

############################################################################



####### GPS INTERPOLATION #######
diffs = gps_idx[-1]-gps_idx[-length(gps_idx)]
mean(diffs) # GPS devices were configured to return a reading every 30s but on average did only...
gaps = which(diffs>60) #
diffs[gaps]/30

ixx = gps_idx[10000:10030]

tsting = cbind(ts_data_d$Time[ixx], ts_data_d$`location-lat`[ixx], ts_data_d$`location-lon`[ixx])
tsting2 = tsting
tsting2[2,2] = tsting2[2,3] = tsting2[5:11,2:3]  = tsting2[17,2] = tsting2[17,3] = NA
tsting3 = cbind(tsting[,1], zoo::na.approx(tsting2[,2:3]))
colnames(tsting) = colnames(tsting2) = colnames(tsting3) = c('Time', 'lat', 'long')
tsting = data.frame(tsting)
tsting2 = data.frame(tsting2)
tsting3 = data.frame(tsting)

# The value at the xth index of diffs is the no. of rows between the xth GPS row and the
# next (the x+1th)

# TODO:
gps_idx = which(!is.na(ts_data_d$`location-lon`))
gps_idx2 = gps_idx
# Calculate differences
diffs = diff(gps_idx2)
# 1. Find gps_indexes between which to interpolate
gaps = which(diffs>60) # indexes of gps_idx with larger gaps after
starts = gps_idx2[gaps]
# 2. Determine number of interpolations to be made between each
steps = floor(diffs[gaps]/30)
# 3. Find hypothetical indexes to interpolate at
int_ix = apply(cbind(starts,steps), 1, function(v) seq(v[1], v[1]+v[2]*30, 30))
# 4. Add these indexes to gps_index (maybe a new vector)
#for (i in 1:length(int_ix)){
#  ins = int_ix[[i]]
#  strt = which(gps_idx2==ins[1])
#  gps_idx2 = append(gps_idx2, ins, strt)[-strt]
#}
#gps_idx2 = unique(gps_idx2) # drop duplicates

## OR:
gps_idx2 = sort(unique(c(unlist(int_ix), gps_idx2)))
#### CHECK ####
#all(diff(gps_idx) > 0) #=> TRUE
#all(diff(gps_idx2) > 0) #=> TRUE
#all(diff(unique(gps_idx3)) > 0) #=> TRUE
#err2 = which(diff(gps_idx2) <= 0)
#err2
#err3 = which(diff(gps_idx3) <= 0)
#err3
#gps_idx2[(err2[1]-2):(err2[1]+3)]
#for (i in 1:length(err2)){
#  print(gps_idx2[(err2[i]-2):(err2[i]+3)])
#}

#for (i in 1:length(err3)){
#  print(gps_idx3[(err3[i]-2):(err3[i]+3)])
#}

################
# 5. Subset data for location interpolation
loc_data_int = ts_data_d[gps_idx2]
# 6. Interpolate GPS rows
loc_cols_int = zoo::na.approx(loc_data_int[,c("location-lat", "location-lon")])
# 7 Push back into main df
ts_data_d2 = ts_data_d
ts_data_d2[gps_idx2, c("location-lat", "location-lon")] = data.frame(loc_cols_int)

# 5. Subset data
# 6. Interpolate GPS values 



tsting[c(2,5,17),c(2,3)]
tsting2[c(2,5,17),c(2,3)]
tsting3[c(2,5,17),c(2,3)]


leaflet(data = tsting) %>% 
  addTiles() %>% 
  #addProviderTiles('Esri.WorldImagery') %>%
  addCircleMarkers(lng = as.numeric(tsting$long), 
                   lat = as.numeric(tsting$lat), 
                   #color = ~pal(Mean_depth_m), 
                   radius = 1) %>% 
  addPolylines(lng = as.numeric(tsting[,'long']), 
               lat = as.numeric(tsting[,'lat']), 
               color = "black",
               weight = 2,
               opacity = .4) #%>%


leaflet(data = tsting2) %>% 
  addTiles() %>% 
  #addProviderTiles('Esri.WorldImagery') %>%
  addCircleMarkers(lng = as.numeric(tsting2[-c(2,5,17),'long']), 
                   lat = as.numeric(tsting2[-c(2,5,17),'lat']), 
                   #color = ~pal(Mean_depth_m), 
                   radius = 1) %>% 
  addPolylines(lng = as.numeric(tsting2[-c(2,5,17),'long']), 
               lat = as.numeric(tsting2[-c(2,5,17),'lat']), 
               color = "black",
               weight = 2,
               opacity = .4) #%>%


leaflet(data = tsting3) %>% 
  #addTiles() %>% 
  addProviderTiles('Esri.WorldImagery') %>%
  addCircleMarkers(lng = as.numeric(tsting3$long), 
                   lat = as.numeric(tsting3$lat), 
                   #color = ~pal(Mean_depth_m), 
                   radius = 1) %>% 
  addPolylines(lng = as.numeric(tsting3$long), 
               lat = as.numeric(tsting3$lat), 
               color = "black",
               weight = 2,
               opacity = .4) #%>%



d_data_loc = d_data_df[!is.na(`location-lat`)]
d_data_loc = d_data_loc[TagID=="ch_gps03_S1"]
#d_data_loc = d_data_loc[Mean_depth_m<0.4]
##############################################################


############# ALL GPS PLOTS ###############
# Load Data
d_data_loc = fread(file = "../Data/BIOT_DGBP/all_gps_data.csv")
# Grab IDs
ids = unique(d_data_loc$TagID)
# Subset of base layers to test
views = c('OpenTopoMap', 'Esri.WorldStreetMap', 'Esri.WorldImagery', 'Esri.OceanBasemap', 'Esri.NatGeoWorldMap')
# Plot all bird tracks over each layer
for (view in views){
  m = leaflet(data = d_data_loc) %>% addProviderTiles(view)# %>% setView(zoom = 18)
  for (i in 1:length(ids)){
    m = addPolylines(m, lng = ~`location-lon`, lat = ~`location-lat`, 
                     data = d_data_loc[TagID==ids[i]], 
                     color = rainbow(length(ids))[i],
                     weight = 2,
                     opacity = .9)
  }
  m = addLegend(m, position = "bottomright", colors = rainbow(length(ids)), labels = ids, title = "Bird")
  mapshot(m, file = sprintf('../Plots/GPS_%s.png', view))
}
#############################################






pal <- colorNumeric(palette = "YlOrRd", domain = d_data_loc$Mean_depth_m, reverse = TRUE)

leaflet(data = d_data_loc) %>% 
  #addTiles() %>% 
  #addProviderTiles(providers$Esri.NatGeoWorldMap) %>%
  #addProviderTiles('Esri.WorldImagery') %>%
  #addProviderTiles(providers$OpenStreetMap.Mapnik) %>%
  #addProviderTiles('Stadia.Outdoors') %>%
  addProviderTiles('OpenTopoMap') %>%
  #addProviderTiles(providers$OpenStreetMap.HOT) %>%
  #addCircleMarkers(lng = d_data_loc$`location-lon`, 
  #                 lat = d_data_loc$`location-lat`, 
  #                 color = ~pal(Mean_depth_m), 
  #                 radius = 1) %>% 
  addPolylines(lng = d_data_loc$`location-lon`, 
               lat = d_data_loc$`location-lat`, 
               color = "red",
               weight = 2,
               opacity = .4) %>%
  addLegend(position = "bottomright", pal = pal, values = d_data_loc$Mean_depth_m, title = "Depth")#, 
            #labFormat = labelFormat(transform = function(x) sort(x, decreasing = TRUE))) 
  #addMarkers(lng = lox$`location-lon`, lat = lox$`location-lat`, icon = birdIcon)
  #%>% addMarkers(lng = lox$`location-lon`, lat = lox$`location-lat`)











### DEEP LEARNING CODE ####
# Get Keras ready
library(keras)
library(data.table)
install_keras()

data <- fread(file = '../Data/centred_acc_train_data.csv', header = TRUE)

# Alternative simple classification MLP with dropout
classify_model <- keras_model_sequential()
# 2048 inputs into 784 hidden units (with dropout to help generalisation)
classify_model %>%
  layer_dense(units = 784, input_shape = c(2048)) %>% # Size of layer
  layer_dropout(rate=0.2) %>%                 # Apply dropout to nodes in layer
  layer_activation(activation = 'relu') %>% # Activation of nodes in layer
  layer_dense(units = 1, activation = 'sigmoid') # final layer is is single softmax output

#compiling the defined model with metric = accuracy and optimiser as adam.
classify_model %>% compile(
  optimizer = 'rmsprop',
  loss = 'binary_crossentropy',
  metrics = c('accuracy')
)

# Fit model
classify_model %>% fit(train_x, train_y, epochs = 200, batch_size = 128)

# Evaluate model
loss_and_metrics <- classify_model %>% evaluate(test_x, test_y, batch_size = 128)





### FLATTENING BASELINE, TRY AFTER LINE 124 OF EXTRACT_DEPTH ###
ts_data_d = ts_data_d[Depth<10]
#k=3000

for (k in c(20, 30, 50, 100, 250, 500, 750, 1000, 1500, 3000, 5000, 10000, 15000, 20000)){
  
  offset <- zoo::rollmedian(ts_data_d$Depth, k=k)
  
  dif <- length(ts_data_d$Depth) - length(offset)
  
  offset <- c(offset, rep(offset[length(offset)], dif))
  
  new_series = ts_data_d$Depth - offset
  
  new_series[new_series<0] = 0
  
  ts_data_d$Depth_mod = new_series

  g2 = ggplot(ts_data_d, aes(x = datetime, y = Depth_mod))  +  
    geom_point(size=0.1) 
  
  ggsave(g2, file=paste0('../Plots/comp/', this_bird, "_alt_plot_", k, "Postv.png"), width = 9, height = 9)
  
}



files = list.files("../Data/BIOT_DGBP/BIOT_DGBP/", pattern = "1.csv", full.names = TRUE)

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
  
  
  ################ Flatten curve ##############
  k = 30
  
  #offset <- zoo::rollmedian(ts_data_d$Depth, k=k)
  
  offset = zoo::rollapply(ts_data_d$Depth, width=k, by=k, FUN=median)
  offset= rep(offset, each=k)

  dif <- length(ts_data_d$Depth) - length(offset)
  
  offset <- c(offset, rep(tail(offset, 1), dif))
  
  new_series = ts_data_d$Depth - offset
  
  new_series[new_series<0] = 0 # negative depth meaningless
  
  #tail(new_series, dif) = 0 # because no dives when device removed
  
  new_series[(length(new_series)-dif):length(new_series)] = 0
  
  ts_data_d$Depth_mod = new_series
  ################################################
  
  # Plot depth time series
  cat("\rPlotting depth time series...")
  threshold = 0.1

  g = ggplot(ts_data_d, aes(x = datetime, y = Depth_mod))  +  
    geom_point(size = 0.1) + 
    geom_hline(yintercept=threshold, color="red", linetype="dashed", size=0.2) +
    geom_hline(yintercept=0.02, color="blue", linetype="dashed", size=0.2) +
    geom_hline(yintercept=0.03, color="green", linetype="dashed", size=0.2) 
    
  #g
  
  ggsave(g, file=paste0('../Plots/New_method/M_', this_bird, "_alt_plot_", k, ".png"), width = 9, height = 9)
}
