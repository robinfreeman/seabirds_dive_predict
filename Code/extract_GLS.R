# Author: Luke Swaby (lds20@ic.ac.uk)
# Script: extract_GLS.R
# Desc: Extracts and cleans raw data from Intigeo C330 geolocator
# Date: Aug 2021
# TODO: make fread read last line as the transition is important! (just make it dry until the end?)

## Imports
suppressMessages(library(data.table))
suppressMessages(library(splitstackshape))
suppressMessages(library(dplyr))

# Get all GLS files (containing immersion/lux data)
cat('\nExtracting GLS data...\n\n')
dir = "../Data/GLS Data 2019 Jan DG RFB Short-term/matched/"
im_files = list.files(dir, pattern = "T.deg", full.names = TRUE)
lux_files = list.files(dir, pattern = "T.lux", full.names = TRUE)
files = tools::file_path_sans_ext(im_files)

# Debug
if (!all(files %in% tools::file_path_sans_ext(lux_files))){
  stop("There are .deg files with no corresponding .lux file.")
}

res = 6  # resolution (seconds)

for (i in 1:length(files)){
  
  f = files[i]
  this_bird = basename(f)
  
  cat(sprintf("Processing files for: '%s'\n", this_bird))
  
  # Read in files
  im_data = fread(file = paste0(f, '.deg'))
  lux_data = fread(file = paste0(f, '.lux'))
  
  # Convert date and time to readable format
  colnames(im_data)[1] = 'datetime'
  im_data$datetime = as.POSIXct(im_data$datetime, format = "%d/%m/%Y %H:%M:%S")
  colnames(lux_data)[1] = 'datetime'
  lux_data$datetime = as.POSIXct(lux_data$datetime, format = "%d/%m/%Y %H:%M:%S")
  
  # Expand immersion data to 6s resolution
  dur = im_data$duration = im_data$duration/res
  idx = cumsum(dur)
  expnd_data = expandRows(im_data, count = "duration", drop = TRUE)
  
  # Shift wet/dry col back by one
  expnd_data$`wet/dry` = c(expnd_data$`wet/dry`[-1], expnd_data$`wet/dry`[nrow(expnd_data)])
  
  expnd_data$datetime = do.call("c", apply(cbind(idx, dur), 1, FUN = function(v){
    # Expand time series backwards by v[2] seconds from start time 
    # at index = v[1]
    return(rev(expnd_data$datetime[v[1]] - seq(0, length.out = v[2], by = res)))
  }))
  
  # Combine and save
  comb_data = left_join(expnd_data, lux_data, by = 'datetime')
  fwrite(comb_data, paste0(f, "_GLS.csv"))
}