# Author: Luke Swaby (lds20@ic.ac.uk)
# Script: depth_to_IMM.R
# Desc: Matches GLS files to corresponding depth data
# Date: Aug 2021
# TODO: make fread read last line as the transition is important! (just make it dry until the end?)

suppressPackageStartupMessages(library(data.table))
suppressMessages(library(dplyr))
suppressMessages(library(stringr))

f = '../Data/GLS Data 2019 Jan DG RFB Short-term/gps_birdno.txt'
id_map = fread(file = f, header = TRUE, fill = TRUE)

# import gps IDs
gps_files =  list.files("../Data/BIOT_DGBP/BIOT_DGBP/", pattern = "1.csv", full.names = FALSE)
gps_ids = tools::file_path_sans_ext(gps_files)
gps_ids = as.numeric(str_extract(gps_ids, "[0-9]{2}"))

# import immersion ids
dir = "../Data/GLS Data 2019 Jan DG RFB Short-term/matched/"
im_files = list.files(dir, pattern = "T.deg", full.names = FALSE)
im_ids = tools::file_path_sans_ext(im_files)
lux_files = list.files(dir, pattern = "T.lux", full.names = FALSE)
lux_ids = tools::file_path_sans_ext(lux_files)
gls_ids = im_ids[im_ids %in% lux_ids]  # filter out ids for which there is no light data
gls_ids = tolower(str_match(im_ids, "_(.+?)_")[,2])

# match
full_dta = id_map[GPS_No %in% gps_ids & RingNo %in% gls_ids]
cat(sprintf("%d ID matches found in '%s'\n\n", nrow(full_dta), basename(f)))
full_dta$GPS_No = str_pad(full_dta$GPS_No, 2, pad = "0")

gls_files = list.files(dir, pattern = "_GLS.csv", full.names = TRUE)
dep_files = list.files('../Data/BIOT_DGBP/BIOT_DGBP/', pattern = "_dep.csv", full.names = TRUE)

# Merge and write out files 
outdir = '../Data/GLS Data 2019 Jan DG RFB Short-term/matched/'

for (i in 1:nrow(full_dta)){
  gps.id = paste0('ch_gps', full_dta$GPS_No[i])
  gls.id = toupper(full_dta$RingNo[i])
  
  cat(sprintf("Processing files for: '%s' (%s)...\n", gps.id, gls.id))
  
  # Load data
  gls.f =  gls_files[grepl(gls.id, gls_files, fixed=TRUE)]
  gps.f = dep_files[grepl(gps.id, dep_files, fixed=TRUE)]
  dta.gls = fread(file = gls.f)
  dta.gps = fread(file = gps.f, select = c("ix", "datetime", "Depth", "Depth_mod"))

  # Expand light level resolution to 1s
  #strt = min(which(!is.na(dta.gls$`light(lux)`)))
  #dta.gls = dta.gls[strt:nrow(dta.gls),]  # drop all rows before first light
  #dta.gls$`light(lux)` = zoo::na.locf(dta.gls$`light(lux)`)

  # Merge data
  dta_join = left_join(dta.gps, dta.gls, by = "datetime")
  dta_join = dta_join[min(which(!is.na(dta_join$`wet/dry`))):max(which(!is.na(dta_join$`wet/dry`)))]
  
  # Reduce depth resolution to 6s
  imm.ix = which(!is.na(dta_join$`wet/dry`))
  mid.points = round(zoo::rollmean(imm.ix, 2))
  wdws = c(1, mid.points, nrow(dta_join))
  deepest = sapply(1:(length(wdws)-1), function(i) max(dta_join$Depth_mod[wdws[i]:wdws[i+1]]))
  dta_join = dta_join %>% filter(!is.na(`wet/dry`)) %>% select(-c('Depth'))
  dta_join$Depth_mod = deepest
  
  # Write out file
  gps.id = paste0(gps.id, '_S1')
  fwrite(dta_join, file = paste0(outdir, 'IMM_', gps.id,'.csv'))
}

cat('\nDone!\n')
