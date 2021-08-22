# Author: Luke Swaby (lds20@ic.ac.uk)
# Script: ACC_sumstats.R
# Desc: Creates labelled rolling window data sets using summary statistics taken from windows
# of 25Hz z-axis ACC data.
# Date: Aug 2021

## Imports ##

library(data.table)
library(stringr)

## Script ##

wdw = as.numeric(commandArgs(trailingOnly=TRUE))  # take arg from command line


if (length(n) != 1) {
  stop("One argument must be supplied (window)", call.=FALSE)
} 

files = list.files("../Data/BIOT_DGBP/", pattern = "ACC_ch", full.names = TRUE)

threshold = 0.1
n = wdw*25*60
b = 25  # move window by 1s each time

dlist = list()

for (i in 1:length(files)){
  f = files[i]
  
  cat(sprintf("Processing file: %s\n", f))
  
  bird = str_extract(f, "ch_(.+)_S1")
  acc = fread(f, select = c('ix', 'Z', 'Depth_mod'))

  ix = zoo::rollapply(acc$ix, width=n, by=b, FUN=function(x) round(median(x)))
  zz = zoo::rollapply(acc$Z, width=n, by=b, FUN=function(x) c(mean(x), max(x), sum(abs(diff(x))), sum(diff(x))))
  depth = zoo::rollapply(acc$Depth_mod, width=n, by=b, FUN=function(x) max(x, na.rm=TRUE))
  
  acc.red.4min = data.table(ix, zz, depth)
  colnames(acc.red.4min) = c('ix', 'Mean.ACC', 'Max.ACC', 'SAD.ACC', 'SD.ACC', 'Max.depth')
  
  acc.red.4min$Dive = as.numeric(acc.red.4min$Max.depth > threshold)
  acc.red.4min$TagID = bird
  
  dlist[[i]] = acc.red.4min
}

fourminACCtest = rbindlist(dlist)
fwrite(fourminACCtest, file = paste0('../Data/BIOT_DGBP/ACCTest_', wdw, '_mins.csv'))