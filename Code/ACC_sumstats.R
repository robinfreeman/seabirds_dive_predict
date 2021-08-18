#!/usr/bin/env Rscript

library(data.table)
library(stringr)

n = commandArgs(trailingOnly=TRUE)


if (length(n) != 1) {
  stop("One argument must be supplied (window)", call.=FALSE)
} 

files = list.files("../Data/BIOT_DGBP/", pattern = "ACC_ch", full.names = TRUE)

threshold = 0.1
#n = 6000  # no of ACC rows in 4 min   , 1500, 3000
n = as.numeric(n)
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
fwrite(fourminACCtest, file = paste0('../Data/BIOT_DGBP/ACCTest_', n/(25*60), '_mins.csv'))