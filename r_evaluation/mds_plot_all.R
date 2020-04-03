library(MASS)
library(BiDimRegression)
library(vegan)
rm(list = ls())

#converts similarities to dissimilarities
simtodist <- function(sims, gamma = 1) {
  base <- exp(-gamma * sims)
  base/sum(base)
}

# simtodist <- function(sims, gamma = 1) {
#   #base <- exp(-gamma * sims)
#   base <- 1 - sims
#   #base/sum(base)
# }

num_points = 20

files = list.files('../distributions/points/') # get all filens in this directory
for (file in files){
  #file = files[2]
  data = data.matrix(read.csv(paste('../distributions/points/',file, sep = ''), header = TRUE,check.names=F)) # load data
  
  # plot solution
  x <- data[,1]
  y <- data[,2]
  
  jpeg(paste('plots/original/',file,'.jpg'))
  plot(x, y, xlab="", ylab="", type="n")
  text(x, y, labels = LETTERS[1:nrow(data)], cex=1.5)
  dev.off()
}

bid_reg = c()
files = list.files('../cosines/distance/') # get all filens in this directory
for (file in files){
  
  # for each item in files
  #file = files[25]
  
  data = read.csv(paste('../cosines/distance/',file, sep = ''), header = TRUE,check.names=F) # load data
  
  d = simtodist(data.matrix(data)[0:num_points,0:num_points],gamma=1)
  
  #fit <- cmdscale(d,eig=TRUE, k=2) # k is the number of dim
  fit <- isoMDS(d, k=2) # k is the number of dim
  fit # view results
  
  # plot solution
  x <- fit$points[,1]
  y <- fit$points[,2]
  
  jpeg(paste('plots/distance/',file,'.jpg'))
  plot(x, y, xlab="", ylab="",
       main="", type="n")
  text(x, y, labels = colnames(data)[0:num_points], cex=1.5)
  dev.off()
  
  specs = strsplit(file,'\\.')[[1]][1]
  specs = strsplit(specs,'_')[[1]]
  
  model = specs[1]
  source = specs[2]
  relationship = specs[3]
  sampledAs = specs[5]
  
  # evaluate against cluster1
  points = read.csv('../distributions/points/cluster1.csv')
  points = cbind(points,x,y)
  colnames(points) = c('indepV1','indepV2','depV1','depV2')
  cluster1 = BiDimRegression(points)
  
  # evaluate against cluster2
  points = read.csv('../distributions/points/cluster2.csv')
  points = cbind(points,x,y)
  colnames(points) = c('indepV1','indepV2','depV1','depV2')
  cluster2 = BiDimRegression(points)
  
  # evaluate against shape
  points = read.csv('../distributions/points/shape_20.csv')
  points = cbind(points,x,y)
  colnames(points) = c('indepV1','indepV2','depV1','depV2')
  shape = BiDimRegression(points)
  
  bid_reg = rbind(bid_reg, c(model,source,relationship,sampledAs,
                             cluster1$affine.r, cluster1$affine.pValue,
                             cluster2$affine.r, cluster2$affine.pValue,
                             shape$affine.r, shape$affine.pValue))
}

  colnames(bid_reg) = c('model','source','relationship','sampledAs',
                        'cluster1 r', 'cluster1 p',
                        'cluster2 r', 'cluster2 p',
                        'shape r', 'shape p')
write.csv(bid_reg, 'bidim_output/bidimensional_regression.csv')

