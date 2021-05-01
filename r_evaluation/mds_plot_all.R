library(MASS)
library(BiDimRegression)
library(vegan)
rm(list = ls())

# global parameter, number of points in map
num_points = 20

#converts similarities to dissimilarities
simtodist <- function(sims, gamma = 1) {
  base <- exp(-gamma * sims)
  base/sum(base)
}

# plot a predicted distribution of cities, marking offset from original plot
plot_svg <- function(target, sample, data, folder, file) {
  
  rotated = procrustes(target, sample)
  
  svg(paste('plots/',folder,file,'.svg', sep = ''), width = 6, height = 6)
  plot(rbind(rotated$Yrot[,1],target[,1]), rbind(rotated$Yrot[,2],target[,2]), xlab="", ylab="",
       main="", type="n",cex = 1.5)
  segments(target[,1],target[,2],rotated$Yrot[,1],rotated$Yrot[,2], col = 'gray70', lwd = 2.5)
  text(rotated$Yrot[,1], rotated$Yrot[,2], labels = colnames(data)[0:num_points], cex=1.65)
  dev.off()
}

# plot the original distributions
files = list.files('../distributions/points/') # get all files in this directory
for (file in files){
  #file = files[3]
  data = data.matrix(read.csv(paste('../distributions/points/',file, sep = ''), header = TRUE,check.names=F)) # load data
  
  # plot solution
  x <- data[,1]
  y <- data[,2]
  
  svg(paste('plots/original/',strsplit(file,'[.]')[[1]][1],'.svg'), width = 6, height = 6)
  plot(x, y, xlab="", ylab="", type="n")
  text(x, y, labels = LETTERS[1:nrow(data)], cex=1.65)
  dev.off()
}

# Perform bidimensional regression and plot all outputs for 
bid_reg = c()
files = list.files('../cosines/uniform/') # get all files in this directory
for (file in files){
  
  # for each item in files
  #file = files[1]
  
  data = read.csv(paste('../cosines/uniform/',file, sep = ''), header = TRUE,check.names=F) # load data
  
  d = simtodist(data.matrix(data)[0:num_points,0:num_points],gamma=1)
  
  fit <- isoMDS(d,k=2) # k is the number of dim
  #fit # view results
  
  # check which distribution
  if (grepl('cluster1',file)){
    original = read.csv('../distributions/points/cluster1_20.csv')
  }
  if (grepl('cluster2',file)){
    original = read.csv('../distributions/points/cluster2_20.csv')
  }
  if (grepl('shape',file)){
    original = read.csv('../distributions/points/shape_20.csv')
  }
  
  points = cbind(original,fit$points)
  colnames(points) = c('indepV1','indepV2','depV1','depV2')
  out = BiDimRegression(points)
  vals = strsplit(strsplit(file,'[.]')[[1]][1],'_')[[1]]
  bid_reg = rbind(bid_reg, c('uniform',vals[1:2], paste(vals[3:length(vals)],collapse='_'),out$affine.r, out$affine.pValue, out$affine.rsqr))
  
  plot_svg(original, fit$points, data, 'uniform/', strsplit(file,'[.]')[[1]][1])
  
}

files = list.files('../cosines/distance/') # get all filens in this directory
for (file in files){
  
  # for each item in files
  # file = files[1]
  
  data = read.csv(paste('../cosines/distance/',file, sep = ''), header = TRUE,check.names=F) # load data
  
  d = simtodist(data.matrix(data)[0:num_points,0:num_points],gamma=1)
  
  fit <- isoMDS(d, k=2) # k is the number of dim
  
  # check which distribution
  if (grepl('cluster1',file)){
    original = read.csv('../distributions/points/cluster1_20.csv')
  }
  if (grepl('cluster2',file)){
    original = read.csv('../distributions/points/cluster2_20.csv')
  }
  if (grepl('shape',file)){
    original = read.csv('../distributions/points/shape_20.csv')
  }
  
  points = cbind(original,fit$points)
  colnames(points) = c('indepV1','indepV2','depV1','depV2')
  out = BiDimRegression(points)
  vals = strsplit(strsplit(file,'[.]')[[1]][1],'_')[[1]]
  bid_reg = rbind(bid_reg, c('distance',vals[1:2], paste(vals[3:length(vals)],collapse='_'),out$affine.r, out$affine.pValue, out$affine.rsqr))
  
  plot_svg(original, fit$points, data, 'distance/', strsplit(file,'[.]')[[1]][1])
  
}

colnames(bid_reg) = c('sampling','model','distribution','corpus','r','p','r^2')
bid_reg2 = bid_reg[as.numeric(bid_reg[,4]) < .05,]
write.csv(bid_reg, 'bidim_output/bidimensional_regression.csv')
write.csv(bid_reg2, 'bidim_output/bidimensional_regression_sig_p.csv')


# get best performing model for LSA
bid_reg = as.data.frame(bid_reg)
test = bid_reg[grepl('lsa',bid_reg$model),]
test$r = as.numeric(test$r)
max_lsa = as.data.frame(aggregate(test$r,by=list(test$model, test$sampling), FUN=mean))
lsa = max_lsa[which.max(max_lsa[max_lsa$Group.2 == 'distance',]$x),]$Group.1


test = bid_reg[!grepl('lsa',bid_reg$model) | grepl(lsa,bid_reg$model),]
write.csv(test, 'bidim_output/bidimensional_regression.csv')
