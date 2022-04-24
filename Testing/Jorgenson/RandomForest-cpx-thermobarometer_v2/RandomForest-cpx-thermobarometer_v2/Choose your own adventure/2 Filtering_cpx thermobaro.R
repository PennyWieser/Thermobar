###### Filtering #####
######################
#Here we filter on the basis of Kd and cations
#we also filter for reasonable SiO2 and P ranges with enough datapoints (currently max 50 kbar)
#this spits out a file called input - this is what the calibration dataset will be
######################

####INSTALL REQUIRED PACKAGES IF NECESSARY
options(java.parameters = "-Xmx4g")
pack1 <- suppressWarnings(require(PerformanceAnalytics))
if(pack1 == FALSE) {install.packages("PerformanceAnalytics");library(PerformanceAnalytics)}
pack2 <- suppressWarnings(require(rJava))
if(pack2 == FALSE) {install.packages("rJava");library(rJava)}
pack3 <- suppressWarnings(require(extraTrees))
if(pack3 == FALSE) {install.packages("extraTrees");library(extraTrees)}
pack4 <- suppressWarnings(require(EnvStats))
if(pack4 == FALSE) {install.packages("EnvStats");library(EnvStats)}
pack5 <- suppressWarnings(require(xlsx))
if(pack5 == FALSE) {install.packages("xlsx");library(xlsx)}
pack6 <- suppressWarnings(require(readxl))
if(pack6 == FALSE) {install.packages("readxl");library(readxl)}
pack7 <- suppressWarnings(require(rstudioapi))
if(pack7 == FALSE) {install.packages("rstudioapi");library(rstudioapi)}
rm(pack1,pack2,pack3,pack4,pack5,pack6, pack7)


setwd(paste(dirname(rstudioapi::getActiveDocumentContext()$path)))

#Load all data
load("raw.Rdata")
dat <- raw

#Make a final input.cats dataframe on which to perform the filtering
cats <- dat[,paste0(c("Si", "Al", "Ti", "Ca", "Na", "K", "Fe", "Mg", "Mn", "Cr", "Ni", "P"), ".cpx")]

#Make a column to decide if the experiment should be removed or not
dat$Rm <- "N"

#Remove experiments that lie outside the mean plus minus standard deviation
kd <- (dat$FeO.cpx/dat$MgO.cpx)/ (dat$FeO.liq/dat$MgO.liq)
dat$kd <- kd
upper.kd <- mean(kd, na.rm = T)+sd(kd, na.rm = T); lower.kd <- mean(kd, na.rm = T)-sd(kd, na.rm = T)
dat$Rm[which(kd>upper.kd | kd<lower.kd)] <- "Y"

#Remove experiments that we decide are too deep for either crustal or mantle models
max.p <- 30
dat$Rm[which(dat$P > max.p)] <- "Y"

#Remove bad data
dat$Rm[which(dat$SiO2.liq < 35)] <- "Y"
dat$Rm[which(dat$K2O.cpx > 1.5)] <- "Y"

#Remove experiments and set final file for input
input <- dat[which(dat$Rm == "N"),]

#Mix data (to avoid the bias linked to the organisation of the data with the most recent at the end of the matrix)
input<- input[sample(seq(1,nrow(input),1), nrow(input)),]
rownames(input) <- NULL

#Write and save data
alldat = dat
save(alldat,file="alldat.Rdata")
save(input, file = "input.Rdata")
write.csv(input,'input.csv')

