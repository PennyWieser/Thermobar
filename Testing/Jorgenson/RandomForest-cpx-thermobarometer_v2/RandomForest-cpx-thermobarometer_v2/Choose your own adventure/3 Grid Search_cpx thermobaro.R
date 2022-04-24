########Grid Search#########
############################
#This script extracts the test dataset using a grid system to maintain 
#uniform sampling across the PT space.
#We extract 200 test/train datasets - this is becasue the largest source of variation in the SEE
#is the extraction of the test/train dataset. So we repeated the model 200 times to get an 
#average and robust SEE.
############################

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

options(scipen = 999)
'%!in%' <- function(x,y)!('%in%'(x,y))

mround <- function(x,base){ 
  base*round(x/base) 
} 

#Load raw data 
load("input.Rdata")

###### SELECT TEST DATASET#####

#Select the sample size for the number of test points
r <- 200                #This controls how many test/train datasets you can have is                        
n <- rep(1, times = r)  

test.ids <- list()      #Make a list to save the testids

#Loop over the number of test points to get the ids for the rows
for(i in 1:length(n)) {
  
  #Set number of test and train experiments
  n.test<- floor(nrow(input)/n[i])
  
  #Make a grid of P and T from which to sample the experiments and produce the test dataset
  T.upper <- round(seq(from = min(input$T)-1, to = max(input$T)+1, length.out = ceiling(n.test^0.5)), digits = 0)[2:ceiling(n.test^0.5)]     #Upper T bounds for the grid
  T.lower <- round(seq(from = min(input$T)-1, to = max(input$T)+1, length.out = ceiling(n.test^0.5)), digits = 0)[1:(ceiling(n.test^0.5)-1)] #Lower T bounds for the grid
  
  P.upper <- round(seq(from = min(input$P)-0.1, to = max(input$P)+0.1, length.out = ceiling(n.test^0.5)), digits = 1)[2:ceiling(n.test^0.5)]     #Upper P bound
  P.lower <- round(seq(from = min(input$P)-0.1, to = max(input$P)+0.1, length.out = ceiling(n.test^0.5)), digits = 1)[1:(ceiling(n.test^0.5)-1)] #Lower P bound
  
  perms <- unique(expand.grid(data.frame(P.lower, T.lower)))  # This gives all the possible grid point combos for the lower points 
  perms$P.upper <- P.upper[match(perms$P.lower, P.lower)]     # Match upper bounds to the lower and add
  perms$T.upper <- T.upper[match(perms$T.lower, T.lower)]     # Repeat for T
  
  #Sample 1 experiment from each of the PT brackets
  sam <- lapply(1:nrow(perms), function(x) which(input$P < perms$P.upper[x] & input$P >= perms$P.lower[x] & input$T < perms$T.upper[x] & input$T >= perms$T.lower[x])) #Creates each grid, which are defined by df perms
  sam[sapply(sam, function(sam) length(sam)==0)] <- NA        # For sam if the length is 0 set as NA
  
  samp <- lapply(1:length(sam), function(x) sample(sam[[x]], size = 1)) # Samples points from each gridspace
  perms$samp <- unlist(samp) #Add sample to perms
  
  #Find the number of experiments in each bracket
  perms$n <- unlist(lapply(1:length(sam), function(x) length(which(is.na(sam[[x]]) == F))))
  
  #Remove brackets from perm which don't have any experiments (or only have a single experiment)
  no.perms <- which(is.na(perms$samp) | perms$n < 2)
  perms <- perms[-no.perms,]
  
  #N.test is the number of unique values of the perms id
  test.ids[[i]] <- perms$samp
  
  #End loop
}

##### FIND THE ID OF THE TRAIN IDS ####
train.ids <- lapply(1:length(test.ids), function(x) {sample(which(1:nrow(input) %!in% test.ids[[x]]), size = (nrow(input) - length(test.ids[[x]])), replace = F)})

#### SAVE TEST AND TRAIN IDS ####
save(test.ids, file = "testids.Rdata")
save(train.ids, file = "trainids.Rdata")




