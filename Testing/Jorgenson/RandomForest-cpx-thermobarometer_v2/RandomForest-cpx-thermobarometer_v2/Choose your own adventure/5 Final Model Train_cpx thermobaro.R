###############################
#### MAKE THE ACTUAL MODEL ####
###############################

#### INSTALL REQUIRED PACKAGES IF NECESSARY
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

######### READ DATA AND REMOVE FILTERED EXPERIMENTS #######
#Load raw data 
load("input.Rdata")

#### What do you have? ####
#SAME AS #4 or else the SEE you calculated is incorrect!
liq <- c("NoLiquid") #OR Liquid

#Set the oxides
ox <- c("SiO2", "TiO2","Al2O3", "Cr2O3","FeO","MgO", "MnO", "CaO", "Na2O") #Make sure this order stays the same!!
liqox <- c("SiO2", "TiO2","Al2O3", "FeO","MgO", "MnO", "CaO", "Na2O","K2O")

var <- paste0(ox,".cpx") #adds .cpx to the cpx oxides
if(liq == "Liquid") {var <- c(var, paste0(liqox,".liq"))} #adds .liq to the liquid variables

#Hyperparamters
r <- 200                #Number of test/train splits
n.cuts <- 1             #number of random cuts
n.tree <- 201           #number of trees 
m.try <-length(var)*2/3 #this should be 12 for liquid and 6 for no liquid.

#Train pressure model
P_C <- extraTrees(x = input[, var], y = input$P, ntree = n.tree, mtry = m.try, numRandomCuts = n.cuts, numThreads = 8)

#Train temperature model
T_C <- extraTrees(x = input[, var], y = input$T, ntree = n.tree, mtry = m.try, numRandomCuts = n.cuts, numThreads = 8)

#Save models
prepareForSave(P_C)
save(P_C, file = "P_C.Rdata")
prepareForSave(T_C)
save(T_C, file = "T_C.Rdata")

