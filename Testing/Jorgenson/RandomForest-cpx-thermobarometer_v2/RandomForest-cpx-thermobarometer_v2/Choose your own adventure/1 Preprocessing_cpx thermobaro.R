####PREPROCESSING#####
#####################
#This re-caluclates the calibration dataset oxides as cations, for filtering

#####################
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

###### LOAD RAW DATA #####
load("OxiWeight.Rdata")
raw <- read.csv("cpx_dat.csv") 

raw[is.na(raw)] <- 0
#Change FeO and Fe2O3 in cpx and liquid
raw$FeO.cpx <- raw$FeO.cpx+raw$Fe2O3.cpx*0.89981
raw$FeO.liq <- raw$FeO.liq+raw$Fe2O3.liq*0.89981
raw$Fe2O3.cpx <- NULL
raw$Fe2O3.liq <- NULL

##### CALCULATE MOLAR AND CLINOPYROXENE SPECIFIC SITE CHEMISTRY #####

#Choose oxides
ox <- c("SiO2", "Al2O3", "TiO2", "CaO", "Na2O", "K2O", "FeO", "MgO", "MnO", "Cr2O3", "NiO", "P2O5")

#Extract columns that are only oxides
input <- raw[,paste0(ox, ".cpx")]
colnames(input) <- ox

#The number of oxygens to calculate on the basis of which depends upon the mineral formula you are calculating
OxNum <- 6

#The names of all oxides inside oxiweight
all.ox <- rownames(OxiWeight) 

#Take each oxide and divide by its appropriate atomic weight as taken from the data OxiWeight
molprop <- apply(input, MARGIN = 1, function(x) x / round(OxiWeight[ox, 'OWeight'], digits = 2))

#This is the number of oxygens in the oxide formula for each of the oxides.
Ox.frame <- data.frame(all.ox, c(2,2,3,3,1,1,1,1,1,1,5,2,3,1,1,1,1,1,1,1,2,3,3,0,0,0,1))
colnames(Ox.frame) <- c("Oxide", "N")

#This is the numbers from Ox.frame relevant to the raw data inputted
Ox <- Ox.frame[match(ox, Ox.frame$Oxide), "N"]

#calculate the atomic proportion of oxygen. This is the molar proportion multiplied by the number of oxygens in the oxide
AtPropOx <- apply(molprop, MARGIN = 2, function(x) x * Ox)

#Sum the atomic proportion of oxygen to give the total atomic proportion of oxygen
TotalOx <- apply(AtPropOx, MARGIN = 2, function(x) sum(x))

#Calculate the value to use in the recalculation
OxRecalc <- OxNum / TotalOx

#This is the number used to multiply by for the cations in the final step. This is essentially the number of cations
#per oxygen in the oxide formula.
CatRecalc.frame <- data.frame(all.ox, c(2,2,1.5,1.5,1,1,1,1,0.5,0.5,2.5,2,3,1,1,1,1,0.5,0.5,0.5,2,1.5,1.5,2,1,2,0.5))
colnames(CatRecalc.frame) <- c("Oxide", "N")

#This is the numbers from CatRecalc.frame relevant to the raw data inputted
CatRecalc <- CatRecalc.frame[match(ox, CatRecalc.frame$Oxide), "N"]

#Calculate the number of anions on the basis of the desired number of oxygens
AnionsPerOxNum <- apply(AtPropOx, MARGIN = 1, function(x) x * OxRecalc)

#Calculate the number of cations in the desired formula
cations <- as.data.frame(t(apply(AnionsPerOxNum, MARGIN = 1, function(x) x / CatRecalc)))

#Save a clean copy
cats <- cations

#Change column names to cations only
colnames(cats) <- paste0(c("Si", "Al", "Ti", "Ca", "Na", "K", "Fe", "Mg", "Mn", "Cr", "Ni", "P"), ".cpx")

#Round dataframe 
cats <- as.data.frame(apply(cats, 2, round, digits = 3))
cats$cat.sum <- apply(cats, 1, sum)

#### NORMALISE LIQUIDS TO 100 

#Subset liquids
liq <- raw[,paste0(ox, ".liq")]

#normalise
raw[,paste0(ox,".liq")] <- round(as.data.frame(t(apply(liq,1,function(x){(x/sum(x))*100}))),2)

#Bind together the data
raw <- cbind(raw, cats)
save(raw, file = "raw.Rdata")


