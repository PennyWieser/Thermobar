
####### HIGGINS ET AL., 2021 AMPHIBOLE THERMOBAROMETER AND CHEMOMETER ######
####### 0 - 12 kbar; basalt to rhyolite #######
####### written by Oliver Higgins (oliver.higgins@unige.ch) ########

#### STEP 1: INSTALL REQUIRED PACKAGES IF NECESSARY
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

#Set working directory
rm(list = ls())
setwd(paste(dirname(rstudioapi::getActiveDocumentContext()$path)))

#### STEP 2: RECALCULATE CATIONS TO OXIDES #####

#Double check directory is correct
setwd(paste(dirname(rstudioapi::getActiveDocumentContext()$path)))
#Load input spreadsheet
input <- as.data.frame(read_excel(path = "Input.xlsx"))
# input <- load("") # <- alternatively load in an Rdata file if required
#Load oxygen weights datatframe
load("OxiWeight.Rdata")
#Make a list of all possible oxiides that can be dealt with
all.ox <- rownames(OxiWeight)
#Locate oxide columns
headers <- colnames(input)
input.ox <- headers[which(headers %in% paste0("amphibole_",rownames(OxiWeight)))]
ox <- gsub(x = input.ox,pattern = "amphibole_", replacement = "")
rm(headers)
#The number of oxygens to calculate on the basis of which depends upon the mineral formula you are calculating
OxNum <- 23
#Moles of each oxide
molprop <- apply(input[, input.ox], MARGIN = 1, function(x) x / round(OxiWeight[ox, 'OWeight'], digits = 2))
#This is the numbers from Ox.frame relevant to the raw data inputted
Ox <- OxiWeight$Ox[match(ox, all.ox)]
#calculate the atomic proportion of oxygen. This is the molar proportion multiplied by the number of oxygens in the oxide
AtPropOx <- apply(molprop, MARGIN = 2, function(x) x * Ox)
#Sum the atomic proportion of oxygen to give the total atomic proportion of oxygen
TotalOx <- apply(AtPropOx, MARGIN = 2, function(x) sum(x))
#Calculate the value to use in the recalculation
OxRecalc <- OxNum / TotalOx
#Calculate the number of anions on the basis of the desired number of oxygens
AnionsPerOxNum <- apply(AtPropOx, MARGIN = 1, function(x) x * OxRecalc)
CatRecalc <- OxiWeight$Cat[match(ox, all.ox)]
#Calculate the number of cations in the desired formula
cations <- apply(AnionsPerOxNum, MARGIN = 1, function(x) x / CatRecalc)
#Transpose the result to allow simpler calculation of mineral ratios etc
cations <- as.data.frame(t(cations))
#Select the cation names
elems <- OxiWeight$element[match(ox, all.ox)]
colnames(cations) <- elems
#Bind together to make a final input dataframe
dat <- cbind(input, cations)
#calculate cation sum
dat$CatSum <- apply(cations, 1, sum)
#Clean environment (all except dat)
rm(list = ls()[which(ls()!="dat")])

#### STEP 3: LOAD IN THE MODELS ####
setwd("ExtraTrees objects")
invisible(lapply(list.files(),load,.GlobalEnv))
setwd("../../Amphibole (PTX)/")

#### STEP 4: PREDICT PARAMETERS (P,T,X) ####

#Establish cation variables to be used in the PT model
id.cats <- c("Si", "Al", "Ti", "Ca", "Na", "K", "Fe", "Mg", "Mn")
#Run pressure prediction
P <- predict(P_cats_A, newdata = dat[, id.cats], allValues = T)
dat$P <- round(apply(P, 1, median), 1)
dat$P_uncer <- round(apply(P,1,IQR),1)/2
#Run temperature prediction
T <- predict(T_cats_A, newdata = dat[, id.cats], allValues = T)
dat$T <- round(apply(T, 1, median), 0)
dat$T_uncer <- round(apply(T,1,IQR),0)/2

#Establish cation variables to be used in the X model
id.cats <- c("Si", "Al", "Ti", "Mg", "Fe", "Mn", "Ca", "K", "Na", "T")
#Run SiO2 prediction
SiO2_pred <- predict(Si_chem_A, newdata = dat[, id.cats], allValues = T)
dat$SiO2_pred <- round(apply(SiO2_pred, 1, median), 1)
dat$SiO2_uncer <- round(apply(SiO2_pred,1,IQR),1)/2
#Run Al2O3 prediction
Al2O3_pred <- predict(Al_chem_A, newdata = dat[, id.cats], allValues = T)
dat$Al2O3_pred <- round(apply(Al2O3_pred, 1, median), 1)
dat$Al2O3_uncer <- round(apply(Al2O3_pred,1,IQR),1)/2
#Run CaO prediction
CaO_pred <- predict(Ca_chem_A, newdata = dat[, id.cats], allValues = T)
dat$CaO_pred <- round(apply(CaO_pred, 1, median), 1)
dat$CaO_uncer <- round(apply(CaO_pred,1,IQR),1)/2
#Run Na2O prediction
Na2O_pred <- predict(Na_chem_A, newdata = dat[, id.cats], allValues = T)
dat$Na2O_pred <- round(apply(Na2O_pred, 1, median), 1)
dat$Na2O_uncer <- round(apply(Na2O_pred,1,IQR),1)/2
#Run K2O prediction
K2O_pred <- predict(K_chem_A, newdata = dat[, id.cats], allValues = T)
dat$K2O_pred <- round(apply(K2O_pred, 1, median), 2)
dat$K2O_uncer <- round(apply(K2O_pred,1,IQR),2)/2
#Run FeO prediction
FeO_pred <- predict(Fe_chem_A, newdata = dat[, id.cats], allValues = T)
dat$FeO_pred <- round(apply(FeO_pred, 1, median), 1)
dat$FeO_uncer <- round(apply(FeO_pred,1,IQR),1)/2
#Run MgO prediction
MgO_pred <- predict(Mg_chem_A, newdata = dat[, id.cats], allValues = T)
dat$MgO_pred <- round(apply(MgO_pred, 1, median), 1)
dat$MgO_uncer <- round(apply(MgO_pred,1,IQR),1)/2
#Run TiO2 prediction
TiO2_pred <- predict(Ti_chem_A, newdata = dat[, id.cats], allValues = T)
dat$TiO2_pred <- round(apply(TiO2_pred, 1, median), 2)
dat$TiO2_uncer <- round(apply(TiO2_pred,1,IQR),2)/2

#Clean environment (all except dat)
rm(list = ls()[which(ls()!="dat")])

### STEP 5: SAVE AND EXPORT DATA ####
write.xlsx(x = dat, file = "Output.xlsx", row.names = F)

