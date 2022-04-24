########################
#####Run the model######
########################
#Load your data in! The SEE for this is what you calculated in #4
#######################

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

####load models####
load("P_C.Rdata")
load("T_C.Rdata")

#### What do you have? ####
#SAME AS #4 or else the SEE you calculated is incorrect!
liq <- c("NoLiquid") #OR Liquid

#Load input data
load(file = "YOURDATA.Rdata")
YOURDATA <- input.user

####Load in your data####
#Filter your data using sheet 1 and 2 before this step. 
INPUTDATA <- YOURDATA[,c("SiO2.cpx","TiO2.cpx","Al2O3.cpx","Cr2O3.cpx","FeO.cpx","MgO.cpx","MnO.cpx","CaO.cpx","Na2O.cpx")] #make sure the elements are the same and they have the same naming convention 
if(liq== "Liquid"){INPUTDATA <- YOURDATA[,c("SiO2.cpx","TiO2.cpx","Al2O3.cpx","Cr2O3.cpx","FeO.cpx","MgO.cpx","MnO.cpx","CaO.cpx","Na2O.cpx","SiO2.liq", "TiO2.liq","Al2O3.liq", "FeO.liq","MgO.liq", "MnO.liq", "CaO.liq", "Na2O.liq","K2O.liq")]}  

####Run the models ####
predP <- predict(P_C, newdata = INPUTDATA, allValues=TRUE) #this applies the model to your data 
predT <- predict(T_C, newdata = INPUTDATA, allValues=TRUE) #this applies the model to your data 

#Create dataframe to save results
OUTPUTDATA <- YOURDATA

#Calculations for pressure (mean, median, mode IQR)
P_mean <- round(apply(predP, 1, mean), digits = 1); OUTPUTDATA$P_mean <- P_mean
P_median <- round(apply(predP, 1, median), digits = 1); OUTPUTDATA$P_median <- P_median
P_tab <- apply(round(predP, digits = 1), 1, table)
P_mode <- unlist(lapply(P_tab, function(x) {as.numeric(names(sort(x, decreasing = T)[1]))})); OUTPUTDATA$P_mode <- P_mode
P_IQR <- round(apply(predP, 1, IQR), digits = 1); OUTPUTDATA$P_IQR <- P_IQR

#Calculations for temperature (mean, median, mode IQR)
T_mean <- round(apply(predT, 1, mean), digits = 0); OUTPUTDATA$T_mean <- T_mean
T_median <- round(apply(predT, 1, median), digits = 0); OUTPUTDATA$T_median <- T_median
T_tab <- apply(round(predT, digits = 0), 1, table)
T_mode <- unlist(lapply(T_tab, function(x) {as.numeric(names(sort(x, decreasing = T)[1]))})); OUTPUTDATA$T_mode <- T_mode
T_IQR <- round(apply(predT, 1, IQR), digits = 0); OUTPUTDATA$T_IQR <- T_IQR

#Create final output dataframe
write.csv(OUTPUTDATA,'OutputData.csv') #save it! :)
#P out outputs are in kbar, T outputs are in celsius


