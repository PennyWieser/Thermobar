########################
#####PLUG AND PLAY######
########################

# written by Corin Jorgenson and Oliver Higgins 2021 #

#################################
####### USER INPUT REQUIRED #####
#################################

#### CHOOSE WHICH MODEL YOU WOULD LIKE TO RUN####
liq <- "NoLiquid"
#liq <- "Liquid"

###########################################
####### DO NOT EDIT BEYOND THIS POINT #####
###########################################

#this model has a SEE's precalculated
#you can only modify if you have liquid data or not
#NOTE: you cannot have a dataset with the liquid and no liquid model and use both models you must choose one model 

#### INSTALL REQUIRED PACKAGES IF NECESSARY
pack1 <- suppressWarnings(require(PerformanceAnalytics))
if(pack1 == FALSE) {install.packages("PerformanceAnalytics")}
pack2 <- suppressWarnings(require(rJava))
if(pack2 == FALSE) {install.packages("rJava")}
pack3 <- suppressWarnings(require(extraTrees))
if(pack3 == FALSE) {install.packages("extraTrees")}
pack4 <- suppressWarnings(require(readxl))
if(pack4 == FALSE) {install.packages("readxl")}
pack5 <- suppressWarnings(require(EnvStats))
if(pack5 == FALSE) {install.packages("EnvStats")}
options(java.parameters = "-Xmx4g") #Note: you must run this at the beginning of you session else problems may arise
#if after running this section you get errors then run it again.

setwd(paste(dirname(rstudioapi::getActiveDocumentContext()$path)))

####Load models###
load("P_C_liq.Rdata")  #SEE = 2.7 kbar
load("T_C_liq.Rdata")  #SEE = 44.9 deg C
load("P_C_noliq.Rdata")#SEE = 3.2 kbar
load("T_C_noliq.Rdata")#SEE = 72.5 deg C

#### LOAD INPUT DATA####

YOURDATA <- read.csv("InputData.csv") 
#Note: your data should have the same elements as the InputData.csv file and be in the same naming convention (.cpx and .liq suffix)
#CPX DATA: SiO2.cpx,TiO2.cpx,Al2O3.cpx,Cr2O3.cpx,FeO.cpx,MgO.cpx,MnO.cpx,CaO.cpx,Na2O.cpx
#corresponding liquid data should be in the same row as the cpx pair
#LIQ DATA: SiO2.liq,TiO2.liq,Al2O3.liq,FeO.liq,MgO.liq,MnO.liq,CaO.liq,Na2O.liq,K2O.liq

####Define input data####
if(liq == "NoLiquid") {INPUTDATA <- YOURDATA[,c("SiO2.cpx","TiO2.cpx","Al2O3.cpx","Cr2O3.cpx","FeO.cpx","MgO.cpx","MnO.cpx","CaO.cpx","Na2O.cpx")]} 
if(liq == "Liquid")   {INPUTDATA <- YOURDATA[,c("SiO2.cpx","TiO2.cpx","Al2O3.cpx","Cr2O3.cpx","FeO.cpx","MgO.cpx","MnO.cpx","CaO.cpx","Na2O.cpx","SiO2.liq", "TiO2.liq","Al2O3.liq", "FeO.liq","MgO.liq", "MnO.liq", "CaO.liq", "Na2O.liq","K2O.liq")]}  

####Run the model####
if(liq == "Liquid")   {predP <- predict(P_C_liq, newdata = INPUTDATA, allValues=TRUE)} 
if(liq == "Liquid")   {predT <- predict(T_C_liq, newdata = INPUTDATA, allValues=TRUE)} 

if(liq == "NoLiquid") {predP <- predict(P_C_noliq, newdata = INPUTDATA, allValues=TRUE)} 
if(liq == "NoLiquid") {predT <- predict(T_C_noliq, newdata = INPUTDATA, allValues=TRUE)} 

## extract values and calculate iqr
P_values <- apply(predP,1,median)
T_values <- apply(predT,1,median)
P_IQR <- apply(predP,1,IQR)
T_IQR <- apply(predT,1,IQR)

############SAVE DATA###########
YOURDATA_PT = cbind(YOURDATA, P_values, P_IQR, T_values, T_IQR) #this adds  the estimates to your data 
write.csv(YOURDATA_PT,'OutputData.csv') #saves data as a csv file

#########Optional plots ##############
#scatter plot
plot(YOURDATA_PT$T_values, YOURDATA_PT$P_values, #this calls in the data
     xlim=c(950,1300),ylim=c(0,30),              #this sets the axis limits
     xlab=expression(paste("Predicted T (",degree,"C)")), #this is y label, it needs special functions for the degree symbol
     ylab = "Predicted P (kbar)",                #y axis label
     col = "purple",                          # You can change the colour, R can use HEX colours too if you prefer
     pch  = 19)                                # shapes have number codes, 15 is a square, 17 is a triangle
     
##histogram
hist(YOURDATA_PT$T_values, xlab=expression(paste("Predicted T (",degree,"C)")),main= "", col= "darkred")
hist(YOURDATA_PT$P_values, xlab= "Predicted P (kbar)", main = "",col="darkgreen")


