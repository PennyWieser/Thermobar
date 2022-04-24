######################
#### DETERMINE SEE####
######################
#we recommend cleaning the global environment before running this code, sometimes if you don't you will get an error with the "times" argument at the end

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

######### READ DATA AND REMOVE FILTERED EXPERIMENTS #######
#Load raw data 
load("input.Rdata")

##### LOAD IN TEST AND TRAIN IDS
load("testids.Rdata")
load("trainids.Rdata")

#### What do you have? ####
#YOU DECIDE#
liq <- c("NoLiquid") #OR Liquid

#Set the oxides
ox <- c("SiO2", "TiO2","Al2O3", "Cr2O3","FeO","MgO", "MnO", "CaO", "Na2O")
liqox <- c("SiO2", "TiO2","Al2O3", "FeO","MgO", "MnO", "CaO", "Na2O","K2O")
var <- paste0(ox,".cpx") #adds .cpx to the cpx oxides
if(liq == "Liquid") {var <- c(var, paste0(liqox,".liq"))} #adds .liq to the liquid variables

#Select hyperparamters
r <- 200                #Number of test/train splits
n.cuts <- 1             #number of random cuts
n.tree <- 201           #number of trees 
m.try <-length(var)*2/3 #this should be 12 for liquid and 6 for no liquid.

#PRESSURE

#Make empty lists to populate
Pred_P_mean <- list(); Pred_P_median <- list(); Pred_P_mode <- list()
Pred_P_all <- list(); Resid_P_mean <- list(); Resid_P_median <- list(); Resid_P_mode <- list()
SEE_P_mean <- list(); SEE_P_median <- list(); SEE_P_mode <- list()
R2_P_mean <- list(); R2_P_median <- list(); R2_P_mode <- list(); model_P <- list()
P_IQR <- list()

for(j in 1:r) {

#Train the pressure model
model_P <- extraTrees(x = input[train.ids[[j]], var], y = input$P[train.ids[[j]]], ntree = n.tree, mtry = m.try, numRandomCuts = n.cuts, numThreads = 8) ; print(j)

#Add pause
# Sys.sleep(time = 0.1)

#Predict the pressure in the testset by taking the mean value
Pred_P_mean[[j]] <- round(apply(predict(model_P, newdata = input[test.ids[[j]], var], allValues = T), 1, mean), digits = 1)

#Predict the pressure in the testset by taking the median value
Pred_P_median[[j]] <- round(apply(predict(model_P, newdata = input[test.ids[[j]], var], allValues = T), 1, median), digits = 1)

#Predict the pressure in the testset by taking the modal value
Pred_P_all[[j]] <- round(predict(model_P, newdata = input[test.ids[[j]], var], allValues = T), digits = 1)
Pred_P_tab <- apply(round(predict(model_P, newdata = input[test.ids[[j]], var], allValues = T), digits = 1), 1, table)
Pred_P_mode[[j]] <- unlist(lapply(Pred_P_tab, function(x) {as.numeric(names(sort(x, decreasing = T)[1]))}))

#Calculate the IQR of the voting distribution
P_IQR[[j]] <- round(apply(predict(model_P, newdata = input[test.ids[[j]], var], allValues = T), 1, IQR), digits = 1)

#Calculate the pressure residual from each of the models
Resid_P_mean[[j]] <- input$P[test.ids[[j]]]-unlist(Pred_P_mean[[j]])
Resid_P_median[[j]] <- input$P[test.ids[[j]]]-unlist(Pred_P_median[[j]])
Resid_P_mode[[j]] <- input$P[test.ids[[j]]]-unlist(Pred_P_mode[[j]])

#Calculate the pressure SEE for each study
SEE_P_mean[[j]] <- round((sum((Pred_P_mean[[j]]-input$P[test.ids[[j]]])^2)/length(input$P[test.ids[[j]]]))^(0.5), digits = 2)
SEE_P_median[[j]] <- round((sum((Pred_P_median[[j]]-input$P[test.ids[[j]]])^2)/length(input$P[test.ids[[j]]]))^(0.5), digits = 2)
SEE_P_mode[[j]] <- round((sum((Pred_P_mode[[j]]-input$P[test.ids[[j]]])^2)/length(input$P[test.ids[[j]]]))^(0.5), digits = 2)

#Calculte the pressure R for each study
R2_P_mean[[j]] <- round(summary(lm(Pred_P_mean[[j]]~input$P[test.ids[[j]]]))$r.squared, digits = 3)
R2_P_median[[j]] <- round(summary(lm(Pred_P_median[[j]]~input$P[test.ids[[j]]]))$r.squared, digits = 3)
R2_P_mode[[j]] <- round(summary(lm(Pred_P_mode[[j]]~input$P[test.ids[[j]]]))$r.squared, digits = 3)

#Add pause
# Sys.sleep(time = 0.1)
rm(model_P)

}

#TEMPERATURE

#Make empty lists to populate
Pred_T_mean <- list(); Pred_T_median <- list(); Pred_T_mode <- list()
Pred_T_all <- list(); Resid_T_mean <- list(); Resid_T_median <- list(); Resid_T_mode <- list()
SEE_T_mean <- list(); SEE_T_median <- list(); SEE_T_mode <- list()
R2_T_mean <- list(); R2_T_median <- list(); R2_T_mode <- list(); model_T <- list()
T_IQR <- list()

for(j in 1:r) {

#Train the pressure model
model_T <- extraTrees(x = input[train.ids[[j]], var], y = input$T[train.ids[[j]]], ntree = n.tree, mtry = m.try, numRandomCuts = n.cuts, numThreads = 8) ; print(j)

#Add pause
# Sys.sleep(time = 0.1)

#Predict the pressure in the testset by taking the mean value
Pred_T_mean[[j]] <- round(apply(predict(model_T, newdata = input[test.ids[[j]], var], allValues = T), 1, mean), digits = 0)

#Predict the pressure in the testset by taking the median value
Pred_T_median[[j]] <- round(apply(predict(model_T, newdata = input[test.ids[[j]], var], allValues = T), 1, median), digits = 0)

#Predict the pressure in the testset by taking the modal value
Pred_T_all[[j]] <- round(predict(model_T, newdata = input[test.ids[[j]], var], allValues = T), digits = 0)
Pred_T_tab <- apply(round(predict(model_T, newdata = input[test.ids[[j]], var], allValues = T), digits = 0), 1, table)
Pred_T_mode[[j]] <- unlist(lapply(Pred_T_tab, function(x) {as.numeric(names(sort(x, decreasing = T)[1]))}))

#Calculate the IQR of the voting distribution
T_IQR[[j]] <- round(apply(predict(model_T, newdata = input[test.ids[[j]], var], allValues = T), 1, IQR), digits = 0)

#Calculate the pressure residual from each of the models
Resid_T_mean[[j]] <- input$T[test.ids[[j]]]-unlist(Pred_T_mean[[j]])
Resid_T_median[[j]] <- input$T[test.ids[[j]]]-unlist(Pred_T_median[[j]])
Resid_T_mode[[j]] <- input$T[test.ids[[j]]]-unlist(Pred_T_mode[[j]])

#Calculate the pressure SEE for each study
SEE_T_mean[[j]] <- round((sum((Pred_T_mean[[j]]-input$T[test.ids[[j]]])^2)/length(input$T[test.ids[[j]]]))^(0.5), digits = 0)
SEE_T_median[[j]] <- round((sum((Pred_T_median[[j]]-input$T[test.ids[[j]]])^2)/length(input$T[test.ids[[j]]]))^(0.5), digits = 0)
SEE_T_mode[[j]] <- round((sum((Pred_T_mode[[j]]-input$T[test.ids[[j]]])^2)/length(input$T[test.ids[[j]]]))^(0.5), digits = 0)

#Calculte the pressure R for each study
R2_T_mean[[j]] <- round(summary(lm(Pred_T_mean[[j]]~input$T[test.ids[[j]]]))$r.squared, digits = 3)
R2_T_median[[j]] <- round(summary(lm(Pred_T_median[[j]]~input$T[test.ids[[j]]]))$r.squared, digits = 3)
R2_T_mode[[j]] <- round(summary(lm(Pred_T_mode[[j]]~input$T[test.ids[[j]]]))$r.squared, digits = 3)

#Add pause
# Sys.sleep(time = 0.1)
rm(model_T)

}

#### SAVE CATION DATA #####
output <- input[unlist(test.ids),]
output$r <- rep(1:r, times = lapply(test.ids, length))
#Add pressure data
output$Pred.P.mean <- unlist(Pred_P_mean)
output$Pred.P.med <- unlist(Pred_P_median)
output$Pred.P.mode <- unlist(Pred_P_mode)
output$Resid.P.mean <- unlist(Resid_P_mean)
output$Resid.P.med <- unlist(Resid_P_median)
output$Resid.P.mode <- unlist(Resid_P_mode)
output$SEE.P.mean <- rep(unlist(SEE_P_mean), times = lapply(test.ids, length))
output$SEE.P.med <- rep(unlist(SEE_P_median), times = lapply(test.ids, length))
output$SEE.P.mode <- rep(unlist(SEE_P_mode), times = lapply(test.ids, length))
output$R2.P.mean <- rep(unlist(R2_P_mean), times = lapply(test.ids, length))
output$R2.P.med <- rep(unlist(R2_P_median), times = lapply(test.ids, length))
output$R2.P.mode <- rep(unlist(R2_P_mode), times = lapply(test.ids, length))
#Add temperature data
output$Pred.T.mean <- unlist(Pred_T_mean)
output$Pred.T.med <- unlist(Pred_T_median)
output$Pred.T.mode <- unlist(Pred_T_mode)
output$Resid.T.mean <- unlist(Resid_T_mean)
output$Resid.T.med <- unlist(Resid_T_median)
output$Resid.T.mode <- unlist(Resid_T_mode)
output$SEE.T.mean <- rep(unlist(SEE_T_mean), times = lapply(test.ids, length))
output$SEE.T.med <- rep(unlist(SEE_T_median), times = lapply(test.ids, length))
output$SEE.T.mode <- rep(unlist(SEE_T_mode), times = lapply(test.ids, length))
output$R2.T.mean <- rep(unlist(R2_T_mean), times = lapply(test.ids, length))
output$R2.T.med <- rep(unlist(R2_T_median), times = lapply(test.ids, length))
output$R2.T.mode <- rep(unlist(R2_T_mode), times = lapply(test.ids, length))

##### SAVE AND EXPORT ####
final <- output
save(final, file = "final.Rdata")
write.csv(final,'final.csv')

#### print the SEE ####
mean(unlist(SEE_P_mean))
mean(unlist(SEE_T_mean))




