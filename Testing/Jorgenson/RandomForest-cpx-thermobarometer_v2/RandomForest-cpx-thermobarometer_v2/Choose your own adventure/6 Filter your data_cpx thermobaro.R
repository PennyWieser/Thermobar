################################
##PREPROCESS/FILTER YOUR DATA###
################################
#Same as script 1 and 2 but for your data
#####################

#### INSTALL REQUIRED PACKAGES IF NECESSARY
pack1 <- suppressWarnings(require(readxl))
if(pack1 == FALSE) {install.packages("readxl")}

setwd(paste(dirname(rstudioapi::getActiveDocumentContext()$path)))

#### What do you have? ####
#SAME AS #4 or else the SEE you calculated is incorrect!
liq <- c("NoLiquid") #OR Liquid

###### LOAD YOUR DATA #####
load("OxiWeight.Rdata")
userdat <- read.csv("InputData.csv")

userdat[is.na(userdat)] <- 0

##### CALCULATE MOLAR AND CLINOPYROXENE SPECIFIC SITE CHEMISTRY #####

#Choose oxides
ox <- c("SiO2", "Al2O3", "TiO2", "CaO", "Na2O", "FeO", "MgO", "MnO", "Cr2O3")

#Extract columns that are only oxides
input <- userdat[,paste0(ox, ".cpx")]
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
colnames(cats) <- paste0(c("Si", "Al", "Ti", "Ca", "Na", "Fe", "Mg", "Mn", "Cr"), ".cpx")

#Round dataframe 
cats <- as.data.frame(apply(cats, 2, round, digits = 3))

#Bind together the data
dat <- cbind(userdat, cats)

#Make a column to decide if the experiment should be removed or not
dat$Rm <- "N"

#Remove obviously bad experiments. These should never be added to the model because they are poor
dat$Rm[which(is.na(apply(cats, 1, sum)))] <- "Y"
dat$Rm[which(apply(cats, 1, sum) > 4.04 | apply(cats, 1, sum) < 3.96)] <- "Y"

#Normalise liquids to 100 wt% anhyrous
liq.cols <- grep(pattern = ".liq", x = colnames(dat))
dat[,liq.cols] <- round(as.data.frame(t(apply(dat[,liq.cols],1,function(x){(x/sum(x))*100}))),2)

#You can filter your data on the basis of Kd here
#(conditional on whether liquid was selected)
if(liq == "Liquid") {
upper.kd <- 0.393
lower.kd <- 0.159
dat$kd <- (dat$FeO.cpx/dat$MgO.cpx) / (dat$FeO.liq/dat$MgO.liq)
dat$Rm[which(is.na(dat$kd) == TRUE)] <- "Y"
dat$Rm[which(dat$kd > upper.kd | dat$kd < lower.kd)] <- "Y"
}

#Remove experiments and set final file for input
input.user <- dat[which(dat$Rm == "N"),]

#Save data
save(input.user, file = "YOURDATA.Rdata")



