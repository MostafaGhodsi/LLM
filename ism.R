# ISM Modelling 
rm(list = ls())
install.packages("xlsx")
install.packages("ISM")
library("ISM")
ISM_Matrix <- read.csv("C:/Users/mooshi/Downloads/Covid-19- ISM/ISM_Matrix_R1.csv",header = TRUE )
ISM(fname=ISM_Matrix,Dir="C:/Users/mooshi/Downloads/Covid-19- ISM/Lynda.R.Statistics.Essential.Training.Full_p30download.com")
?micmac
install.packages("easyAHP")
library("easyAHP")

data=data.frame(maker1=c(2,5,5,6,4,3,7,7,8,8))
#row.names(data)=c("item1","item2","item3")
#AHP_Matrix <- read.csv("C:/Users/mooshi/Downloads/Compressed/Lynda.R.Statistics.Essential.Training.Full_p30download.com/AHP_Matrix_RI.csv",header = TRUE )
q <- easyAHP(data)