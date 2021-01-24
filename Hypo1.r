Using R.
cutlet <-read.csv('/home/sushil/Documents/Assingment/hypo-ass3/Cutlets.csv')
cutlet

install.packages("")
library(readxl)
View(cutlet)
attach(cutlet)
colnames(cutlet)

####### Normality Test##########

shapiro.test(Unit.A) # p-value = 0.32

shapiro.test(Unit.B) # p-value = 0.5225

######### Variance Test #######

var.test(Unit.A,Unit.B) # p-value = 0.3136

######### 2 Sample T Test ######

t.test(Unit.A,Unit.B,alternative = "two.sided",conf.level = 0.95,correct = TRUE) # p-value = 0.4723
