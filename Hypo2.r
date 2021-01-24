Using R.

lab<-read.csv("/home/sushil/Documents/Assingment/hypo-ass3/LabTAT.csv")
lab
View(lab)
attach(lab)

######### Normality Test #########

shapiro.test(Laboratory.1) # p-value = 0.5508
shapiro.test(Laboratory.2) # p-value = 0.8637
shapiro.test(Laboratory.3) # p-value = 0.4205
shapiro.test(Laboratory.4) # p-value = 0.6619


stacked_data <-stack(lab)
View(stacked_data)
attach(stacked_data)
Anova_result <- aov(values~ind , data = stacked_data)
summary(Anova_result) # p-value = <2e-16 ***
