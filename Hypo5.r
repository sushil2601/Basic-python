fantaloon<-read.csv('/home/sushil/Documents/Assingment/hypo-ass3/Faltoons.csv')
fantaloon
View(fantaloon)
attach(fantaloon)
stacked<-stack(lapply(fantaloon,as.character))
stacked
View(stacked)
attach(stacked)

table(values,ind)
t7<-prop.table(table(ind))
t7
t8<-table(values)
t8
chisq.test(table(values,ind)) # p-value = 8.543e-05
