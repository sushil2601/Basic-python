Using R.
buyer<-read.csv('/home/sushil/Documents/Assingment/hypo-ass3/BuyerRatio.csv')
buyer
View(buyer)
attach(buyer)
stacked <- stack(lapply(buyer,as.integer))
stacked
attach(stacked)
View(stacked)
table(values,ind)

t<-prop.table(table(ind))
t
t6<-table(values)
t6
chisq.test(table(values,ind)) # p-value = 0.297
