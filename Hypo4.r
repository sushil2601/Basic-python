Using R.
customerform<-read.csv('/home/sushil/Documents/Assingment/hypo-ass3/Costomer+OrderForm.csv')
customerform
View(customerform)
stacked <- stack(lapply(customerform,as.character))
stacked
View(stacked)
attach(stacked)
table(values,ind)

t1<-prop.table(table(ind))
t1
t2<-table(values)
t2
chisq.test(table(values,ind)) # p-value = 0.2771
