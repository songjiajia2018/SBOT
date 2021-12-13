
####################### Random Forest #########################

### 1.data process

label=read.table("label1.csv",header=T,sep=",")
label=label$recurrence
data$label=label
data=data[!is.na(data$label),]
data1=data[data$label==1,]
data2=data[data$label==0,]
index=sample(1:59,153,replace=T)
data11=data1[index,]
data=rbind(data1,data11,data2)
data=data[,1:(dim(data)[2]-1)]

library(missForest)
zs<-missForest(d)
save(zs,file="z.Rdata")
data=zs$ximp[1:424,]
dim(zs$ximp)
data=as.data.frame(data)
data$label=c(rep(1,212),rep(0,212))
write.table(data,'first.txt',sep = '\t', quote=FALSE,
            col.names = TRUE,row.names = FALSE)
data_out<-zs$ximp[425:524,]
label2<-read.table("label2.csv",header=T,sep=",")
label2=label2$recurrence
data_out$label=label2
data_out<-data_out[!is.na(data_out$label),]
write.table(data_out,'second.txt',sep = '\t',
            quote=FALSE,col.names = TRUE,row.names = FALSE)

### 2.Random Forest Regressor
data=read.table("first.txt",header=TRUE) #inter data
ndata=read.table("second.txt",header=TRUE) # exter data

set.seed(520)
nn=2/3
length(data[,1])
sub<-sample(1:nrow(data),round(nrow(data)*nn))
length(sub)
data_train<-data[sub,]
label_train<-label[sub]
data_test<-data[-sub,]
label_test<-label[-sub]
a=tune.randomForest(label~.,data=data_train, importance = TRUE, proximity = TRUE)
print(a)
a$best.model

### 3.Predict
# predict internal dataset
library(pROC) #加载pROC包
data_predict=predict(a$best.model, data_test[-dim(data_test)[2]])
data_pre1=data_predict[data_test$label==1]
data_pre2=data_predict[data_test$label==0]
min(na.omit(data_pre1))
max(na.omit(data_pre2))
roc2<-roc(data_test$label,data_predict)
plot(roc2,print.auc=TRUE,plot=TRUE,print.thres=TRUE) 

cut<-0.5 # set cuttoff
data_predict[data_predict>cut]=1
data_predict[data_predict<cut]=0
paste0('Precision:',sum(data_predict==data_test$label)/dim(data_test)[1])

# predict internal dataset
ext_pre=predict(a$best.model, ndata[-dim(ndata)[2]])
ext_pre1=ext_pre[ndata$label==1]
ext_pre2=ext_pre[ndata$label==0]
min(na.omit(ext_pre1))
max(na.omit(ext_pre2))
roc2<-roc(ndata$label,ext_pre)
plot(roc2,print.auc=TRUE,plot=TRUE,print.thres=TRUE) 
ext_pre[ext_pre>cut]=1
ext_pre[ext_pre<cut]=0
paste0('Precision:',sum(ext_pre==ndata$label)/dim(ndata)[1])


