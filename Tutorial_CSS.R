# packages installieren und laden
library(foreign)
library(caret)
library(zoo)
library(foreach)


users=seq(1:7)
activities=c("WALKING","STILL","CYCLING","STAIRS")

# Einlesen aller Nutzerdaten und Minimm der Zeilen berechnen
min=foreach(u=users,.combine=min)%:%
  foreach(a=activities,.combine=min)%do%
  as.numeric(nrow(assign(paste("user",u,a,sep="."),read.arff(paste("R_Tutorial",u,paste(a,"arff",sep="."),sep="/")))))


# Funktion zur Berechnung von Features mit Sliding Window
calc_features<- function(data,columns,features,width,by){
  x<-foreach(s=columns,.combine=cbind) %:% 
    foreach(f=features,.combine=cbind) %do% 
{data.frame(rollapply(data=data[,s], width=width, by=by, FUN=get(f), na.rm=TRUE)) }
# Spaltennamen setzen
c<-foreach(s=columns,.combine=c) %:% 
  foreach(f=features,.combine=c) %do% { paste(s,f,sep="_")}
colnames(x)[]<-c
return(x)  
}

w=40
sensors=c("Accelerometer-X","Accelerometer-Y","Accelerometer-Z")
fun=c("sd","mean")

data=foreach(u=users,.combine=rbind)%:%
  foreach(a=activities,.combine=rbind)%do%
{
  # Prävalenz der Nutzer/Klassen
  d=get(paste("user",u,a,sep="."))[1:min,]
  t=calc_features(d,sensors,fun,w,w/2)
  
  l=data.frame(d[seq(w-1,nrow(d),w/2),"Traininglabel"],rep(u,length.out=nrow(t)))
  colnames(l) = c("label","user")
  return(cbind(l,t))
}


# Daten visualisieren oder Übersicht
featurePlot(x = data[,-c(1,2)],     y = data$label,  plot = "pairs")


# Daten PCA
pca_c <- preProcess(data[,-c(1,2)],method=c("pca"))
data=cbind(data[1],predict(pca_c,data[,-c(1,2)]))
featurePlot(x = data[-1],     y = data$label,  plot = "pairs")


# Prediction ohne Holdout (Resubstitution)
x=train(data[,-1], data[,1], method = "nb") #knn,nb,...
p=predict(x,data[-1])
confusionMatrix(p,as.factor(data$label))


# BREAK


# Daten in Test und Train aufteilen
set.seed(42)
trainIndex1 <- createDataPartition(data$label, p = .8, list = FALSE, times = 1) # Split at 80%
train <- data[trainIndex1,]
test  <- data[-trainIndex1,]

x=train(train[,-1], train[,1], method = "nb") #knn,nb,...
p=predict(x,test[-1])
confusionMatrix(p,as.factor(test$label))
#Klassifizierte Klassenaufteilung
featurePlot(x = test[-1],     y = p,  plot = "pairs")


# Klassifikation mit method = nb, knn o.Ä.

# mit trainControl und none
trc <- trainControl(## without model tuning
  method = "none",
  number = 10,
  ## repeated ten times
  repeats = 1)

x = train(train[,-1], train[,1], method = "nb", trControl=trc, tuneGrid = data.frame(usekernel = FALSE, fL = 0))
p=predict(x,test[-1])
confusionMatrix(p,as.factor(test$label))

# mit trainControl und Bootstrapping
trc <- trainControl(## bootstrapping
  method = "boot",
  number = 10,
  ## repeated ten times
  repeats = 1)

x = train(data[,-1], data[,1], method = "nb", trControl=trc)


# mit trainControl und Crossvalidation
trc <- trainControl(## 10-fold CV
  method = "cv",
  number = 10,
  ## repeated ten times
  repeats = 1)

x = train(data[,-1], data[,1], method = "nb", trControl=trc)
