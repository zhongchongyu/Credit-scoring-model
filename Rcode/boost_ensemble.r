# download R
# download R Studio
# Put this file with your train and test file under same folder
# Click on Session --> Set Working Directory --> To Source File Location
# In the console, type --> install('data.table')
# In R, you can select each row and click Run button to run the file line by line
rm(list=ls())
library(data.table)


#Load Base Datasets
train = read.csv("D:/DataSet/Credit/featured/train_all_feature.csv", fileEncoding='UTF-8')
test = read.csv("D:/DataSet/Credit/featured/test_all_feature.csv", fileEncoding='UTF-8')
table(train$y)
dim(train)
dim(test)


#AUC function
AUC <-function (actual, predicted) {
  r <- as.numeric(rank(predicted))
  n_pos <- as.numeric(sum(actual == 1))
  n_neg <- as.numeric(length(actual) - n_pos)
  auc <- (sum(r[actual == 1]) - n_pos * (n_pos + 1)/2)/(n_pos *  n_neg)
  auc
}



#Load Model1
tmp1 <- read.csv('D:/DataSet/Credit/result/xgboost_train.csv', fileEncoding='UTF-8')
tmp1$y <- c()
tmp2 <- read.csv('D:/DataSet/Credit/result/xgboost_test.csv', fileEncoding='UTF-8')
model1 <- rbind( tmp1, tmp2  )
colnames(model1)[2] <- "xgb"


#Load Model2
tmp1 <- read.csv('D:/DataSet/Credit/result/NN_train.csv', fileEncoding='UTF-8')
tmp1$y <- c()
tmp2 <- read.csv('D:/DataSet/Credit/result/NN_test.csv', fileEncoding='UTF-8')
model2 <- rbind( tmp1, tmp2  )
colnames(model2)[2] <- "NN"

#Load Model3
tmp1 <- read.csv('D:/DataSet/Credit/result/SimpleBayes_train.csv', fileEncoding='UTF-8')
tmp1$y <- c()
tmp2 <- read.csv('D:/DataSet/Credit/result/SimpleBayes_test.csv', fileEncoding='UTF-8')
model3 <- rbind( tmp1, tmp2  )
colnames(model3)[2] <- "SimpleBayes"

#Load Model4
tmp1 <- read.csv('D:/DataSet/Credit/result/RGF_train.csv', fileEncoding='UTF-8')
tmp1$y <- c()
tmp2 <- read.csv('D:/DataSet/Credit/result/RGF_test.csv', fileEncoding='UTF-8')
model4 <- rbind( tmp1, tmp2  )
colnames(model4)[2] <- "RGF"

#Load Model5
tmp1 <- read.csv('D:/DataSet/Credit/result/RF_train.csv', fileEncoding='UTF-8')
tmp1$y <- c()
tmp2 <- read.csv('D:/DataSet/Credit/result/RF_test.csv', fileEncoding='UTF-8')
model5 <- rbind( tmp1, tmp2  )
colnames(model5)[2] <- "RF"

#Load Model6
tmp1 <- read.csv('D:/DataSet/Credit/result/linear_train.csv', fileEncoding='UTF-8')
tmp1$y <- c()
tmp2 <- read.csv('D:/DataSet/Credit/result/linear_test.csv', fileEncoding='UTF-8')
model6 <- rbind( tmp1, tmp2  )
colnames(model6)[2] <- "linear"

#Load Model7
tmp1 <- read.csv('D:/DataSet/Credit/result/GLM_train.csv', fileEncoding='UTF-8')
tmp1$y <- c()
tmp2 <- read.csv('D:/DataSet/Credit/result/GLM_test.csv', fileEncoding='UTF-8')
model7 <- rbind( tmp1, tmp2  )
colnames(model7)[2] <- "GLM"

#Load Model8
tmp1 <- read.csv('D:/DataSet/Credit/result/gcForest_train.csv', fileEncoding='UTF-8')
tmp1$y <- c()
tmp2 <- read.csv('D:/DataSet/Credit/result/gcForest_test.csv', fileEncoding='UTF-8')
model8 <- rbind( tmp1, tmp2  )
colnames(model8)[2] <- "gcForest"

#Load Model9
tmp1 <- read.csv('D:/DataSet/Credit/result/FFM_train.csv', fileEncoding='UTF-8')
tmp1$y <- c()
tmp2 <- read.csv('D:/DataSet/Credit/result/FFM_test.csv', fileEncoding='UTF-8')
model9 <- rbind( tmp1, tmp2  )
colnames(model9)[2] <- "FFM"

#Load Model10
tmp1 <- read.csv('D:/DataSet/Credit/result/BayesNet_train.csv', fileEncoding='UTF-8')
tmp1$y <- c()
tmp2 <- read.csv('D:/DataSet/Credit/result/BayesNet_test.csv', fileEncoding='UTF-8')
model10 <- rbind( tmp1, tmp2  )
colnames(model10)[2] <- "BayesNet"



#Merge all models by ID
raw <- data.table( report_id=c(train$report_id, test$report_id)  )
raw <- merge(    raw, model1, by="report_id", sort=F )#xgb_y
raw <- merge(    raw, model2, by="report_id", sort=F )#ffm
raw <- merge(    raw, model3, by="report_id", sort=F )#lgb
raw <- merge(    raw, model4, by="report_id", sort=F )#log_n
raw <- merge(    raw, model5, by="report_id", sort=F )#rgf
raw <- merge(    raw, model6, by="report_id", sort=F )#nn
raw <- merge(    raw, model7, by="report_id", sort=F )#xgb
raw <- merge(    raw, model8, by="report_id", sort=F )#alltree
raw <- merge(    raw, model9, by="report_id", sort=F )#log
raw <- merge(    raw, model10, by="report_id", sort=F )#log
#Split Train and Test
tr <- data.table( report_id=train$report_id )
tr <- raw[ raw$report_id %in% train$report_id  ]
ts <- raw[ raw$report_id %in% test$report_id  ]
tr[, report_id:=NULL ]
ts[, report_id:=NULL ]
y <- train$y


#Rank Train and Test
for( i in 1:ncol(tr) ){
  tr[[i]] <- rank(tr[[i]], ties.method = "average")
  ts[[i]] <- rank(ts[[i]], ties.method = "average")
}


#turn Matrix
tr <- as.matrix(tr)
ts <- as.matrix(ts)

#Optim transform function
fn.optim.sub <- function( mat, pars ){
  as.numeric( rowSums( mat * matrix( pars, nrow=nrow(mat) , ncol=ncol(mat), byrow=T ) ) )
}


#Optim evaluation maximization function
fn.optim <- function( pars ){
  AUC( y , fn.optim.sub( tr , pars ) )
}



#Bag optim 3 times using random initial Weigths
# we can test more seeds here if needed
set.seed(2)
initial_w <- rep(1/ncol(tr),ncol(tr) ) + runif( ncol(tr) ,-0.005,0.005 )
opt1 <- optim( par=initial_w , fn.optim, control = list(maxit=3333, trace=T, fnscale = -1)   )

set.seed(999)
initial_w <- rep(1/ncol(tr),ncol(tr) ) + runif( ncol(tr) ,-0.005,0.005 )
opt2 <- optim( par=initial_w , fn.optim, control = list(maxit=3333, trace=T, fnscale = -1)   )

set.seed(852)
initial_w <- rep(1/ncol(tr),ncol(tr) )
opt3 <- optim( par=initial_w , fn.optim, control = list(maxit=3333, trace=T, fnscale = -1)   )


#Show AUC
AUC( y , fn.optim.sub( tr , opt1$par ) )
print( data.frame( colnames(tr) , opt1$par ) )

AUC( y , fn.optim.sub( tr , opt2$par ) )
print( data.frame( colnames(tr) , opt2$par ) )

AUC( y , fn.optim.sub( tr , opt3$par ) )
print( data.frame( colnames(tr) , opt3$par ) )

tmp <-       rank( fn.optim.sub( tr, opt1$par ) )
tmp <- tmp + rank( fn.optim.sub( tr, opt2$par ) )
tmp <- tmp + rank( fn.optim.sub( tr, opt3$par ) )
print( AUC( y , tmp ) )


#Calcule predictions of TestSet
tmp <-       rank( fn.optim.sub( ts, opt1$par ) )
tmp <- tmp + rank( fn.optim.sub( ts, opt2$par ) )
tmp <- tmp + rank( fn.optim.sub( ts, opt3$par ) )


#Build Submission File
sub  <- data.frame( id=test$id, target = tmp/max(tmp) )
summary( sub$target  )
write.table( sub, 'D:/optim_ensemble_submission.csv', row.names=F, quote=F, sep=','  )
