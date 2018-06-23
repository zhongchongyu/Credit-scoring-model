normalizedGini <- function(aa, pp) {
  Gini <- function(a, p) {
    if (length(a) !=  length(p)) stop("Actual and Predicted need to be equal lengths!")
    temp.df <- data.frame(actual = a, pred = p, range=c(1:length(a)))
    temp.df <- temp.df[order(-temp.df$pred, temp.df$range),]
    population.delta <- 1 / length(a)
    total.losses <- sum(a)
    null.losses <- rep(population.delta, length(a)) # Hopefully is similar to accumulatedPopulationPercentageSum
    accum.losses <- temp.df$actual / total.losses # Hopefully is similar to accumulatedLossPercentageSum
    gini.sum <- cumsum(accum.losses - null.losses) # Not sure if this is having the same effect or not
    sum(gini.sum) / length(a)
  }
  Gini(aa,pp) / Gini(aa,aa)
}

train = read.csv("D:/DataSet/Credit/featured/train_bayes.csv", fileEncoding='UTF-8')
test = read.csv("D:/DataSet/Credit/featured/test_bayes.csv", fileEncoding='UTF-8')
id_train <- train$report_id
id_test <- test$report_id
train$report_id <- c()
test$report_id <- c()


# 线性模型
lmmod <- lm(y ~ ., data = train)
summary(lmmod)
confint(lmmod, level=0.95)

# 逐步回归
lm.sol<-lm(y~.,data=train) 
summary(lm.sol)
lm.step<-step(lm.sol) 
summary(lm.step)
drop1(lm.step) 

predict(lm.step,test)


# K-folds交叉验证
train$pred <- 0
test$pred <- 0
K = 5
suppressPackageStartupMessages(library(caret))
folds<-createFolds(y=train$y,k=K)
train$pred <- 0
for(i in 1:K){
  #每次先选好训练集和测试集
  train_cv<-train[-folds[[i]],]
  test_cv<-train[folds[[i]],]
  
  #然后训练模型并预测,假设train_cv最后一列是target，前面的列都是features
  model<-lm(y ~ ., data = train_cv)
  pred<-predict(model,test_cv)  # type = 'response'
  
  #计算oof结果（oof：out of folds）
  train[folds[[i]],]$pred <- pred
  print(summary(model))
  
  # 预测测试集
  pred_test <- predict(model,test) / K
  test$pred <- test$pred + pred_test
  
}


normalizedGini(train$y, train$pred)




#############################
##### 逐步回归交叉验证 #####
#############################
train_copy <- train
test_copy <- test
train_copy$pred <- 0
test_copy$pred <- 0
K = 5
suppressPackageStartupMessages(library(caret))
folds<-createFolds(y=train$y,k=K)

for(i in 1:K){
  #每次先选好训练集和测试集
  train_cv<-train[-folds[[i]],]
  test_cv<-train[folds[[i]],]
  
  #然后训练模型并预测,假设train_cv最后一列是target，前面的列都是features
  lm.sol<-lm(y ~ ., data = train_cv)
  lm.step<-step(lm.sol) 
  pred<-predict(lm.step,test_cv)  # type = 'response'
  
  #计算oof结果（oof：out of folds）
  train_copy[folds[[i]],]$pred <- pred
  print(summary(lm.step))
  
  # 预测测试集
  pred_test <- predict(lm.step,test) / K
  test_copy$pred <- test_copy$pred + pred_test
  
}


normalizedGini(train_copy$y, train_copy$pred)












# 保存 sub_train 结果
sub_train <- list()
sub_train$report_id <- id_train
sub_train$y <- train$y
sub_train$pred <- train_copy$pred
sub_train <- data.frame(sub_train)

# 保存 sub 结果
sub <- list()
sub$report_id <- id_test
sub$pred <- test_copy$pred
sub <- data.frame(sub)

# 保存结果文件
write.csv(sub_train, file="D:/DataSet/Credit/result/linear_train.csv",row.names=FALSE)
write.csv(sub, file="D:/DataSet/Credit/result/linear_test.csv",row.names=FALSE)
