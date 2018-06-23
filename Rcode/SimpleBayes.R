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

# 选择最重要的前(i-1)个变量
i <- 15
while (length(train) >= i)
{
  train[[i]] <- c()
  test[[i-1]] <- c()
}

###################################################
################## 朴素贝叶斯 #####################
###################################################
library(klaR)
train_simple_bayes <- train
train_simple_bayes$y <- as.factor(train_simple_bayes$y)
test_simple_bayes <- test

library(caret)
K = 5
train$pred <- 0
test$pred <- 0
folds<-createFolds(y=train_simple_bayes$y,k=K)
for(i in 1:K){
  #每次先选好训练集和测试集
  train_cv <- train_simple_bayes[-folds[[i]],]
  valid_cv <- train_simple_bayes[folds[[i]],]
  
  # 训练
  bfit <- NaiveBayes(y~.,train_cv,na.action=na.pass)
  
  # 预测验证集
  result_train <- predict(bfit,valid_cv)
  result_train_summary <- data.frame(result_train)
  result_train_class <- result_train_summary[,1]
  train[folds[[i]],]$pred <- result_train_summary[[3]]  #计算oof结果（oof：out of folds）
  
  # 预测测试集
  result_test <- predict(bfit,test_simple_bayes)
  result_test_summary <- data.frame(result_test)
  result_test_class <- result_test_summary[,1]
  test$pred <- test$pred + result_test_summary[[3]] / K
  
  
}

# 保存 sub_train 结果
sub_train <- list()
sub_train$report_id <- id_train
sub_train$y <- train$y
sub_train$pred <- train$pred
sub_train <- data.frame(sub_train)

# 保存 sub 结果
sub <- list()
sub$report_id <- id_test
sub$pred <- test$pred
sub <- data.frame(sub)

#train$pred <- c()
#test$pred <- c()
normalizedGini(as.numeric(train_simple_bayes$y), sub_train$pred)

# 保存结果文件
write.csv(sub_train, file="D:/DataSet/Credit/result/SimpleBayes_train.csv",row.names=FALSE)
write.csv(sub, file="D:/DataSet/Credit/result/SimpleBayes_test.csv",row.names=FALSE)
