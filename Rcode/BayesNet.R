# 计算 gini
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
################### 贝叶斯网络 ####################
###################################################
library(bnlearn)   			# 加载包 
train_bayes_net <- train
train_bayes_net$y <- c()
train_bayes_net$y <- train$y
train_bayes_net$y <- as.factor(train_bayes_net$y)

test_bayes_net <- test
y <- train$y[1:dim(test)[1]]
test_bayes_net <- cbind(test_bayes_net,y)

all_bayes_net <- rbind(train_bayes_net,test_bayes_net)

# 数据转化为浮点数
i <- 1
while(i <= length(all_bayes_net)-1 ){
  all_bayes_net[[i]] <- as.numeric(all_bayes_net[[i]])
  i <- i+1
}

# 数据离散化 
i <- 1
while(i <= length(all_bayes_net)-1 )
{
  if ( length(unique(all_bayes_net[[i]]))  == 2 )
  {
    all_bayes_net[i] <- discretize(all_bayes_net[i],method='interval',breaks = 2)
  }
  if( 2 < length(unique(all_bayes_net[[i]])) & length(unique(all_bayes_net[[i]]))  <= 100 )
  {
    all_bayes_net[i] <- discretize(all_bayes_net[i],method='interval',breaks = 8)
  }
  if( 100 < length(unique(all_bayes_net[[i]])) & length(unique(all_bayes_net[[i]]))  <= 500 )
  {
    all_bayes_net[i] <- discretize(all_bayes_net[i],method='interval',breaks = 40)
  }
  if( length(unique(all_bayes_net[[i]]))  > 500 )
  {
    all_bayes_net[i] <- discretize(all_bayes_net[i],method='interval',breaks = 90)
  }
  i <- i+1
}

train_bayes_net <- all_bayes_net[1:dim(train_bayes_net)[1],]
test_bayes_net <- all_bayes_net[dim(train_bayes_net)[1]+1:dim(test_bayes_net)[1],]
test_bayes_net$y <- c()

# 使用爬山算法进行结构学习  
bayesnet <- hc(train_bayes_net)  
# 显示网络图  
plot(bayesnet) 
library(caret)
K = 5
folds<-createFolds(y=train_bayes_net$y,k=K)
train$pred <- 0
test$pred <-0
for(i in 1:K){
  #每次先选好训练集和测试集
  train_cv <- train_bayes_net[-folds[[i]],]
  valid_cv <- train_bayes_net[folds[[i]],]
  
  fitted <- bn.fit(bayesnet, train_cv,method='mle')  
  
  # 预测验证集  
  result_train <- predict(fitted,data=valid_cv,node='y', prob = TRUE)
  result_train_summary <- data.frame(t(attr(result_train,"prob")))
  train[folds[[i]],]$pred <- result_train_summary[[2]]
  
  # 预测测试集
  result_test <- predict(fitted,data=test_bayes_net,node='y', prob = TRUE)
  result_test_summary <- data.frame(t(attr(result_test,"prob")))
  test$pred <- test$pred + result_test_summary[[2]] / K
  
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

# 计算 oof gini
normalizedGini(as.numeric(train_bayes_net$y), sub_train$pred)

# 保存结果文件
write.csv(sub_train, file="D:/DataSet/Credit/result/BayesNet_train.csv",row.names=FALSE)
write.csv(sub, file="D:/DataSet/Credit/result/BayesNet_test.csv",row.names=FALSE)
