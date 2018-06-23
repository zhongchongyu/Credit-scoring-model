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


##########################
###### 广义线性模型 ######
##########################
library("glmnet") #加载该软件包
# 根据不同的 L1 惩罚，训练70个不同的模型，然后观察“由模型解释的残差的比”
# 对着 L1 惩罚的减少，直到“由模型解释的残差的比”不在显著增加为止，选择此时的 L1 惩罚
# 同时能够得到 系数 随着“L1惩罚”、“Log-L1惩罚”、“由模型解释的残差的比”的变化情况，用来选择L1
fit<-glmnet(as.matrix(train[-1]),as.matrix(train[1]),
            family="binomial",
            alpha=1,
            nlambda=70,
            standardize=TRUE)

# 列出每个模型的 （自由度、由模型解释的残差的比、L1 惩罚） 表
print(fit)
# 绘制  "由模型解释的残差的比"  随着 "模型序号"、“自由度”、“L1惩罚”变化的曲线
dev <- fit$dev.ratio
Log_L1 <- log(fit$lambda)
DF <- fit$df
models <- c(1:length(dev))
plot(models,dev,type='o',col="red",pch=18,bg="yellow",lwd=1.5)
grid(nx=12,ny=6,lwd=0.5,col='black')
plot(DF,dev,type='o',col="red",pch=18,bg="yellow",lwd=1.5)
grid(nx=12,ny=6,lwd=2)
plot(Log_L1,dev,type='o',col="red",pch=18,bg="yellow",lwd=1.5)
grid(nx=12,ny=6,lwd=2)
# 绘制 系数 随着 “L1 惩罚”、“Log-L1”、"由模型解释的残差的比"的变化
plot(fit, xvar="lambda", label=TRUE)
grid(nx=12,ny=6,lwd=0.5,col='grey')
plot(fit, xvar="norm", label=TRUE)
grid(nx=12,ny=6,lwd=0.5,col='grey')
plot(fit, xvar="dev", label=TRUE)
grid(nx=12,ny=6,lwd=0.5,col='grey')

# 选定第几个模型
model_index <- 20
# 保存该模型的 L1 惩罚
L1_regular <- fit$lambda[model_index]
# 保存第20个模型的系数
coefficients<-coef(fit,s=fit$lambda[model_index])
#系数不为0的特征索引
Active.Index<-which(coefficients!=0)
#系数不为0的特征系数值
Active.coefficients<-coefficients[Active.Index]   
# 运用该模型进行预测
pred <- predict(fit, newx=as.matrix(train[-1]),type="response", s=c(fit$lambda[model_index]))


####################
####  交叉验证  ####
####################

cv.fit=cv.glmnet(as.matrix(train[-1]),as.matrix(train[1]),
                 family='binomial',
                 standardize=TRUE,
                 type.measure="deviance")
plot(cv.fit)
print(cv.fit)
cv.fit$lambda.min  #最佳lambda值
cv.fit$lambda.1se#指在lambda.min一个标准差范围内得到的最简单模型的那一个lambda值。

# 预测验证集
pred_train <- predict(cv.fit, 
                      newx=as.matrix(train[-1]),
                      type="response", 
                      s=cv.fit$lambda.1se)
pred_train <- data.frame(pred_train)
colnames(pred_train) <- "pred"

# 预测测试集
pred_test  <- predict(cv.fit, 
                      newx=as.matrix(test),
                      type="response", 
                      s=cv.fit$lambda.1se)
pred_test <- data.frame(pred_test)
colnames(pred_test) <- "pred"


normalizedGini(train$y, pred_train$pred)


# 保存 sub_train 结果
sub_train <- list()
sub_train$report_id <- id_train
sub_train$y <- train$y
sub_train$pred <- pred_train$pred
sub_train <- data.frame(sub_train)

# 保存 sub 结果
sub <- list()
sub$report_id <- id_test
sub$pred <- pred_test$pred
sub <- data.frame(sub)

# 保存结果文件
write.csv(sub_train, file="D:/DataSet/Credit/result/GLM_train.csv",row.names=FALSE)
write.csv(sub, file="D:/DataSet/Credit/result/GLM_test.csv",row.names=FALSE)


#0.6277358 GINI = 2AUC - 1
#0.6479677 gini