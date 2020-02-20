########################################################
### Final Project: Predicting Pulsar Stars
########################################################
### Section B Team 30
### Team Members: Avani Bhargava, Chu-Ru (Ruth) Cheng, Roshni Shahani, 
###               Xinyan (Sarah) Wang, Yuchen (Vinnie) Zhang

##############################################
### Initial Preparations
##############################################
### source functions
source("DataAnalyticsFunctions.R")
source("PerformanceCurves.R")

### Load packages
library(tidyverse)
library(caret)
library(corrplot)
library(ggplot2)
library(randomForest)
library(tree)
library(glmnet)
library(DMwR)
library(MASS)
library(rpart)
library(class)
library(nnet)
library(pROC)

##############################################
### Data Cleaning, Preparations & Understanding
##############################################
### Read the data
pulsar <- read.csv("pulsar_stars.csv")
summary(pulsar)

### Data Cleaning
sum(is.na(pulsar))
pulsar$target_class <- as.factor(ifelse(pulsar$target_class == 0, "No", "Yes"))

### Exploratory Plots
hist(pulsar$Mean.of.the.integrated.profile,
     main="Mean of the integrated profile Histogram",
     xlab="Mean of the integrated profile",
     col = 6)

hist(pulsar$Standard.deviation.of.the.integrated.profile,
     main="Standard deviation of the integrated profile Histogram",
     xlab="Standard deviation of the integrated profile",
     col = 6)

hist(pulsar$Excess.kurtosis.of.the.integrated.profile,
     main="Excess kurtosis of the integrated profile Histogram",
     xlab="Excess kurtosis of the integrated profile",
     col = 6)

hist(pulsar$Skewness.of.the.integrated.profile,
     main="Skewness of the integrated profile Histogram",
     xlab="Skewness of the integrated profile",
     col = 6)

hist(pulsar$Mean.of.the.DM.SNR.curve,
     main="Mean of the DM-SNR curve Histogram",
     xlab="Mean of the DM-SNR curve",
     col = 7)

hist(pulsar$Standard.deviation.of.the.DM.SNR.curve,
     main="Standard deviation of the DM-SNR curve Histogram",
     xlab="Standard deviation of the DM-SNR curve",
     col = 7)

hist(pulsar$Excess.kurtosis.of.the.DM.SNR.curve,
     main="Excess kurtosis of the DM-SNR curve Histogram",
     xlab="Excess kurtosis of the DM-SNR curve",
     col = 7)

hist(pulsar$Skewness.of.the.DM.SNR.curve,
     main="Skewness of the DM-SNR curve Histogram",
     xlab="Skewness of the DM-SNR curve profile",
     col = 7)

ggplot(pulsar,aes(target_class)) + geom_bar(width = 0.5) + ggtitle("Number of Oberservations in Each Classification Group") + xlab("Pulsar Star or Not")

### Correlation plot
pMatrix <- cor(as.matrix(pulsar[,-9]))
colnames(pMatrix) <- c("MIP", "SDIP", "EKIP", "STP", "MDNC", "SDDSC", "EKDSC","SDSC")
corrplot.mixed(pMatrix, lower = "number", upper = "circle",
               tl.col="black", tl.pos = "lt")

### Look at class proportion
prop.table(table(pulsar$target_class))
### It is very imbalanced

### Create holdout sample
set.seed(1)
nrow(pulsar)
holdout.indices <- sample(nrow(pulsar), nrow(pulsar)*0.8)
train <- pulsar[holdout.indices,]
test <- pulsar[-holdout.indices,]
nrow(train)
nrow(test)

### Checking how balanced they are
mean(train$target_class=="Yes")
mean(test$target_class=="Yes")
### Very similar to original dataset

### We want to handle the imbalance, and we will use SMOTE function as a sampling method
trainS <- SMOTE(target_class ~., data = train)
testS <- SMOTE(target_class ~., data = test)
prop.table(table(trainS$target_class))
prop.table(table(testS$target_class))
### Now both our train and test sets have more balanced class variable

##############################################
### Unsupervised Learning
##############################################
### 1. k-means
xdata <- model.matrix(target_class ~ ., data=pulsar)[,-1]
xdata <- scale(xdata)
### Computing number of clusters
### Use the  script kIC in DataAnalyticsFunctions.R
kfit <- lapply(1:200, function(k) kmeans(xdata,k,nstart=10))
kaic <- sapply(kfit,kIC)
kbic  <- sapply(kfit,kIC,"B")
kHDic  <- sapply(kfit,kIC,"C")
### Plot the KICs       
par(mar=c(1,1,1,1))
par(mai=c(1,1,1,1))
plot(kaic, xlab="k (# of clusters)", ylab="IC (Deviance + Penalty)", 
     ylim=range(c(kaic,kbic,kHDic)), 
     type="l", lwd=2)
abline(v=which.min(kaic))
lines(kbic, col=4, lwd=2)
abline(v=which.min(kbic),col=4)
lines(kHDic, col=3, lwd=2)
abline(v=which.min(kHDic),col=3)
### Insert labels
text(c(which.min(kaic),which.min(kbic),which.min(kHDic)),c(mean(kaic),mean(kbic),mean(kHDic)),c("AIC","BIC","HDIC"))
### Number of clusters suggested by each information criterion, and how much variation is explained
which.min(kaic)
1 - sum(kfit[[which.min(kaic)]]$tot.withinss)/kfit[[which.min(kaic)]]$totss
which.min(kbic)
1 - sum(kfit[[which.min(kbic)]]$tot.withinss)/kfit[[which.min(kbic)]]$totss
which.min(kHDic)
1 - sum(kfit[[which.min(kHDic)]]$tot.withinss)/kfit[[which.min(kHDic)]]$totss
### Choose the 8 clusters suggested by HDIC
pulsar_kmeans <- kmeans(xdata,8,nstart=30)
### Sizes of clusters
pulsar_kmeans$size
### How these segments relate to identification of pulsar? 
aggregate(pulsar$target_class=="Yes"~pulsar_kmeans$cluster, FUN=mean)
### Look at the two clusters where majorities are real pulsars
pulsar_kmeans$centers[4,]
pulsar_kmeans$centers[7,]

### 2. PCA
pulsar_pca <- prcomp(xdata, scale=TRUE)
### Plot the variance that each component explains
par(mar=c(4,4,4,4)+0.3)
plot(pulsar_pca,main="PCA: Variance Explained by Factors")
mtext(side=1, "Factors",  line=1, font=2)
### 3 factors seems explanatory
### Loading 1
loadings <- pulsar_pca$rotation[,1:3]
v<-loadings[order(abs(loadings[,1]), decreasing=TRUE)[1:ncol(xdata)],1]
loadingfit <- lapply(1:ncol(xdata), function(k) ( t(v[1:k])%*%v[1:k] - 3/4 )^2)
v[1:which.min(loadingfit)]
### First factor is low mean, high skewness/kurtosis of integrated profile and high standard deviation, low kurtosis of DM.SNR 
### Loading 2
v<-loadings[order(abs(loadings[,2]), decreasing=TRUE)[1:ncol(xdata)],2]
loadingfit <- lapply(1:ncol(xdata), function(k) ( t(v[1:k])%*%v[1:k] - 3/4 )^2)
v[1:which.min(loadingfit)]
### Second factor is high mean, high standard deviation, low kurtosis of integrated profile and low skewness/kurtosis of DM.SNR
### Loading 3
v<-loadings[order(abs(loadings[,3]), decreasing=TRUE)[1:ncol(xdata)],3]
loadingfit <- lapply(1:ncol(xdata), function(k) ( t(v[1:k])%*%v[1:k] - 3/4 )^2)
v[1:which.min(loadingfit)]
### Third factor is high mean and skewness of DM.SNR

##############################################
### Supervised Learning
##############################################
### Use caret package to do cross-validation
### Set-ups
Metrics <- function(...) c(twoClassSummary(...),
                              defaultSummary(...))
ctrl <- trainControl(method = "cv",
                     number = 10,
                     summaryFunction = Metrics,
                     classProbs = TRUE,
                     verboseIter = TRUE)
install.packages("caret")

### 1. logistic regression
lr_model <- train(target_class~.,
                  data = trainS,
                  method = "glm",
                  family = "binomial",
                  metric = "ROC",
                  trControl = ctrl)
lr_model

### 2. logistic regression with manually removed collinearity
### Identify high correlation
HighCor <- findCorrelation(cor(pMatrix), cutoff = 0.9)
lr_manual_model <- train(target_class~.,
                        data = trainS[,-HighCor],
                        method = "glm",
                        family = "binomial",
                        metric = "ROC",
                        trControl = ctrl)
lr_manual_model

### 3. logistic regression with interaction and stepwise feature selection using AIC
lr_step <- train(target_class~., 
                 data = trainS,
                 method = "glmStepAIC",
                 family = "binomial",
                 metric = "ROC",
                 trControl = ctrl)
lr_step
 
### 4. logistic regression with interaction using lasso
### cv will select the best lambda from a sequence
lr_lasso <- train(target_class~.^2, 
                  data = trainS,
                  method = "glmnet",
                  family = "binomial",
                  tuneGrid = expand.grid(lambda = seq(0, 1, by = 0.001),
                                         alpha = 1),
                  metric = "ROC",
                  trControl = ctrl)
### The final best lambda generated by cv is 0
lr_lasso

### 5. decision tree
set.seed(1)
tree_model <- train(target_class~., 
                    data = trainS,
                    method = "rpart",
                    tuneLength = 10,
                    metric = "ROC",
                    trControl = ctrl)
### the complex parameter (cp) determined by cv is 0.000633232
### any split which does not improve the fit by cp will likely be pruned off by cv
tree_model

### 6. K-NN model
set.seed(1)
knn_model <- train(target_class~., 
                   data = trainS,
                   method = "knn",
                   tuneLength = 10,
                   metric = "ROC",
                   trControl = ctrl)
### a 17-NN model is generated by cv
knn_model
plot(knn_model)

### 7. Support Vector Machine (SVM)
set.seed(1)
svm_model <- train(target_class~., 
                   data = trainS,
                   method = "svmLinear",
                   preProcess = c("center", "scale"),
                   tuneLength = 10,
                   metric = "ROC",
                   trControl = ctrl)
svm_model

### 8. random forest
set.seed(1)
rf_model <- train(target_class~.,
                  data = trainS,
                  method = "rf",
                  ntree = 300,
                  trControl = ctrl,
                  metric = "ROC")
### rf model with subset of 2 features is generated by cv
rf_model
plot(rf_model)
plot(varImp(rf_model))

### 9. neural network
set.seed(1)
### We define grid for size and decay for caret to tune, where 
### size is the number of units in hidden layer and decay is the regularization parameter to avoid over-fitting
nnetGrid <-  expand.grid(size = seq(from = 1, to = 10, by = 1),
                         decay = seq(from = 0.1, to = 0.5, by = 0.1))
nnet_model <- train(target_class~., 
                    data = trainS,
                    method = "nnet",
                    preProcess = c("center", "scale"),
                    trControl = ctrl,
                    tuneGrid = nnetGrid,
                    metric = "ROC")
nnet_model

##############################################
### Evaluations
##############################################
### Plot the performance of all the models
all_models <- resamples((list(Logistic = lr_model,
                              LogisticRemovedHighCor = lr_manual_model,
                              LogisticStepAIC = lr_step,
                              LogisticInteractionLasso = lr_lasso,
                              DecisionTree = tree_model,
                              KNN = knn_model,
                              SVM = svm_model,
                              RandomForest = rf_model,
                              NeurelNetwork = nnet_model)),decreasing = T)
summary(all_models)
dotplot(all_models, main = "OOS Performance with 95 Percent Confidence Interval")
### random forest model is ranked highest based on its OOS performance
### but we will use the top three models and compare their performance

### 1. random forest
### Now run the random forest model on the test set
rf_pred <- predict(rf_model, newdata = testS, type="raw")
rf_pred_prob <- predict(rf_model, newdata = testS, type="prob")
actual = testS$target_class == "Yes"
###
### Different thresholds
rf.performance75 <- FPR_TPR(rf_pred_prob[,"Yes"]>=0.75, actual)
rf.performance75
rf.performance25 <- FPR_TPR(rf_pred_prob[,"Yes"]>=0.25, actual)
rf.performance25
rf.performance <- FPR_TPR(rf_pred_prob[,"Yes"]>=0.5, actual)
rf.performance
### threshold = .5 has the highest accuracy
###
### Confusion Matrix for random forest model
confusionMatrix(rf_pred, testS$target_class)
###
### ROC Curve for random forest model
rf_roc <- pROC::roc(testS$target_class, rf_pred_prob[,"No"])
plot(rf_roc, print.auc = T, legacy.axis = T, main = "Random Forest Model ROC Curve")

### 2. neural network
### Now run the neural network model on the test set
nn_pred <- predict(nnet_model, newdata = testS, type="raw")
nn_pred_prob <- predict(nnet_model, newdata = testS, type="prob")
###
### Different thresholds
nn.performance75 <- FPR_TPR(nn_pred_prob[,"Yes"]>=0.75, actual)
nn.performance75
nn.performance25 <- FPR_TPR(nn_pred_prob[,"Yes"]>=0.25, actual)
nn.performance25
nn.performance <- FPR_TPR(nn_pred_prob[,"Yes"]>=0.5, actual)
nn.performance
### threshold = .5 has the highest accuracy
###
### Confusion Matrix for neural network model
confusionMatrix(nn_pred, testS$target_class)
###
### ROC Curve for neural network model
nn_roc <- pROC::roc(testS$target_class, nn_pred_prob[,"No"])
plot(nn_roc, print.auc = T, legacy.axis = T, main = "Neural Network Model ROC Curve")

### 3. logistic regression with interaction using lasso
### Now run the lasso model on the test set
lasso_pred <- predict(lr_lasso, newdata = testS, type="raw")
lasso_pred_prob <- predict(lr_lasso, newdata = testS, type="prob")
###
### Different thresholds
lasso.performance75 <- FPR_TPR(lasso_pred_prob[,"Yes"]>=0.75, actual)
lasso.performance75
lasso.performance25 <- FPR_TPR(lasso_pred_prob[,"Yes"]>=0.25, actual)
lasso.performance25
lasso.performance <- FPR_TPR(lasso_pred_prob[,"Yes"]>=0.5, actual)
lasso.performance
### threshold = .5 has the highest accuracy
###
### Confusion Matrix for lasso model
confusionMatrix(lasso_pred, testS$target_class)
###
### ROC Curve for lasso model
lasso_roc <- pROC::roc(testS$target_class, lasso_pred_prob[,"No"])
plot(lasso_roc, print.auc = T, legacy.axis = T, main = "Lasso Model ROC Curve")

### Use Lift Curve to compare the performance of our three models
lift_results <- data.frame(Class = testS$target_class)
lift_results$RF <- predict(rf_model, newdata = testS, type = "prob")[,"No"]
lift_results$NN <- predict(nnet_model, newdata = testS, type = "prob")[,"No"]
lift_results$Lasso <- predict(lr_lasso, newdata = testS, type = "prob")[,"No"]
head(lift_results)
trellis.par.set(caretTheme())
lift_obj <- lift(Class ~ RF + NN + Lasso, data = lift_results)
###
### Lift Curve
plot(lift_obj, auto.key = list(columns = 3,
                               lines = TRUE,
                               points = FALSE))
### Almost overlapping curves, let's zoom in
###
### Lift Curve zoom in when 50% sample found
plot(lift_obj, values = 50, auto.key = list(columns = 3,
                                            lines = TRUE,
                                            points = FALSE), xlim = c(25, 30), ylim = c(45,55))

### RF and NN almost the same, Lasso slightly worse
###
### Lift Curve zoom in when 90% sample found
plot(lift_obj, values = 90, auto.key = list(columns = 3,
                                            lines = TRUE,
                                            points = FALSE), xlim = c(50, 60), ylim = c(60,100))
### Lasso best, RF second, NN worst, but all three are within 1% sample tested
###
### Lift Curve zoom in when 100% sample found
plot(lift_obj, values = 100, auto.key = list(columns = 3,
                                            lines = TRUE,
                                            points = FALSE), xlim = c(60, 90), ylim = c(95,100))
### Very different, RF the best at 66%, NN the second at 76%, and Lasso the worst at 83%

