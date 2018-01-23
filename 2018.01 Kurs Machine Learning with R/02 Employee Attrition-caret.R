# Source of data ---------------------------------------------------------
# 
# https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset




# Initialisation ----------------------------------------------------------

# I use to keep the CRAN repository "stable" and update libraries only several times a year
options(repos = c(CRAN = "https://mran.revolutionanalytics.com/snapshot/2018-01-18"))

# install library "pacman" if necessary
# install.packages("pacman")

# load some libraries
library(pacman)
p_load(dplyr, ggplot2, gridExtra, caret, reshape2, Metrics, plotmo) 

# source some additional functions
source("00 Functions.R")




# Read and prepare data ---------------------------------------------------

# read data
df <- read.csv("02 Employee Attrition - data.csv")

# check classes of variables
str(df)

# check summaries of data
summary(df)




### Prepare data

# Change Attrition to 0/1
# df$Attrition <- ifelse(df$Attrition == "No", 0, 1)

# Remove columns with only 1 value
df <- remove_columns_with_only_one_value(df)



### Feature engineering
# ...



### Generate train and test data

# generate a new column and mark some 30% of data as test-data
set.seed(123456)
df <- df %>% 
  mutate(test_data = sample(c(0, 1), 
                            nrow(df), 
                            replace = T, 
                            prob = c(0.7, 0.3)))
# count number of 0/1 of df$test
table(df$test_data)

# select train-data: data to train the models on
train <- df %>% 
  filter(test_data == 0) %>% 
  select(-test_data)

# select test-data: on this data we evaluate model performance
test <- df %>% 
  filter(test_data == 1) %>% 
  select(-test_data)




# Summary of results (accuracy) -------------------------------------------
#                             
#                          untuned     tuned     Comment
# 1.  mean ensemble         0.???      0.???     
# 2.  XGBOOST               0.853      0.???     
# 3.  GBM                   0.867      0.???     
# 4.  randomforrest         0.855      0.867     
# 5.  mars                  0.878      0.878     GLM
# 7.  base line             0.830                Most frequent value as prediction





# base line ---------------------------------------------------------------
# choose most frequent value as prediction

# "train model"
bl <- data.frame(obs = test$Attrition,
                 pr = as.numeric(names(sort(table(train$Attrition), decreasing = T)[1])))

# show confusion matrix
confusionMatrix(bl$pr, bl$obs)

# clean up
rm(bl)




# mars --------------------------------------------------------------------

# load library
p_load(earth)

# train model
# degree = 1, no interaction terms
mars <- earth(Attrition~., data = train, glm = list(family = binomial), degree = 1, penalty = 2)
mars

# degree = 2, with 2-way interaction terms
mars <- earth(Attrition~., data = train, glm = list(family = binomial), degree = 2, penalty = 3)
mars

# show confusion matrix
pr <- predict(mars, test %>% select(-Attrition))
confusionMatrix(ifelse(pr > 0.5, 1, 0), test$Attrition)

# show some model evaluation plots
plot(mars, which = 1:4)

# variable importance
ev <- evimp(mars, trim = T)
print(ev)
plot(ev)

# show the influence of variables
plotmo(mars)

# plot the distribution of the predicted values for each class
plotd(mars)

# print model rules
summary(mars, digits = 2, style = "pmax")

# clean up
rm(mars, pr, ev)




# randomForest ------------------------------------------------------------

# load library
p_load(randomForest)

# train model (mtry at default value = floor(sqrt(ncol(train))) = 5)
rf <- randomForest(as.factor(Attrition)~., data = train, ntree = 5000, importance = T)
rf

# train model with specific mtry
rf <- randomForest(as.factor(Attrition)~., data = train, ntree = 5000, mtry = 13, importance = T)
rf

# show confusion matrix
pr <- predict(rf, test %>% select(-Attrition))
confusionMatrix(pr, as.factor(test$Attrition))

# plot how error developes during training process
plot(rf)

# plot importance of each variable
varImpPlot(rf, sort = T, scale = T)

# partial dependence plot
plotmo(rf, nrug = "density", pmethod = "apartdep", ylim = NULL, degree2 = F)



# Parametertuning: mtry (default = floor(sqrt(ncol(train))) = 5)
tuneRF(train %>% select(-Attrition), as.factor(train$Attrition), 
       ntreeTry = 5000, mtryStart = 10, stepFactor = 1.3, improve = 0.00001, 
       trace = T, plot = T, doBest = F)

# Parametertuning: mtry über eine Schleife optimieren
Summary <- data.frame(mtry = c(2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 25, 30), train = NA, test = NA)
i <- 1
for (mtry in Summary$mtry) {
  print(paste(Sys.time(), "   mtry =", mtry))
  rf <- randomForest(as.factor(Attrition)~., data = train, ntree = 5000, mtry = mtry)
  Summary$mtry[i] <- mtry
  Summary$train[i] <- round(accuracy(predict(rf, train %>% select(-Attrition)), train$Attrition), 5)
  Summary$test[i] <- round(accuracy(predict(rf, test %>% select(-Attrition)), test$Attrition), 5)
  print(Summary[i, ])
  i <- i + 1
  rm(rf)
}
print(Sys.time())
# show results
Summary
ggplot(melt(Summary, id = "mtry"), aes(x = mtry, y = value, colour = variable)) + 
  geom_line()
print(paste0("Maximum accuracy at mtry = ", Summary$mtry[which.max(Summary$test)]))



# clean up
rm(rf, Summary, mtry, i, pr)




# GBM ---------------------------------------------------------------------

# load library
p_load(gbm)

# train model
ntrees <- 500
gbm <- gbm(Attrition~., data = train, distribution = "bernoulli",
           n.trees = ntrees, shrinkage = 0.05, 
           interaction.depth = 3,     # 1: additive model, 2: two-way interactions, etc.
           n.minobsinnode = 10,
           bag.fraction = 0.5,
           cv.folds = 5)
gbm

# plot how error developes during training process
best_ntrees <- gbm.perf(gbm, method = "cv")

# show confusion matrix
pr <- predict(gbm, test %>% select(-Attrition), n.trees = best_ntrees)
confusionMatrix(ifelse(pr > 0, 1, 0), as.factor(test$Attrition))

# variable importance
VarImp <- summary(gbm, n.trees = best_ntrees)
VarImp

# Partial dependence plot
temp_names <- VarImp[1:4, ]
temp_names <- as.character(temp_names$var[order(-temp_names$rel.inf)])
par(mfrow = c(2, 2))
for (i in 1:length(temp_names)) {
  plot(gbm, i.var = temp_names[i], ntrees = best_ntrees, type = "link") 
}
par(mfrow = c(1, 1))

# 2 way-interactions
# calculate most important interactions
zaehler <- 1
tm <- names(train %>% select(-Attrition))
ie <- data.frame(matrix(NA, nrow = length(tm) * (length(tm) - 1) / 2, ncol = 3))
names(ie) <- c("Var1", "Var2", "H_Statistic")
for (i in 1:(length(tm) - 1)) {
  for (j in (i + 1):length(tm)) {
    ie[zaehler, 1] <- tm[i]
    ie[zaehler, 2] <- tm[j]
    ie[zaehler, 3] <- round(interact.gbm(gbm, df, i.var = c(i, j), n.trees = best_ntrees), 5)
    zaehler <- zaehler + 1
  } 
}
ie <- ie %>% arrange(-H_Statistic)
# plot most important interactions
top <- 5
ie_top <- ie[1:top, ]
ie_top
for (i in top:1) { 
  plot(gbm, i.var = c(ie$Var1[i], ie$Var2[i]), n.trees = best_ntrees, return.grid = F)
}

# 3 way-interactions (slow!)
# calculate most important interactions
zaehler <- 1
tm <- names(train %>% select(-Attrition))
ie <- data.frame(matrix(NA, nrow = length(tm) * (length(tm) - 1) * (length(tm) - 2) / 6, ncol = 4))
names(ie) <- c("Var1", "Var2", "Var3", "H_Statistic")
for (i in 1:(length(tm) - 2)) {
  for (j in (i + 1):(length(tm) - 1)) {
    for (k in (j + 1):length(tm)) {
      ie[zaehler, 1] <- tm[i]
      ie[zaehler, 2] <- tm[j]
      ie[zaehler, 3] <- tm[k]
      ie[zaehler,4] <- round(interact.gbm(gbm, df, i.var = c(i, j, k), n.trees = best_ntrees), 8)
      zaehler <- zaehler + 1
    } 
  }
}
ie <- ie %>% arrange(-H_Statistic)
# plot most important interactions
top <- 5
ie_top <- ie[1:top, ]
ie_top
for (i in top:1) {
  plot(gbm, i.var = c(ie$Var1[i], ie$Var2[i], ie$Var3[i]), n.trees = best_ntrees, return.grid = F)
}



# clean up
rm(gbm, ie, ie_top, top, VarImp, best_ntrees, ntrees, i, j, k, pr, temp_names, 
   temp_names2, tm, zaehler)




# XGBOOST -----------------------------------------------------------------


xgb <- train(x = data.matrix(train %>% select(-Attrition)),
             y = train$Attrition, 
             method = "xgbTree", metric = "Accuracy",
             trControl = trainControl(method = "cv", number = 5), tuneLength = 10)

pdp.lstat <- partial(xgb, pred.var = "Age", plot = T, rug = T)


# prepare data
train_y <- train$Attrition
train_mat <- sparse.model.matrix(~., data = train %>% select(-Attrition))
train_d <- xgb.DMatrix(data = train_mat, label = train_y)
test_y <- test$Attrition
test_mat <- sparse.model.matrix(~., data = test %>% select(-Attrition))
test_d <- xgb.DMatrix(data = test_mat, label = test_y)

# train model
xgb <- xgb.cv(data = train_d, 
              objective = "binary:logistic", eval_metric = "error", nfold = 5,
              nrounds = 1000, eta = 0.05,
              max_depth = 6, min_child_weight = 1, gamma = 0,
              subsample = 1, colsample_bytree = 1,
              early_stopping_rounds = 30)

# plot how error developes during training process
ggplot(melt(xgb$evaluation_log %>% select(-train_error_std, -test_error_std), id = "iter"),
       aes(x = iter, y = value, colour = variable)) +
  geom_line(size = I(1))

# re-run model with nrounds = best_nrounds from cv
xgb <- xgb.train(data = train_d, 
                 objective = "binary:logistic", eval_metric = "error",
                 nrounds = 57, eta = 0.05,
                 max_depth = 6, min_child_weight = 1, gamma = 0,
                 subsample = 1, colsample_bytree = 1)

# show confusion matrix
pr <- predict(xgb, test_d)
confusionMatrix(ifelse(pr > 0.5, 1, 0), test$Attrition)

# variable importance
xgb.importance(colnames(train_mat), xgb)
xgb.plot.importance(xgb.importance(colnames(train_mat), xgb), rel_to_first = T, xlab = "Relative importance")




p_load(pdp)

pdp.lstat <- partial(xgb, pred.var = "Age", plot = T, rug = T)

xgb %>%
  partial(pred.var = "Age") %>%
  autoplot(rug = T, train = train_mat)






# clean up
rm(train_y, train_mat, train_d, test_y, test_mat, test_d, xgb, pr)

