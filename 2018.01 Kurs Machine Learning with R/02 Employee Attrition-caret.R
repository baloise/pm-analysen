# Source of data ---------------------------------------------------------
# 
# https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset




# Initialisation ----------------------------------------------------------

# Keep the CRAN repository "stable" by a fixed date
# (and update libraries only several times a year)
options(repos = c(CRAN = "https://mran.revolutionanalytics.com/snapshot/2018-01-18"))

# turn off scientific notation
options(scipen = 999)

# load some libraries
# install library "pacman" if necessary
# install.packages("pacman")
library(pacman)
p_load(dplyr, ggplot2, caret, pdp, gbm, tibble)

# source some additional functions
source("00 Functions.R")




# Read and prepare data ---------------------------------------------------

# read data
df <- read.csv("02 Employee Attrition - data.csv")
# df <- read.csv("02 Employee Attrition - data.csv") %>% 
#   slice(1:1000)

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





# Explorative data analysis -----------------------------------------------

# numerical variables: plotting distributions and calculate several statistics
plot_num(train)
profiling_num(train)

# categoric variables: plotting frequencies
freq(train)



### plot x vs. y

# adapt theme
theme1 <- trellis.par.get()
theme1$plot.symbol$col = rgb(.2, .2, .2, .4)
theme1$plot.symbol$pch = 16
theme1$plot.line$col = rgb(1, 0, 0, .7)
theme1$plot.line$lwd <- 2
trellis.par.set(theme1)

# numeric variables vs. y
featurePlot(x = train %>% select_if(is.numeric), y = train$Attrition, plot = "box", 
            scales = list(y = list(relation = "free"), x = list(rot = 90)),
            layout = c(3, 1), auto.key = list(columns = 3),  labels = c("Attrition", ""))

# categoric variables vs. y
featurePlot(x = train %>% select_if(is.factor) %>% data.matrix(), 
            y = train$Attrition, plot = "box",
            scales = list(y = list(relation = "free"), x = list(rot = 90)),
            layout = c(3, 1), auto.key = list(columns = 3),  labels = c("Attrition", ""))




# Summary of results ------------------------------------------------------
#                             
# Rg  Model                Accuracy     Comment
# 1.  ensemble XGBOOST      0.872       
# 2.  XGBOOST               0.872       
# 3.  GBM                   0.870       
# 4.  mars                  0.868       
# 5.  ensemble GLM          0.867       
# 6.  randomforrest         0.865       
# 7.  naive bayes           0.842       
# 8.  base line             0.830       Most frequent value as prediction




# base line ---------------------------------------------------------------
# choose most frequent value as prediction

# "train model"
bl <- data.frame(obs = test$Attrition,
                 pr = as.numeric(names(sort(table(train$Attrition), decreasing = T)[1])))

# show confusion matrix
confusionMatrix(bl$pr, bl$obs)

# clean up
rm(bl)




# Naive Bayes -------------------------------------------------------------

# train model (random search 20x)
set.seed(12345)
nb <- train(Attrition ~ ., data = train,
            method = "naive_bayes", metric = "Accuracy",
            trControl = trainControl(method = "cv", number = 5, search = "random"), 
            tuneLength = 5)
nb

print(paste0("Maximum Accuracy: ", round(max(nb$results$Accuracy), 4)))
ggplot(nb, metric = "Accuracy")

# show confusion matrix
pr <- predict(nb, test %>% select(-Attrition), type = "prob")
confusionMatrix(ifelse(pr$Yes > 0.5, "Yes", "No"), test$Attrition)

# variable importance
VarImp <- varImp(nb)
VarImp
plot(VarImp, top = 20)

# clean up
rm(pr, VarImp)




# Multiple additive regression splines ------------------------------------

# train model (random search 20x)
set.seed(12345)
mars <- train(Attrition ~ ., data = train,
            method = "earth", metric = "Accuracy",
            trControl = trainControl(method = "cv", number = 5, search = "random"),
            tuneLength = 20)
mars
print(paste0("Maximum Accuracy: ", round(max(mars$results$Accuracy), 4)))
ggplot(mars$results, aes(x = nprune, y = Accuracy, colour = degree)) + 
  geom_point()

# show confusion matrix
pr <- predict(mars, test %>% select(-Attrition), type = "prob")
confusionMatrix(ifelse(pr$Yes > 0.5, "Yes", "No"), test$Attrition)

# variable importance
VarImp <- varImp(mars)
VarImp
plot(VarImp, top = 20)

# model coefficients
mars$finalModel$coefficients

# model coefficients as plots
plotmo(mars$finalModel)

# clean up
rm(pr, VarImp)




# randomForest ------------------------------------------------------------
# 
# train model (random search 20x)
set.seed(12345)
rf <- train(Attrition ~ ., data = train,
             method = "parRF", metric = "Accuracy",
             trControl = trainControl(method = "cv", number = 5, search = "random"), 
             tuneLength = 20, verbose = F)
rf
print(paste0("Maximum Accuracy: ", round(max(rf$results$Accuracy), 4)))
ggplot(rf, metric = "Accuracy")

# show confusion matrix
pr <- predict(rf, test %>% select(-Attrition), type = "prob")
confusionMatrix(ifelse(pr$Yes > 0.5, "Yes", "No"), test$Attrition)

# variable importance
VarImp <- varImp(rf)
VarImp
plot(VarImp, top = 20)

# Partial dependence plot
for (i in 3:1) {
  variable <- match(sort(VarImp$importance$Overall, T)[i], VarImp$importance$Overall)
  variable <- names(train)[amatch(rownames(VarImp$importance)[variable], 
                                  names(train), maxDist = 20)]
  rf %>%
    partial(pred.var = c(variable), chull = T) %>%
    autoplot(contour = T, main = "Attrition", xlab = paste0(variable)) %>% 
    print()
}

# 2 way-Partial dependence plot
for (i in 3:2) {
  for (j in (i - 1):1) {
    variable1 <- match(sort(VarImp$importance$Overall, T)[i], VarImp$importance$Overall)
    variable1 <- names(train)[amatch(rownames(VarImp$importance)[variable1], 
                                     names(train), maxDist = 15)]
    variable2 <- match(sort(VarImp$importance$Overall, T)[j], VarImp$importance$Overall)
    variable2 <- names(train)[amatch(rownames(VarImp$importance)[variable2], 
                                     names(train), maxDist = 15)]
    rf %>%
      partial(pred.var = c(variable1, variable2), chull = T) %>%
      autoplot(contour = T, 
               main = paste0("Attrition: ", variable1, " vs. ", variable2)) %>% 
      print()
  }
}

# clean up
rm(VarImp, i, j, variable, variable1, variable2)




# GBM ---------------------------------------------------------------------

# train model (random search 20x)
set.seed(12345)
gbm <- train(Attrition ~ ., data = train,
             method = "gbm", metric = "Accuracy",
             trControl = trainControl(method = "cv", number = 5, search = "random"), 
             tuneLength = 20, verbose = F)
gbm
print(paste0("Maximum Accuracy: ", round(max(gbm$results$Accuracy), 4)))
ggplot(gbm, metric = "Accuracy")

# show confusion matrix
pr <- predict(gbm, test %>% select(-Attrition), type = "prob")
confusionMatrix(ifelse(pr$Yes > 0.5, "Yes", "No"), test$Attrition)

# variable importance
VarImp <- varImp(gbm$finalModel, numTrees = gbm$bestTune$n.trees) %>% 
  rownames_to_column("variable") %>% 
  mutate(importance = Overall / max(Overall) * 100) %>% 
  select(-Overall) %>% 
  arrange(-importance)
VarImp
ggplot(VarImp %>% slice(1:20), 
       aes(x = variable %>% reorder(importance), y = importance)) +
  geom_bar(stat = "identity") + 
  labs(y = "variable importance", x = "") +
  coord_flip()

# Partial dependence plot
for (i in 3:1) {
  variable <- names(train)[amatch(VarImp$variable[i], names(train), maxDist = 15)]
  gbm %>%
    partial(pred.var = c(variable), chull = T) %>%
    autoplot(contour = T, main = "Attrition", xlab = paste0(variable)) %>% 
    print()
}

# # 2 way-Partial dependence plot (slow!)
# for (i in 3:2) {
#   for (j in (i - 1):1) {
#     variable1 <- names(train)[amatch(VarImp$variable[i], names(train), maxDist = 15)]
#     variable2 <- names(train)[amatch(VarImp$variable[j], names(train), maxDist = 15)]
#     gbm %>%
#       partial(pred.var = c(variable1, variable2), chull = T) %>%
#       autoplot(contour = T, main = "Attrition", xlab = paste0(variable1, " vs ", variable2)) %>% 
#       print()
#   }
# }

# clean up
rm(pr, VarImp, i, j, variable, variable1, variable2)




# XGBOOST -----------------------------------------------------------------

# train model (random search 20x)
set.seed(12345)
xgb <- train(Attrition ~ ., data = train,
             method = "xgbTree", metric = "Accuracy",
             trControl = trainControl(method = "cv", number = 5, search = "random"), 
             tuneLength = 20)
xgb
print(paste0("Maximum Accuracy: ", round(max(xgb$results$Accuracy), 4)))
ggplot(xgb, metric = "Accuracy")

# show confusion matrix
pr <- predict(xgb, test %>% select(-Attrition), type = "prob")
confusionMatrix(ifelse(pr$Yes > 0.5, "Yes", "No"), test$Attrition)

# variable importance
VarImp <- varImp(xgb)
VarImp
plot(VarImp, top = 20)

# Partial dependence plot
for (i in 3:1) {
  variable <- match(sort(VarImp$importance$Overall, T)[i], VarImp$importance$Overall)
  variable <- names(train)[amatch(rownames(VarImp$importance)[variable], 
                                  names(train), maxDist = 20)]
  xgb %>%
    partial(pred.var = c(variable), chull = T) %>%
    autoplot(contour = T, main = "Attrition", xlab = paste0(variable)) %>% 
    print()
}

# 2 way-Partial dependence plot
for (i in 3:2) {
  for (j in (i - 1):1) {
    variable1 <- match(sort(VarImp$importance$Overall, T)[i], VarImp$importance$Overall)
    variable1 <- names(train)[amatch(rownames(VarImp$importance)[variable1], 
                                     names(train), maxDist = 15)]
    variable2 <- match(sort(VarImp$importance$Overall, T)[j], VarImp$importance$Overall)
    variable2 <- names(train)[amatch(rownames(VarImp$importance)[variable2], 
                                     names(train), maxDist = 15)]
    xgb %>%
      partial(pred.var = c(variable1, variable2), chull = T) %>%
      autoplot(contour = T, 
               main = paste0("Attrition: ", variable1, " vs. ", variable2)) %>% 
      print()
  }
}

# clean up
rm(pr, i, j, variable, variable1, variable2, VarImp)




# Caret Ensemble ----------------------------------------------------------

# load library
p_load(caretEnsemble)

# train some models (with tuning random search 10x)
set.seed(12345)
models <- caretList(Attrition ~ ., data = train,
                    metric = "Accuracy",
                    trControl = trainControl(method = "cv", number = 5, 
                                             search = "random", 
                                             savePredictions = "final",
                                             classProbs = T), 
                    tuneLength = 20,
                    methodList = c("naive_bayes", "earth", "parRF", "gbm", "xgbTree"))
# models

# show correlation between models
resamps <- resamples(list(nb = models$naive_bayes,
                          mars = models$earth,
                          rf = models$parRF,
                          gbm = models$gbm,
                          xgb = models$xgbTree))
cor(resamps$values %>% select(contains("Accuracy")))

# ensemble with GLM
set.seed(12345)
ensemble <- caretEnsemble(models, 
                          metric = "Accuracy",
                          trControl = trainControl(method = "cv", number = 5, 
                                                   classProbs = T))
ensemble
ensemble$ens_model$finalModel$coefficients

# ensemble with xgbTree
set.seed(12345)
ensemble_stacked <- caretStack(models,
                               method = "xgbTree",
                               metric = "Accuracy",
                               trControl = trainControl(method = "cv", number = 5, 
                                                        search = "random", 
                                                        savePredictions = "final",
                                                        classProbs = T),
                               tuneLength = 20)
ensemble_stacked
print(paste0("Maximum Accuracy: ", round(max(ensemble_stacked$ens_model$results$Accuracy), 4)))

