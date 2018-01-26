# Source of data ---------------------------------------------------------
# 
# https://www.kaggle.com/c/bike-sharing-demand/data
# 
# Data Fields:
# 
# datetime: date and hour  
# season: 
#   1: spring
#   2: summer
#   3: fall
#   4: winter 
# holiday: whether the day is considered a holiday
# workingday: whether the day is neither a weekend nor holiday
# weather
#   1: Clear, Few clouds, Partly cloudy, Partly cloudy 
#   2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist 
#   3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + 
#      Scattered clouds 
#   4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog 
# temp: temperature in Celsius
# atemp: "feels like" temperature in Celsius
# humidity: relative humidity
# windspeed: wind speed
# casual: number of non-registered user rentals initiated
# registered: number of registered user rentals initiated
# count: number of total rentals




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
p_load(funModeling, dplyr, ggplot2, caret, pdp, gbm, tibble, lubridate)

# source some additional functions
source("00 Functions.R")




# Read and prepare data ---------------------------------------------------

### read data
df_raw <- read.csv("03 Bike Sharing Demand - data.csv")
# df_raw <- read.csv("03 Bike Sharing Demand - data.csv") %>% 
#   slice(1:1000)
#   sample_n(1000)



### check dataset health status
df_status(df_raw)



### Prepare data


data_preparation <- function(df, is_train = T) {
  # Change class of datetime
  df$datetime <- as.POSIXct(df$datetime)
  # extract date
  df$date <- as.Date(df$datetime)
  # extract time
  df$time <- hour(df$datetime)
  # change class of season, holiday, workingday, weather
  if (is_train) {
    df <- df %>%
      mutate_at(vars(season, holiday, workingday, weather), factor)
  } else {
    df$season <- factor(df$season, levels = levels(train$season))
    df$holiday <- factor(df$holiday, levels = levels(train$holiday))
    df$workingday <- factor(df$workingday, levels = levels(train$workingday))
    df$weather <- factor(df$weather, levels = levels(train$weather))
  }
  # some more feature engineering
  # ...
  # 
  # # Ausreisser trimmen
  # # ...
  # # 
  # # Clean variables with too many categories
  # # ...
  # # 
  # # Zahlen-Missings durch MEDIAN ersetzen
  # for (i in which(sapply(train, is.numeric))) {
  #   train[is.na(train[, i]), i] <- median(train[, i], na.rm=T)
  # }
  # # 
  # # Faktor-Missings durch MODUS ersetzen
  # for (i in which(sapply(train, is.factor))) {
  #   train[is.na(train[, i]), i] <- names(sort(-table(train[, i])))[1]
  # }
  # # 
  # # extract day of the week
  # df$weekday <- as.factor(weekdays(df$datetime, abbreviate = F))
  # df$weekday <- factor(df$weekday, 
  #                      levels = c("Montag", "Dienstag", "Mittwoch", "Donnerstag", 
  #                                 "Freitag", "Samstag", "Sonntag"))
  # # 
  # # extract year from date and convert to factor to represent yearly growth
  # df$year <- as.integer(substr(df$datetime, 1, 4))
  # df$year <- as.factor(df$year)
  # 
  # remove unnecesary variables
  df <- df %>% select(-datetime, -casual, -registered)
  # Remove columns with only 1 value
  df <- remove_columns_with_only_one_value(df)
}



### Generate train and test data

# generate a new column and mark some 30% of data as test-data
set.seed(12345)
df_raw <- df_raw %>% 
  mutate(test_data = sample(c(0, 1), 
                            nrow(df_raw), 
                            replace = T, 
                            prob = c(0.8, 0.2)))
# count number of 0/1 of df_raw$test
table(df_raw$test_data)

# select train-data: data to train the models on
train <- df_raw %>% 
  filter(test_data == 0) %>% 
  select(-test_data)
train <- data_preparation(train, is_train = T)

# select test-data: on this data we evaluate model performance
test <- df_raw %>% 
  filter(test_data == 1) %>% 
  select(-test_data)
test <- data_preparation(test, is_train = F)




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
featurePlot(x = train %>% select_if(is.numeric) %>% select(-count), y = train$count, 
            plot = "scatter", type = c("p", "smooth"), span = 0.5,
            layout = c(3, 1), auto.key = list(columns = 3),  labels = c("", "count"))

# categoric variables vs. y
featurePlot(x = train %>% select_if(is.factor) %>% data.matrix(),
            y = train$count,
            plot = "scatter", type = c("p", "smooth"), span = 0.5,
            layout = c(3, 1), auto.key = list(columns = 3),  labels = c("", "count"))




# Summary of results ------------------------------------------------------
#                             
# Rg  Model                  RMSE     Comment
# 1.  ensemble XGBOOST      ???       
# 2.  XGBOOST               ???       
# 3.  GBM                   ???       
# 4.  mars                  ???       
# 5.  ensemble GLM          ???       
# 6.  randomforrest         ???       
# 8.  base line             183.0     average value as prediction




# base line ---------------------------------------------------------------
# average value as prediction

# calculate RMSE
RMSE(mean(train$count), test$count)




# Multiple additive regression splines ------------------------------------

# train model (random search 20x)
set.seed(12345)
mars <- train(count ~ ., data = train,
              method = "earth", metric = "RMSE",
              trControl = trainControl(method = "cv", number = 5, search = "random"),
              tuneLength = 20)
mars

print(paste0("Maximum Accuracy: ", round(max(mars$results$Accuracy), 4)))
ggplot(mars$results, aes(x = nprune, y = Accuracy, colour = degree)) + 
  geom_point()

# show confusion matrix
pr <- predict(mars, test %>% select(-count), type = "prob")
confusionMatrix(ifelse(pr$Yes > 0.5, "Yes", "No"), test$count)

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
rf <- train(count ~ ., data = train,
             method = "parRF", metric = "Accuracy",
             trControl = trainControl(method = "cv", number = 5, search = "random"), 
             tuneLength = 20, verbose = F)
rf
print(paste0("Maximum Accuracy: ", round(max(rf$results$Accuracy), 4)))
ggplot(rf, metric = "Accuracy")

# show confusion matrix
pr <- predict(rf, test %>% select(-count), type = "prob")
confusionMatrix(ifelse(pr$Yes > 0.5, "Yes", "No"), test$count)

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
    autoplot(contour = T, main = "count", xlab = paste0(variable)) %>% 
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
               main = paste0("count: ", variable1, " vs. ", variable2)) %>% 
      print()
  }
}

# clean up
rm(VarImp, i, j, variable, variable1, variable2)




# GBM ---------------------------------------------------------------------

# train model (random search 20x)
set.seed(12345)
gbm <- train(count ~ ., data = train,
             method = "gbm", metric = "Accuracy",
             trControl = trainControl(method = "cv", number = 5, search = "random"), 
             tuneLength = 20, verbose = F)
gbm
print(paste0("Maximum Accuracy: ", round(max(gbm$results$Accuracy), 4)))
ggplot(gbm, metric = "Accuracy")

# show confusion matrix
pr <- predict(gbm, test %>% select(-count), type = "prob")
confusionMatrix(ifelse(pr$Yes > 0.5, "Yes", "No"), test$count)

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
    autoplot(contour = T, main = "count", xlab = paste0(variable)) %>% 
    print()
}

# # 2 way-Partial dependence plot (slow!)
# for (i in 3:2) {
#   for (j in (i - 1):1) {
#     variable1 <- names(train)[amatch(VarImp$variable[i], names(train), maxDist = 15)]
#     variable2 <- names(train)[amatch(VarImp$variable[j], names(train), maxDist = 15)]
#     gbm %>%
#       partial(pred.var = c(variable1, variable2), chull = T) %>%
#       autoplot(contour = T, main = "count", xlab = paste0(variable1, " vs ", variable2)) %>% 
#       print()
#   }
# }

# clean up
rm(pr, VarImp, i, j, variable, variable1, variable2)




# XGBOOST -----------------------------------------------------------------

# train model (random search 20x)
set.seed(12345)
xgb <- train(count ~ ., data = train,
             method = "xgbTree", metric = "Accuracy",
             trControl = trainControl(method = "cv", number = 5, search = "random"), 
             tuneLength = 20)
xgb
print(paste0("Maximum Accuracy: ", round(max(xgb$results$Accuracy), 4)))
ggplot(xgb, metric = "Accuracy")

# show confusion matrix
pr <- predict(xgb, test %>% select(-count), type = "prob")
confusionMatrix(ifelse(pr$Yes > 0.5, "Yes", "No"), test$count)

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
    autoplot(contour = T, main = "count", xlab = paste0(variable)) %>% 
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
               main = paste0("count: ", variable1, " vs. ", variable2)) %>% 
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
models <- caretList(count ~ ., data = train,
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

