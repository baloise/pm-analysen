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
p_load(dplyr, ggplot2, gridExtra, caret, reshape2, Metrics) 

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
df$Attrition <- ifelse(df$Attrition == "No", 0, 1)

# Remove columns with only 1 value
df <- Remove_columns_with_only_one_value(df)



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
# 2.  XGBOOST               0.???      0.???     
# 3.  GBM                   0.???      0.???     
# 4.  randomforrest         0.855      0.???     
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
rf <- randomForest(as.factor(Attrition)~., data = train, ntree = 1000, importance = T)
rf

# train model with specific mtry
rf <- randomForest(Attrition~., data = train, ntree = 1000, mtry = 10, importance = T)
rf

# show confusion matrix
pr <- predict(rf, test %>% select(-Attrition))
confusionMatrix(pr, as.factor(test$Attrition))

# plot how error developes during training process
plot(rf)

# plot importance of each variable
varImpPlot(rf, sort = T, scale = T)

# Partial Dependence Plot
p_load(plotmo)
plotmo(rf, nrug = "density", pmethod = "apartdep", ylim = NULL, degree2 = F)



# Parametertuning: mtry (default = floor(sqrt(ncol(train))) = 5)
tuneRF(train %>% select(-Attrition), as.factor(train$Attrition), 
       ntreeTry = 5000, mtryStart = 10, stepFactor = 1.3, improve = 0.00001, 
       trace = T, plot = T, doBest = F)

# Schleife, um den besten Parameter mtry zu finden
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
Summary
ggplot(data = melt(Summary, id = "mtry"), aes(x = mtry, y = value, colour = variable)) + 
  geom_line()
print(paste0("Maximum accuracy at mtry = ", Summary$mtry[which.max(Summary$test)]))



# clean up
rm(rf, Summary, mtry, i)




# GBM ---------------------------------------------------------------------

