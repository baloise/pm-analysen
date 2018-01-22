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
p_load(dplyr) 

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




# Start h2o ---------------------------------------------------------------

# load library
p_load(h2o)

# start h2o
# h2o.init()
h2o.init(max_mem_size = "2g")
# h2o.shutdown(prompt = T)

# convert data.frames to H2OFrame
train_h2o <- as.h2o(train)
test_h2o <- as.h2o(test)
y <- "Attrition"
x <- setdiff(names(train_h2o), y)




# h2o.randomForest --------------------------------------------------------

# train model
rf <- h2o.randomForest(x = x, y = y,
                       training_frame = train_h2o,
                       validation_frame = test_h2o,
                       ntrees = 500,
                       max_runtime_secs = 360,
                       stopping_metric = "accuracy",
                       seed = 12345)
rf
# h2o.gainsLift(rf, test_h2o)

# variable importance
VarImp <- h2o.varimp(rf); VarImp %>% as.data.frame()
# h2o.varimp_plot(rf)
ggplot(VarImp[1: min(20, nrow(VarImp)), ],
       aes(x = reorder(variable, scaled_importance), y = scaled_importance)) +
  geom_bar(stat = "identity") +
  labs(x = "variable") +
  coord_flip()
print(VarImp$variable[1:10])
# [1] "feature35" "feature24" "feature33" "feature46" "feature40"
# [6] "feature32" "feature6"  "feature50" "feature16" "feature13"

# partial dependence plot
h2o.partialPlot(rf, train_h2o, cols = c(VarImp$variable[1:2]), plot_stddev = F)
# h2o.partialPlot(rf, train_h2o, cols = c("feature35"), plot_stddev = F)
# h2o.partialPlot(rf, train_h2o, cols = c("feature35", "feature24"), plot_stddev = F)




# h2o.gbm -----------------------------------------------------------------

gbm <- h2o.gbm(x = x, y = y,
               training_frame = train_h2o,
               validation_frame = valid_h2o,
               # leaderboard_frame = tournament_h2o,
               max_runtime_secs = 360,
               stopping_metric = "logloss",
               stopping_rounds = 50,
               seed = 12345)
gbm




# h2o.automl --------------------------------------------------------------

aml <- h2o.automl(x = x, y = y,
                  training_frame = train_h2o,
                  validation_frame = valid_h2o,
                  leaderboard_frame = tournament_h2o,
                  max_runtime_secs = 3600,   # 30 * 60
                  stopping_metric = "logloss",
                  stopping_rounds = 30,
                  seed = 12345)
h2o.getFrame("leaderboard") %>% as.data.frame()

# Extract leaderboard and leader model
lb <- aml@leaderboard; lb

lb_leader <- aml@leader; lb_leader












# Random Forrest ----------------------------------------------------------

p_load(randomForest)

rf <- randomForest(Attrition~., data = train, ntree = 2000, mtry = 14, importance = T)
rf

# Parametertuning mtry
tuneRF(train %>% select(-Attrition), train$Attrition, 
       ntreeTry = 2000, mtryStart = 10, stepFactor = 1.2, improve = 0.0001, 
       trace = T, plot = T, doBest = F)

# plot how error developes during training process
plot(rf)

# plot importance of each variable
varImpPlot(rf, sort = T, scale = T)

# Partial Dependence Plot
p_load(plotmo)
plotmo(rf, nrug = "density", pmethod = "apartdep", ylim = NULL, degree2 = F)




# GBM ---------------------------------------------------------------------

