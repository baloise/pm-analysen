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
p_load(dplyr, ggplot2, gridExtra) 

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
h2o.init(max_mem_size = "2g")
# h2o.shutdown(prompt = F); gc()

# convert data.frames to H2OFrame
train_h2o <- as.h2o(train)
test_h2o <- as.h2o(test)
y <- "Attrition"
x <- setdiff(names(train_h2o), y)




# h2o.randomForest --------------------------------------------------------

# CV accuracy = 0.867 resp. 0.870

# train model
rf <- h2o.randomForest(x = x, y = y,
                       training_frame = train_h2o, validation_frame = test_h2o,
                       ntrees = 100, mtries = -1, max_depth = 20,
                       nfolds = 5, fold_assignment = "Modulo", 
                       keep_cross_validation_predictions = T,
                       max_runtime_secs = 360, seed = 12345)
rf

# variable importance
VarImp <- h2o.varimp(rf)
h2o.varimp_plot(rf, num_of_features = 10)

# partial dependence plot
h2o.partialPlot(rf, train_h2o, cols = VarImp$variable[5:1], plot_stddev = F)



### Grid search for good hyperparameters

# RF hyperparameters to try out
rf_params <- list(mtries = c(3, 4, 5, 6, 7, 8),
                  max_depth = c(8, 10, 15, 20, 25))
search_criteria <- list(strategy = "Cartesian")

# Train and validate a cartesian grid of RFs
rf_grid <- h2o.grid("randomForest", grid_id = "rf_grid",
                    hyper_params = rf_params, search_criteria = search_criteria,
                    x = x, y = y,
                    training_frame = train_h2o, validation_frame = test_h2o,
                    ntrees = 100,
                    nfolds = 5, fold_assignment = "Modulo", 
                    keep_cross_validation_predictions = T,
                    seed = 12345)

# Get the grid results, sorted by cv accuracy
rf_gridperf <- h2o.getGrid(grid_id = "rf_grid", sort_by = "accuracy", decreasing = T)
print(rf_gridperf)
ggplot(rf_gridperf@summary_table %>% 
         as.data.frame() %>% 
         select(-model_ids) %>% 
         mutate(max_depth = as.factor(max_depth),
                mtries = as.numeric(mtries),
                accuracy = as.numeric(accuracy)), 
       aes(x = mtries, y = accuracy, colour = max_depth)) +
  geom_line() + geom_point()

# Get the best model, chosen by cv accuracy
rf_best <- h2o.getModel(rf_gridperf@model_ids[[1]])
rf_best

# h2o.rm("rf_grid")




# h2o.gbm -----------------------------------------------------------------

# CV accuracy = 0.869 resp. 0.881

# train model
gbm <- h2o.gbm(x = x, y = y,
               training_frame = train_h2o, validation_frame = test_h2o,
               ntrees = 1000, learn_rate = 0.1, max_depth = 5,
               stopping_rounds = 30, stopping_metric = "AUTO",
               nfolds = 5, fold_assignment = "Modulo", keep_cross_validation_predictions = T,
               max_runtime_secs = 360, seed = 12345)
gbm

# variable importance
VarImp <- h2o.varimp(gbm)
h2o.varimp_plot(gbm, num_of_features = 10)

# partial dependence plot
h2o.partialPlot(gbm, train_h2o, cols = VarImp$variable[5:1], plot_stddev = F)



### Randomised grid search for good hyperparameters (too many parameters for grid search)

# GBM hyperparameters to try out
gbm_params <- list(max_depth = seq(from = 2, to = 10, by = 1),
                   sample_rate = seq(from = 0.5, to = 1.0, by = 0.1),
                   col_sample_rate = seq(from = 0.1, to = 1.0, by = 0.1))
# search_criteria <- list(strategy = "RandomDiscrete", max_models = 30, seed = 12345)
search_criteria <- list(strategy = "RandomDiscrete", max_runtime_secs = 60 * 10, seed = 12345)

# Train and validate a random grid of GBMs
gbm_grid <- h2o.grid("gbm", grid_id = "gbm_grid", 
                     hyper_params = gbm_params, search_criteria = search_criteria,
                     x = x, y = y,
                     training_frame = train_h2o, validation_frame = test_h2o,
                     ntrees = 1000, learn_rate = 0.1,
                     stopping_rounds = 30, stopping_metric = "AUTO",
                     nfolds = 5, fold_assignment = "Modulo", 
                     keep_cross_validation_predictions = T,
                     seed = 12345)

# Get the grid results, sorted by cv accuracy
gbm_gridperf <- h2o.getGrid(grid_id = "gbm_grid", sort_by = "accuracy", decreasing = T)
print(gbm_gridperf)
a <- ggplot(gbm_gridperf@summary_table %>% 
              as.data.frame() %>% 
              select(-model_ids) %>% 
              mutate_all(funs(as.numeric)), 
            aes(x = max_depth, y = accuracy)) +
  geom_point() + geom_smooth(method = "loess", span = 0.6)
b <- ggplot(gbm_gridperf@summary_table %>% 
              as.data.frame() %>% 
              select(-model_ids) %>% 
              mutate_all(funs(as.numeric)), 
            aes(x = col_sample_rate, y = accuracy)) +
  geom_point() + geom_smooth(method = "loess", span = 0.6)
c <- ggplot(gbm_gridperf@summary_table %>% 
              as.data.frame() %>% 
              select(-model_ids) %>% 
              mutate_all(funs(as.numeric)), 
            aes(x = sample_rate, y = accuracy)) +
  geom_point() + geom_smooth(method = "loess", span = 0.6)
grid.arrange(a, b, c, ncol = 3)
rm(a, b, c)

# Get the best model, chosen by cv accuracy
gbm_best <- h2o.getModel(gbm_gridperf@model_ids[[1]])
gbm_best

# h2o.rm("gbm_grid")




# h2o.stackedEnsemble -----------------------------------------------------

# accuracy = 0.882 resp. 0.891

# train model
ens <- h2o.stackedEnsemble(x = x, y = y,
                           training_frame = train_h2o, validation_frame = test_h2o,
                           base_models = list(rf, gbm))
ens

ens_best <- h2o.stackedEnsemble(x = x, y = y,
                                training_frame = train_h2o, validation_frame = test_h2o,
                                base_models = list(rf_best, gbm_best))
ens_best




# h2o.automl --------------------------------------------------------------

# accuracy = 0.881

# train models
aml <- h2o.automl(x = x, y = y,
                  training_frame = train_h2o, validation_frame = test_h2o,
                  stopping_rounds = 30, stopping_metric = "AUTO",
                  nfolds = 5,
                  max_runtime_secs = 60 * 10, seed = 12345)
h2o.getFrame("leaderboard") %>% as.data.frame()

# Show leader model
aml@leader
