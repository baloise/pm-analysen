


filter()
arrange()

select()
  -one_of()
  one_of()
  starts_with()
  ends_with()
  contains()
  matches()
  everything()
mutate()

group_by()
summarise()
  n()
  n()/nrow()
  n_distinct()
  mean()
  median()

inner_join()
left_join()
full_join()
anti_join()

bind_rows()
bind_cols()

distinct()
slice_()






# Feature Engineering function: accepts data frame, returns data frame
featureEngineer <- function(df) {
  
  # convert season, holiday, workingday and weather into factors
  names <- c("season", "holiday", "workingday", "weather")
  df[,names] <- lapply(df[, names], factor)
  
  # Convert datetime into timestamps (split day and hour)
  df$datetime <- as.character(df$datetime)
  df$datetime <- strptime(df$datetime, format = "%Y-%m-%d %T", tz = "EST") #tz removes timestamps flagged as "NA"
  
  # extract day of the week
  df$weekday <- as.factor(weekdays(df$datetime, abbreviate = F))
  df$weekday <- factor(df$weekday, 
                       levels = c("Montag", "Dienstag", "Mittwoch", "Donnerstag", 
                                  "Freitag", "Samstag", "Sonntag"))
  
  # extract year from date and convert to factor to represent yearly growth
  df$year <- as.integer(substr(df$datetime, 1, 4))
  df$year <- as.factor(df$year)
  
  # return full featured data frame
  return(df)
} 

# Build features for train and Test set
train <- featureEngineer(train)
test <- featureEngineer(test)


