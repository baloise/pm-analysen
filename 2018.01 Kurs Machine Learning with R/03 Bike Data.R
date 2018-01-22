







# Feature Engineering function: accepts data frame, returns data frame
featureEngineer <- function(df) {
  
  # Ausreisser trimmen
  
  # Clean variables with too many categories
  
  # Zahlen-Missings durch MEDIAN ersetzen
  for (i in which(sapply(train, is.numeric))) {
    train[is.na(train[, i]), i] <- median(train[, i], na.rm=T)
  }
  
  # Faktor-Missings durch MODUS ersetzen
  for (i in which(sapply(train, is.factor))) {
    train[is.na(train[, i]), i] <- names(sort(-table(train[, i])))[1]
  }
  
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


