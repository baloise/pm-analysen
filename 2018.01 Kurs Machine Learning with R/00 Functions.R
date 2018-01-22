remove_columns_with_only_one_value <- function(df) {
  # print summary before processing
  print("Dimensions of data.frame before processing:")
  print(dim(df))
  # save column names for the comparison later
  temp <- names(df)
  # filter columns with only 1 distinct value
  df <- Filter(function(x) (n_distinct(x, na.rm = F) > 1), df)
  # print summary after processing
  print("Dimensions of data.frame after processing:")
  print(dim(df))
  print("Removed columns:")
  print(setdiff(temp, names(df)))
  # return data.frame
  return(df)
}



plot_y_vs_ydach <- function(obs, pr){
  df <- data.frame(y = obs[], pr = pr[])
  p <- ggplot(df, aes(x = y, y = pr)) + 
    geom_point() +
    geom_abline(intercept = 0, slope = 1, colour = "red", linetype = "dashed") +
    labs(y = "model prediction", title = "y vs. model prediction",
         subtitle = paste0("RMSE:    ", round(rmse(df$y, df$pr), 3),
                           "\nMAE:      ", round(Metrics::mae(df$y, df$pr), 3),
                           "\nMAPE:    ", round(sum(abs(df$pr / df$y - 1)) / length(df$y), 3),
                           "\nR2:          ", format(cor(df$y, df$pr)^2, digits = 3)))
  print(p)
}
