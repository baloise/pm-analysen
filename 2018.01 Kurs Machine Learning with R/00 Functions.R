Remove_columns_with_only_one_value <- function(df) {
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

