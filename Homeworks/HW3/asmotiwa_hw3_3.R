require(fpp)
require(forecast)

data(books)
plot(books, main = "Data set books")

#This time series has no seasonality but has a trend component in both the categories.

# A) Holt's linear model
holt_fit.Hardcover <- holt(books[,"Hardcover"])
print("The forecast for Hardcover time series is:")
holt_fit.Hardcover

holt_fit.Paperback <- holt(books[,"Paperback"])
print("The forecast for Paperback time series is:")
holt_fit.Paperback


# B) SSE comparison between ses() and holt() method.
ac_holt_fit.Hardcover <- accuracy(holt_fit.Hardcover)
sse_holt.Hardcover <- (ac_holt_fit.Hardcover[2] ^ 2) * length(books[,"Hardcover"])

ac_holt_fit.Paperback <- accuracy(holt_fit.Paperback)
sse_holt.Paperback <- (ac_holt_fit.Paperback[2] ^ 2) * length(books[,"Paperback"])

ses_fit.Hardcover <- ses(books[,"Hardcover"])
ac_ses_fit.Hardcover <- accuracy(ses_fit.Hardcover)
sse_ses.Hardcover <- (ac_ses_fit.Hardcover[2] ^ 2) * length(books[,"Hardcover"])

ses_fit.Paperback <- ses(books[,"Paperback"])
ac_ses_fit.Paperback <- accuracy(ses_fit.Paperback)
sse_ses.Paperback <- (ac_ses_fit.Paperback[2] ^ 2) * length(books[,"Paperback"])

#Printing the SSE values for the ses and holt model
#For Paperback data
#Holt's linear model SSE
sprintf("The SSE for holt's linear model for Paperback data is: %f" , sse_holt.Paperback)
#ses model SSE
sprintf("The SSE for ses model for Paperback data is: %f" , sse_ses.Paperback)

#For Hardcover data
#Holt's linear model SSE
sprintf("The SSE for holt's linear model for Hardcover data is: %f" , sse_holt.Hardcover)
#ses model SSE
sprintf("The SSE for ses model for Hardcover data is: %f" , sse_ses.Hardcover)


# C) Comparing the forecasts for both the methods
holt_fit.Hardcover
ses_fit.Hardcover 
holt_fit.Paperback
ses_fit.Paperback