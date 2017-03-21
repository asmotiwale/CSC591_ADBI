require(fpp)
require(forecast)
data(ukcars)
plot(ukcars, ylab = "Production, thousands of cars")
stlFit <- stl(ukcars, s.window = "periodic")
plot(stlFit)
adjusted <- seasadj(stlFit)
plot(adjusted)

fcastHoltDamp = holt(adjusted, damped=TRUE, h = 8)
plot(ukcars, xlim = c(1977, 2008))
lines(fcastHoltDamp$mean + 
        stlFit$time.series[2:9,"seasonal"], 
      col = "red", lwd = 2)

dampHoltRMSE = sqrt(mean(((fcastHoltDamp$fitted + stlFit$time.series[,"seasonal"]) - ukcars)^2))
dampHoltRMSE

fcastHolt = holt(adjusted, h = 8)
plot(ukcars, xlim = c(1997, 2008))
lines(fcastHolt$mean + stlFit$time.series[2:9,"seasonal"], 
      col = "red", lwd = 2)

holtRMSE = sqrt(mean(((fcastHolt$fitted + stlFit$time.series[,"seasonal"]) - ukcars)^2))
holtRMSE

######################################################################################################

#A)
ukcars.ets <- ets(ukcars)
accuracy(ukcars.ets)
# The ets model selects the (A,N,A) model as the best model.

#B)
plot(forecast(ukcars.ets, h = 8))
plot(forecast(fcastHolt))
plot(forecast(fcastHoltDamp))

f.ets <- forecast(ukcars.ets)
f.dampHolt <- fcastHoltDamp$mean + stlFit$time.series[2:9,"seasonal"]
rmse_f.ets_damp <- sqrt(mean(((f.ets$mean - f.dampHolt)^2)))

f.Holt <- fcastHolt$mean + stlFit$time.series[2:9,"seasonal"]
rmse_f.ets_damp <- sqrt(mean(((f.ets$mean - f.Holt)^2)))
# Since, the damped Holt model shows less deviation from the ets model(which selects the best model) as compared to the Holt model which can be
# seen by the rmse value between the ets model and the 2 models implemented by Sam.

