require(fma)
require(fpp)

##########################################################################

#usnetelec data
data("usnetelec")
plot(usnetelec)
plot(stl(usnetelec, s.window = 12))
#This data does not contain the seasonal component. So, only mean has to be stabilized to make the data horizontal
#Stabilizing the mean using differencing
new_usnetelec <- diff(usnetelec)
plot(new_usnetelec)
acf(new_usnetelec)

#Stabilizing the variance
lambda <- BoxCox.lambda(new_usnetelec)
new_usnetelec <- BoxCox(new_usnetelec,lambda)
plot(new_usnetelec)

##########################################################################

#usgdp data
data("usgdp")
plot(usgdp)
plot(stl(usgdp, s.window = 12))

#Stabilizing the mean using double differencing, as acf of double differenced data drops to 0 quickly.
new_usgdp <- (diff(diff(usgdp)))
plot(new_usgdp)
acf(new_usgdp)

#Stabilizing the variance
lambda <- BoxCox.lambda(new_usgdp)
new_usgdp <- BoxCox(new_usgdp,lambda)
plot(new_usgdp)

##########################################################################

#mcopper data
data("mcopper")
plot(mcopper)
plot(stl(mcopper, s.window = 12))

#Using differencing to stabilize the mean.
new_mcopper <- diff(mcopper)
plot(new_mcopper)

#Stabilizing the variance
lambda <- BoxCox.lambda(new_mcopper)
new_mcopper <- BoxCox(new_mcopper,lambda)
plot(new_mcopper)

#Since, the acf of the transformed data drops to 0 quickly, it is a stationary data
acf(new_mcopper)

##########################################################################

#enplanements data
data("enplanements")
plot(enplanements)
plot(stl(enplanements, s.window = 12))

#Seasonally adjusting the data
fit <- decompose(enplanements, type = "multiplicative")
seas <- seasadj(fit)
plot(seas)
plot(stl(seas, s.window = 12))

#Stabilizing the variance
lambda <- BoxCox.lambda(seas)
new_enplanements <- BoxCox(seas,lambda)
plot(new_enplanements)
plot(stl(new_enplanements, s.window = 12))

#Using double differencing to stabilize the mean.
new_enplanements <- diff(diff(new_enplanements))
plot(new_enplanements)

#Since, the acf of the transformed data drops to 0 quickly, it is a stationary data
acf(new_enplanements)
 
##########################################################################

#visitors data
data("visitors")
plot(visitors)
plot(stl(visitors, s.window = 12))

#Seasonally adjusting the data
fit2 <- decompose(visitors, type = "multiplicative")
seas2 <- seasadj(fit2)
plot(seas2)
plot(stl(seas2, s.window = 12))

#Stabilizing the variance
lambda <- BoxCox.lambda(seas2)
new_visitors <- BoxCox(seas2,lambda)
plot(new_visitors)
plot(stl(new_visitors, s.window = 12))


#Using differencing to stabilize the mean.
new_visitors <- diff(new_visitors)
plot(new_visitors)

#Since, the acf of the transformed data drops to 0 quickly, it is a stationary data
acf(new_visitors)