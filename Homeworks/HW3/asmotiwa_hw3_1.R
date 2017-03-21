require(fpp) 
data(dj)
plot(dj)
#The data is also not horizontal indicating that it is not a stationary data.

acf(dj)
Box.test(dj, lag=10, fitdf=0, type="Lj")
#Since, the p-value is significant, the data is not a white noise.

#Performing the unit root test.
ns <- nsdiffs(dj)
if(ns > 0){
  djstar <- diff(dj, lag = frequency(dj), differencs = ns)
} else{
  djstar <- dj
}
nd <- ndiffs(djstar)
if(nd > 0){
  djstar <- diff(djstar, differences = nd)
}

#Running the given code again on the transformed dj data.
plot(djstar)
#The data is now horizontal indicating that it is a stationary data.

acf(djstar)
#From the plot it is visible that the data is now white noise and the mean is stabilized.

Box.test(djstar, lag=10, fitdf=0, type="Lj")
#From the box test, we get the p-value to be insignificant. Hence, the data is white noise.
