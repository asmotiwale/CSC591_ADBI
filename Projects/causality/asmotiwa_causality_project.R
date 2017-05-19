## Name: Anuraag Motiwale
## Student ID: 200155847
## Unity ID: asmotiwa

# Load the libraries 
# To install pcalg library you may first need to execute the following commands:
 source("https://bioconductor.org/biocLite.R")
 biocLite("graph")
 biocLite("RBGL")
 
 biocLite("Rgraphviz")
 library(pcalg)
 library(vars)
 library(urca)
 library(stats)
 library(igraph)
 
# Read the input data 
data.f <- read.csv('data.csv',header = TRUE)

# Build a VAR model 
var.model <- VAR(data.f, type = c("const"),ic = c("SC"))

# Select the lag order using the Schwarz Information Criterion with a maximum lag of 10
# see ?VARSelect to find the optimal number of lags and use it as input to VAR()
var.select <- VARselect(data.f, lag.max = 10)

# Extract the residuals from the VAR model 
# see ?residuals
var.residuals <- residuals(var.model)

# Check for stationarity using the Augmented Dickey-Fuller test 
# see ?ur.df
# Checking for 'Move'
ur.Move <- ur.df(var.residuals[, 1])

# Checking for 'RPRICE'
ur.RPRICE <- ur.df(var.residuals[, 2])

# Checking for 'MPRICE'
ur.MPRICE <- ur.df(var.residuals[, 3])

## Since, the test statistic is less than 0.05 for all the three tests,
## we fail to reject the null hypothesis.

# Check whether the variables follow a Gaussian distribution  
# see ?ks.test
gd.ks1 <- ks.test(var.residuals[, 1], rnorm(100))
gd.ks2 <- ks.test(var.residuals[, 2], rnorm(100))
gd.ks3 <- ks.test(var.residuals[, 3], rnorm(100))

### Since, the p-values for all the residuals of all variables are less than 0.05, 
### we fail to reject the null hypothesis.

# Write the residuals to a csv file to build causal graphs using Tetrad software
write.csv(var.residuals[,1:3], file = 'residuals.csv', row.names = FALSE)

# OR Run the PC and LiNGAM algorithm in R as follows,
# see ?pc and ?LINGAM 

# PC Algorithm
suffStat = list(C = cor(var.residuals),n = nrow(var.residuals))
pc.fit <- pc(suffStat, indepTest = gaussCItest,alpha = 0.1, labels = colnames(var.residuals), verbose = TRUE)

# LiNGAM Algorithm
lingam.fit <-LINGAM(var.residuals, verbose = TRUE)
show(lingam.fit)