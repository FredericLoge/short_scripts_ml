# function returning minus the log-likelihood in the logit model
foo <- function(beta, y, x){
  ll <- sum(y*(x %*% beta) - log(1+exp(x %*% beta))) 
  return(-ll)
}

# get iris data
df <- iris[iris$Species %in% c('versicolor', 'virginica'),]
x <- cbind('Intercept'=1, as.matrix(df[,c('Sepal.Length', 'Sepal.Width', 'Petal.Length')]))
y <- 1*(df[,c('Species')] == 'virginica')

# find optimal beta
beta0 <- rep(0, ncol(x))
foo(beta = beta0, y=y, x=x)
opt <- optim(par = beta0, fn = foo, y = y, x = x)
opt

# compute probability estimate
xbeta <- x %*% opt$par
pred_virginica <- exp(xbeta) / (1 + exp(xbeta))

# confusion matrix
table(y, 1*(pred_virginica > 0.5))

# compute ROC curve
compute_roc <- function(y, yhat){
  seuils <- seq(0, 1, by = 0.01)
  coords <- array(data = NA, dim = c(length(seuils)+1, 2))
  for(i in 1:length(seuils)){
    pred <- 1*(yhat >= seuils[i])
    coords[i,1] <- sum(pred==1 & y==1)/sum(y==1)
    coords[i,2] <- sum(pred==1 & y==0)/sum(y==0)
  }
  coords[length(seuils)+1,] <- c(0,0)
  return(coords)
}

# compute and plot ROC
myRoc <- compute_roc(y = y, yhat = pred_virginica)
plot(y = myRoc[,1], x = myRoc[,2], xlab = 'False Positive Rate', ylab = 'True Positive Rate',
        main = 'ROC curve', type = 'l', pch = 20)
segments(x0 = 0, y0 = 0, x1 = 1, y1 = 1, lty = 2)

# plot sigmoid with colored points
color_vector <- c('blue', 'orange')
plot(y = pred_virginica, x = xbeta, col = as.character(factor(x = y, levels = c(0,1), labels = color_vector)), 
     pch = 20, xlab = expression(paste('x', beta)),
     ylab = expression(paste(exp,'(x', beta,') / (1 +', exp, '(x', beta, '))')),
     main = expression(paste('Prob(Species = Virginica) vs x', beta, ' with real species colored')))
legend('bottomright', legend = c('versicolor', 'virginica'), col = color_vector,
       pch = 20, xpd = TRUE, border = NA, bg = 'yellow', box.col = NA, cex = 2)

