# nice article on Linear Discriminant analysis:
# https://sebastianraschka.com/Articles/2014_python_lda.html#histograms-and-feature-selection

# IRIS dataset content
str(iris)

# estimating (mu, sigma) from data
K <- length(levels(iris$Species))
est <- lapply(X = 1:K, FUN = function(i){
  dfi <- iris[iris$Species == levels(iris$Species)[i], 1:4]
  list('n' = nrow(dfi), 'mu' = colMeans(dfi), 'sigma' = cov(dfi))
})
names(est) <- levels(iris$Species)

# estimating S_W = \sum_i (n_i - 1)*\Sigma_i and S_B = \sum_i n_i * (\mu_i - \mu)^T (\mu_i - \mu)
global_mu <- colMeans(iris[,1:4])
sw <- array(data = 0, dim = c(4, 4))
sb <- array(data = 0, dim = c(4, 4))
for(i in 1:K){
  sw <- sw + (est[[i]]$n-1) * est[[i]]$sigma
  sb <- sb + est[[i]]$n * (est[[i]]$mu - global_mu) %*% t(est[[i]]$mu - global_mu)
}

# get SVD from (S_W)^{-1} S_B
eig <- eigen(solve(sw) %*% sb)

# 99% of variability regarding species is contained in first vector dimension
cumsum(eig$values)/sum(eig$values)

# computing projection of X on the subspace found
lda_dim1 <- as.matrix(iris[,1:4]) %*% eig$vectors[,1]
boxplot(lda_dim1 ~ iris$Species, main = 'Projection of X on the first LDA dimension found')

# testing our results with baseline function: lda()
m_lda <- lda(formula = Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width, data = iris)
m_lda$scaling
eig$vectors[,1:2] 

# same as LDA dimensions up to some scale factor:
m_lda$scaling[,1] / eig$vectors[,1]
