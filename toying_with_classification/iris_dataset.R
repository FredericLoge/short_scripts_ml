# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
#                               Descriptive Analysis of IRIS dataset
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# dataset description
str(iris)

# the database contains the following attributes: 
# 1) sepal length in cm
# 2) sepal width in cm
# 3) petal length in cm
# 4) petal width in cm 
# 5) species: Setosa/Versicolour/Virginica

# this toy dataset is used for two main tasks
# A) Learning the relationship between Species and Physical attributes
# B) Learning the relationship amongst physical attributes

# visualize bi-plots with points colored by species
color_vector <- c('black', 'blue', 'orange')
plot(iris[,-5], col = color_vector[iris[,5]], pch = 20, 
     main = 'Bi-plots, species-colored (legend at bottom right)')
legend(legend = levels(iris[,5]), col = color_vector,
       x = 1 - 0.12, y = 0 + 0.25, pch = 20, xpd = TRUE, border = NA, bg = 'yellow', box.col = NA)

# note that the setosa species has very different physical
# characteristics compared to the two other, versicolor and
# virginica, which are much more entangled.

# let us build a classifier to identify versicolor VS virginica.

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
#                                 Classifiying versicolor VS virginica
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

require(MASS)
require(class)

# extract iris of types versicolor and virginica
iris_v <- iris[iris$Species!='setosa',]

# the data is class uniform
table(iris_v$Species)

# refactoring
iris_v$Target <- factor(x = as.character(iris_v$Species), levels = c('versicolor', 'virginica'))

# visualize bi-plots with points colored by species
color_vector <- c('blue', 'orange')
plot(iris_v[,1:4], col = color_vector[iris_v$Target], pch = 20, 
     main = 'Bi-plots, species-colored (legend at bottom right)')
legend(legend = levels(iris_v$Target), col = color_vector,
       x = 1 - 0.12, y = 0 + 0.25, pch = 20, xpd = TRUE, border = NA, bg = 'yellow', box.col = NA)

# randomized train-test split
set.seed(375)
iris_v <- iris_v[sample(1:nrow(iris_v)),]
index_train <- 1:floor(nrow(iris_v)/2)
iris_v_train <- iris_v[index_train,]
iris_v_test <- iris_v[-index_train,]

# logistic regression (LR)
m_logreg <- glm(formula = Target ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width, data = iris_v_train, family = binomial(link = 'logit'))
summary(m_logreg)
m_logreg_prob <- predict(object = m_logreg, newdata = iris_v_test, type = 'response')
m_logreg_pred <- factor(x = (as.numeric(m_logreg_prob) < 0.5), 
                        levels = c(TRUE, FALSE), labels = levels(iris_v_train$Target))

# LDA (MASS::lda())
m_lda <- lda(formula = Target ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width, data = iris_v_train)
m_lda
m_lda_prob <- predict(object = m_lda, newdata = iris_v_test)$posterior[,2] 
m_lda_pred <- factor(x = (as.numeric(m_lda_prob) < 0.5), 
                        levels = c(TRUE, FALSE), labels = levels(iris_v_train$Target))

# knn-1 (class::knn1())
m_knn1_pred <- knn1(train = iris_v_train[,1:4], test = iris_v_test[,1:4], cl = iris_v_train$Target)

# knn-5 (class::knn())
m_knn5_pred <- knn(train = iris_v_train[,1:4], test = iris_v_test[,1:4], cl = iris_v_train$Target, k = 5)

# compiling predictions and printing confusion matrices
m_pred <- list(m_logreg_pred, m_lda_pred, m_knn1_pred, m_knn5_pred)
lapply(1:4, function(i){
  table('pred' = m_pred[[i]], 'target' = iris_v_test$Target)
})

# we observe that all the models work fine on this dataset example.

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
#         Using unsupervised approach to help with classification of Versicolor VS virginica
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# here, we do not expect a much better performance of the classification task, since it is already pretty
# easily done with logistic regression. Yet, it is interesting to use PCA here because all the covariates
# are pretty much correlated, so it is interesting to create uncorrelated variates.

# PCA
m_pca <- princomp(iris_v_train[,1:4])
summary(m_pca) # 85% variance in the first dimension -> high correlation to exploit

# reconstructing principal component i:
i <- 1
m_pca$scores[,i]
scale(as.matrix(iris_v_train[,1:4]), scale = FALSE, center = TRUE) %*% m_pca$loadings[,i]

# adding Principal components to both datasets
for(i in 1:4){
  temp <- scale(as.matrix(iris_v_train[,1:4]), scale = FALSE, center = TRUE) %*% m_pca$loadings[,i]
  eval(parse(text = paste0("iris_v_train$Princomp", i, " <- temp")))
  temp <- scale(as.matrix(iris_v_test[,1:4]), scale = FALSE, center = TRUE) %*% m_pca$loadings[,i]
  eval(parse(text = paste0("iris_v_test$Princomp", i, " <- temp")))
}

# logistic regression on principal component found
m_logreg_pca <- glm(formula = Target ~ Princomp1 + Princomp2 + Princomp3 + Princomp4, data = iris_v_train, family = binomial(link = 'logit'))
summary(m_logreg_pca)
m_logreg_pca_prob <- predict(object = m_logreg_pca, newdata = iris_v_test, type = 'response')
m_logreg_pca_pred <- factor(x = (as.numeric(m_logreg_pca_prob) < 0.5), 
                        levels = c(TRUE, FALSE), labels = levels(iris_v_train$Target))

# confusion matrix for the logistic regression model
table('pred' = m_logreg_pca_pred, 'target' = iris_v_test$Target)

# we went from 0% error rate to 14% error rate. Even with keeping all the principal components we do quite worse.
# this is a show for the fact that by combining the different variates, we lost information relevant to the link between
# the target and the feature space!
