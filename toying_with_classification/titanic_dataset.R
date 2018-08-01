# install.packages('titanic')
require(titanic)
require(randomForest)
require(arules)
df <- titanic_train

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
#                               Handcrafting features
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# creating categorical variables
df$Fareclass <- discretize(x = df$Fare, method = 'frequency', categories = 5)
df$Ageclass <-  discretize(x = df$Age, method = 'frequency', categories = 5)
df$Sex <- factor(df$Sex)
df$Embarked <- factor(df$Embarked)

# quick view at survival rates
temp <- with(df, aggregate(x = Survived, by = list(Sex, Pclass), FUN = function(x) c(length(x), mean(x))))
temp <- cbind(temp[,-ncol(temp)], temp$x)
colnames(temp) <- c('Sex', 'Pclass', 'Fareclass', 'N', 'SurvivalProb')

# train-test split
df <- df[sample(1:nrow(df)),]
index_train <- 1:floor(nrow(df)/2) 
df_train <- df[index_train,]
df_test <- df[-index_train,]

# logistic regression on titanic data
m_logreg_0 <- glm(formula = Survived ~ Sex + Pclass + Fare + Embarked + Parch + SibSp + Age, 
                  data = df_train, family = binomial('logit'))
summary(m_logreg_0)
m_logreg_1 <- glm(formula = Survived ~ Sex * Pclass * Fareclass + Embarked + Parch + SibSp + Ageclass, 
                  data = df_train, family = binomial('logit'))
summary(m_logreg_1)
m_logreg_2 <- glm(formula = Survived ~ Sex * Pclass * Fareclass, 
                  data = df_train, family = binomial('logit'))
summary(m_logreg_2)

# LR predictions
m_logreg <- list(m_logreg_0, m_logreg_1, m_logreg_2)
predSurvived <- sapply(1:length(m_logreg), function(i){
  p <- predict(object = m_logreg[[i]], newdata = df_test, type = 'response')
  as.numeric(p)
})
predSurvived <- data.frame(predSurvived)
colnames(predSurvived) <- paste0('LR', 0:2)

# big difficulty -> which combinations are impactful ? Decision Trees

# Rpart
rp0 <- rpart(formula = Survived ~ Sex + Pclass + Fare + Embarked + Parch + SibSp + Age, data = df_train)
predSurvived$rp0 <- as.numeric(predict(object = rp0, newdata = df_test, type = 'matrix'))

# random forest
set.seed(643)
rf0 <- randomForest(formula = factor(Survived) ~ Sex + Pclass + Fare + Embarked + Parch + SibSp, data = df_train, na.action = na.omit)
predSurvived$rf0 <- as.numeric(as.character(predict(object = rf0, newdata = df_test, type = 'response')))
varImpPlot(rf0)

# do lda
m_lda <- lda(formula = factor(Survived) ~ Sex + Pclass + Fare + Embarked + Parch + SibSp, data = df_train)
predSurvived$LDA <- predict(object = m_lda, newdata = df_test)$posterior[,2]

# visual comparison of model outputs
matplot(t(predSurvived), type ='l', col = 'black', lty = 1)

#
predSurvived$Cmin <- apply(predSurvived[,1:6], 1, function(x) min(x, na.rm = TRUE))
predSurvived$Cmax <- apply(predSurvived[,1:6], 1, function(x) max(x, na.rm = TRUE))

# comaring performances 
res <- lapply(1:ncol(predSurvived), function(i){
  
  r <- list('model_name' = colnames(predSurvived)[i])
  
  pred_is_not_na <- which(!is.na(predSurvived[,i]))
  if(length(pred_is_not_na) == 0){
    r$error_msg <- 'Model sent back NA for each observation of the dataset sent.'
    r$nb_NAs <- nrow(df_test)
    return(r)
  }
  
  r$error_msg <- ''
  r$nb_NAs <- nrow(df_test) - length(pred_is_not_na)
  r$error_rate <- sum(1*(predSurvived[pred_is_not_na,i] > 0.5) != df_test$Survived[pred_is_not_na]) / length(pred_is_not_na)
  r$confusion_matrix <- table('pred' = 1*(predSurvived[pred_is_not_na,i] > 0.5), 'target' = df_test$Survived[pred_is_not_na])
  return(r)
  
})


