# This R script accompanies the R Markdown file and is used to draft code before publication.
library(tidyverse)
library(ggplot2)
library(corrplot)
library(caret)
library(glmnet)
library(neuralnet)

# make sure to set correct working directory before running code
adm <- read_csv("../data/admissions.csv")
sapply(adm, function(x) sum(is.na(x))) # count # of missing values 
summary(adm) # make sure all values of continuous variables fall within range
names(adm) # get column names to see if renaming needed
adm <- rename(adm, GRE = "GRE Score", TOEFL = "TOEFL Score", U_Ranking = "University Rating", adm_chance = "Chance of Admit") # rename some columns
adm$Research <- as.factor(adm$Research)

plot(adm) #check for linear relationships between predictors and admission chance percentage
sub_adm <- adm[, !names(adm) %in% c("Research")] #remove "Research" column to allow for correlation matrix
corrplot(cor(sub_adm), type = "lower", addCoef.col = "black")

#modeling
set.seed(22) #for reproducible results
adm_mod <- adm[, !names(adm) %in% c("Serial No.")] #removing id column
# training <- trainControl(method="cv", number=10)

# full_lm <- train(adm_chance ~ ., data=adm_mod, method="lm", trControl=training)
# print(full_lm)

# trying to do everything at once
norm_data <- sapply(adm_mod[,-8], function(x) if(is.numeric(x)){
  scale(x) #normalize numeric data first
} else x)
norm_data <- as_tibble(norm_data)
norm_data$adm_chance <- adm_mod$adm_chance #add independent variable column
norm_data$Research <- as.factor(norm_data$Research) #transform Research column back into factor column

#regularized model
X <- model.matrix(adm_chance ~ ., data=norm_data)[, -1]
y <- norm_data$adm_chance
#opt out of default standardization
lm_ridge <- cv.glmnet(x=X,y=y,alpha=0,standardize=FALSE)
opt_lam_r <- lm_ridge$lambda.min

lm_lasso <- cv.glmnet(x=X,y=y,alpha=1,standardize=FALSE)
opt_lam_l <- lm_lasso$lambda.min

# 10 fold cross validation
folds <- createFolds(adm$`Serial No.`, k=10)
res <- matrix(0, 10, 3)
for(i in 1:10) {
  s <- folds[[i]]
  train <- norm_data[-s, ]
  trainY <- train$adm_chance
  test <- norm_data[s, ]
  testY <- test$adm_chance
  
  #full linear
  model1 <- lm(adm_chance ~., data = train)
  pred1 <- predict(model1, test)
  err1 <- sum(abs(pred1 - testY))/length(testY) #M.A.E.
  
  #regularization setup
  trainX <- model.matrix(adm_chance ~ ., data=train)[, -1]
  testX <- model.matrix(adm_chance ~ ., data=test)[, -1]
  
  #ridge
  model2 <- glmnet(x=trainX,y=trainY,alpha=0,lambda=opt_lam_r)
  pred2 <- predict(model2, testX)
  err2 <- sum(abs(pred2 - testY))/length(testY)
  
  #lasso
  model3 <- glmnet(x=trainX,y=trainY,alpha=1,lambda=opt_lam_l)
  pred3 <- predict(model3, testX)
  err3 <- sum(abs(pred3 - testY))/length(testY)
  
  #neural network
  # maxs <- apply(train, 2, max) 
  # mins <- apply(train, 2, min)
  # scaled <- as.data.frame(scale(train, center = mins, 
  #                               scale = maxs - mins))
  
  # mydata <- sapply(train, function(x) if(is.numeric(x)){
  #   scale(x)
  # } else x)
  # 
  # nn <- neuralnet(adm_chance ~ GRE + TOEFL + U_Ranking + SOP + LOR + CGPA + Research, data = mydata, hidden = c(4,3), linear.output = TRUE, stepmax=1e7)
  # 
  # pr.nn <- compute(nn, testX)
  
  res[i,] = c(err1,err2,err3)
}



colMeans(res)
mean(unlist(results))
# i want to create regularized models but i realized that some variables are ordinal and if i'm planning on standardizing predictors i need to figure out if standardizing ordinal predictors is acceptable practice.
# Since 