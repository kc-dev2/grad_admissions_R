# This R script accompanies the R Markdown file and is used to draft code before publication.
library(tidyverse)
library(ggplot2)
library(corrplot)
library(caret)
library(glmnet)

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
set.seed(22)
adm_mod <- adm[, !names(adm) %in% c("Serial No.")]
training <- trainControl(method="cv", number=10)

full_lm <- train(adm_chance ~ ., data=adm_mod, method="lm", trControl=training)
print(full_lm)

X <- model.matrix(adm_chance ~ ., data=adm_mod)[, -1]
y <- adm_mod$adm_chance
lm_ridge <- cv.glmnet(x=X,y=y,alpha=0)
opt_lam <- lm_ridge$lambda.min

folds <- createFolds(adm$`Serial No.`, k=10)
res = rep(0,10)
results <- lapply(folds, function(x) {
  train <- adm_mod[-x, ]
  test <- adm_mod[x, ]
  actual <- test$adm_chance
  
  model1 <- lm(adm_chance ~., data = train)
  pred1 <- predict(model1, test)
  err1 <- sum(abs(pred1 - actual))
  
  X <- model.matrix(adm_chance ~ ., data=train)[, -1]
  y <- train$adm_chance
  model2 <- glmnet(x=X,y=y,alpha=0,lambda=opt_lam)
  pred2 <- predict(model2, test)
  err2 <- sum(abs(pred2 - actual))
  
  return(err2)
})
results
mean(unlist(results))
# i want to create regularized models but i realized that some variables are ordinal and if i'm planning on standardizing predictors i need to figure out if standardizing ordinal predictors is acceptable practice.
# Since 