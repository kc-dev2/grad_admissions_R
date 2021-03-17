# This R script accompanies the R Markdown file and is used to draft code before publication.
library(tidyverse)
library(ggplot2)
library(corrplot)
library(caret)

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

# i want to create regularized models but i realized that some variables are ordinal and if i'm planning on standardizing predictors i need to figure out if standardizing ordinal predictors is acceptable practice.