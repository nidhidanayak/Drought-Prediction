library(C50)
library(MASS)
library(psych)
library(tidyverse)
library(lubridate)
library(ggplot2)
library(corrplot)
library(mlbench)
library(caret)
library(pROC)
library(scales)
library(rpart)
library(e1071)
library(DMwR)
library(nnet)

set.seed(100)

soil_train <- read.csv('C:/Users/Nidhi/OneDrive/Desktop/soil/train_pp.csv')

soil_test <- read.csv('C:/Users/Nidhi/OneDrive/Desktop/soil/test_pp.csv')

soil_validation <- read.csv('C:/Users/Nidhi/OneDrive/Desktop/soil/validation_pp.csv')

soil_original <- soil_train

str(soil_train) #dataset analysis

summary(soil_train) #no NAs

unique(soil_train$date)
unique(soil_test$date)
unique(soil_validation$date)

#DATASET'S TOO LARGE TO PROCESS ANYTHING SO WE REDUCE THE OBSERVATIONS AND TAKE 5 YEARS OF DATA (2011-2016)

soil_subset <- subset(x = soil_train[soil_train$date >= "2011-01-04",])
soil_subset <- soil_train

dim(soil_subset)

str(soil_subset)

#Converting the columns to numeric for data analysis

soil_subset$fips = as.numeric(soil_subset$fips)

soil_subset$date = as.Date(soil_subset$date)

soil_subset$score <- as.numeric(soil_subset$score)

str(soil_subset)

#EXPLORATORY DATA ANALYSIS

soil_cor <- cor(soil_subset[, -2])

cor_with_score <- data.frame(soil_cor[20,])

cor_with_score

#The correlation matrix shows that, only 
#PS, T2M_MAX, T2M_RANGE, TS, WS10M_RANGE ARE CORRELATED TO OUR DEPENDENT VARIABLE~ SCORE.

corrplot(soil_cor, type = "lower")


sort(findCorrelation(soil_cor, cutoff = 0.75, names=T)) # High Correlation among 
#"QV2M"        "T2M"         "T2M_MAX"     "T2M_MIN"     "T2MDEW"      "T2MWET"     
#"WS10M"       "WS10M_MAX"   "WS10M_MIN"   "WS10M_RANGE" "WS50M"       "WS50M_MAX"  

soil_nodate <- soil_subset[, -2]
str(soil_nodate)

# plot histogram of each feature
par(mfrow=c(5,4), oma = c(0,0,2,0) + 0.1,  mar = c(3,3,1,1) + 0.1)
for (i in names(soil_nodate)) {
  hist(soil_nodate[[i]], col="wheat2", ylab = "", xlab = "", main = "")
  mtext(names(soil_nodate[i]), cex=0.8, side=1, line=2)
}
mtext(paste("Histograms of Features (", length(names(soil_nodate)), ")", sep = ""), outer=TRUE,  cex=1.2)

#THE HISTOGRAM PLOT SHOWS THAT THE FEATURES ARE NOT NORMALLY DISTRIBUTED

pp_df <- preProcess(soil_nodate[, -c(20)], method = c("BoxCox", "center", "scale")) # Transform values

pp_soil <- data.frame(predict(pp_df, soil_nodate))


# Remove outliers
describe(pp_soil)
for (i in names(pp_soil[-c(20)])) {
  pp_soil <- pp_soil[!abs(pp_soil[[i]]) > 3 ,]
}

pp_soil$score = round(pp_soil$score) 


# plot histogram of preprocessed feature
par(mfrow=c(5,4), oma = c(0,0,2,0) + 0.1,  mar = c(3,3,1,1) + 0.1)
for (i in names(pp_soil)) {
  hist(pp_soil[[i]], col="wheat2", ylab = "", xlab = "", main = "")
  mtext(names(pp_soil[i]), cex=0.8, side=1, line=2)
  
}
mtext(paste("Histograms of Scaled Features (", length(names(pp_soil)), ")", sep = ""), outer=TRUE,  cex=1.2)

#The data has been scaled and normalized

summary(pp_soil)

# plot boxplots of each feature for output values
par(mfrow=c(5,4), oma = c(0,0,2,0) + 0.1,  mar = c(3,3,1,1) + 0.1)
for (i in names(pp_soil)) {
  boxplot(pp_soil[[i]] ~ pp_soil$score, col="wheat2", ylab = "", xlab = "", main = "")
  mtext(names(pp_soil[i]), cex=0.8, side=1, line=2)
}
mtext(paste("BoxPlots of Scaled Features (", length(names(pp_soil)), ")", sep = ""), outer=TRUE,  cex=1.2)



#MODEL BUILDING

#DOWNSAMPLING TO IMPROVE PREDICTION

table(pp_soil$score)

pp_soil$score <- as.factor(pp_soil$score) #Down-sampling requires a factor variable as the response

soil_downsampling <- downSample(x = pp_soil[,-20], y = pp_soil$score, list = FALSE, yname = 'score')

str(soil_downsampling)

table(soil_downsampling$score)

#According to the correlation matrix, only 
#PS, T2M_MAX, T2M_RANGE, TS, WS10M_RANGE 
#ARE CORRELATED TO OUR DEPENDENT VARIABLE SCORE.

#BUILDING A MODEL USING ONLY THESE COLUMNS

feats <- names(soil_downsampling[c(1,  3, 8, 10, 15)])

feats

# Concatenate strings
f <- paste(feats,collapse=' + ')
f <- paste('score ~',f)

# Convert to formula
f <- as.formula(f)
f

#BUILDING AN ORDINAL MODEL

ordinal_model <- polr(f, data= soil_downsampling, Hess = TRUE)

summary(ordinal_model)

summary_table <- coef(summary(ordinal_model))

logLik(ordinal_model)

pval <- pnorm(abs(summary_table[,'t value']), lower.tail = FALSE)*2

summary_table <- cbind(summary_table, "p value" = round(pval,3))

summary_table #p value for all is ~0

#PREDICTING ON THE TRAIN, TEST and VALIDATION SET

prediction_validation_polr <- predict(ordinal_model, newdata = soil_validation)

mean(prediction_validation_polr == soil_validation$score)

prediction_test_polr <- predict(ordinal_model, newdata = soil_test)

mean(prediction_test_polr == soil_test$score)

prediction_train_polr <- predict(ordinal_model, newdata = soil_downsampling)

mean(prediction_train_polr == soil_downsampling$score)

#CHECKING OTHER METRICS ON THE TEST SET

prediction_test_polr <- as.factor(prediction_test_polr)
y_act <- as.factor(soil_test$score)

confusionMatrix(data = prediction_test_polr, reference = y_act)

#C5.0 

model_C50 <- C50::C5.0(f, soil_downsampling, method = "class")

summary(model_C50)

plot(model_C50)

#PREDICTING ON THE TRAIN, TEST and VALIDATION SET

prediction_validation_c5.0 <- predict(model_C50, newdata = soil_validation)

mean(prediction_validation_c5.0 == soil_validation$score)

prediction_test_c5.0 <- predict(model_C50, newdata = soil_test)

mean(prediction_test_c5.0 == soil_test$score)

prediction_train_c5.0 <- predict(model_C50, newdata = soil_downsampling)

mean(prediction_train_c5.0 == soil_downsampling$score)

confusionMatrix(data = prediction_test_c5.0, reference = y_act)

#CART

model_RPART <- rpart(f, soil_downsampling, method = "class")

summary(model_RPART)
par(mfrow=c(1,1))
plot(model_RPART)
text(model_RPART)

#PREDICTING ON THE TRAIN, TEST and VALIDATION SET

prediction_validation_cart <- predict(model_RPART, newdata = soil_validation, "class")

mean(prediction_validation_cart == soil_validation$score)

prediction_test_cart <- predict(model_RPART, newdata = soil_test, "class")

mean(prediction_test_cart == y_act)

prediction_train_cart <- predict(model_RPART, newdata = soil_downsampling, "class")

mean(prediction_train_cart == soil_downsampling$score)

prediction_test_cart <- as.factor(prediction_test_cart)

confusionMatrix(data= prediction_test_cart, reference = y_act)

#NEURAL NETWORK

model_nn <- nnet::nnet(f,soil_downsampling, size=80, linout = FALSE, maxit = 100)

#PREDICTING ON THE TRAIN, TEST and VALIDATION SET

prediction_validation_nn <- predict(model_nn, newdata = soil_validation, "class")

mean(prediction_validation_nn == soil_validation$score)

prediction_test_nn <- predict(model_nn, newdata = soil_test, "class")

mean(prediction_test_nn == soil_test$score)

prediction_train_nn <- predict(model_nn, newdata = soil_downsampling, "class")

mean(prediction_train_nn == soil_downsampling$score)

prediction_test_nn <- as.factor(prediction_test_nn)

confusionMatrix(data= prediction_test_nn, reference = y_act)

table(y_act)

table(prediction_test_nn)

#COMPARING THE MODELS BASED ON ROC CURVE AND AUC FOR TEST DATASET

#ROC takes only numeric values as input

y_label_numeric <- as.numeric(soil_test$score)  

predict_c50_numeric <- as.numeric(prediction_test_c5.0)

predict_polr_numeric <- as.numeric(prediction_test_polr)

predict_cart_numeric <- as.numeric(prediction_test_cart)

predict_nn_numeric <- as.numeric(prediction_test_nn)

roc_c5.0 <- multiclass.roc(predict_c50_numeric, y_label_numeric, plot= TRUE,  percent=TRUE, main= "ROC for C5.0")

roc_polr <- roc(predict_polr_numeric, y_label_numeric, plot= TRUE, auc = TRUE,  percent=TRUE, main= "ROC for POLR")

roc_cart <- roc(predict_cart_numeric, y_label_numeric, plot= TRUE, auc = TRUE,  percent=TRUE, main= "ROC for CART")

roc_nn <- roc(predict_nn_numeric, y_label_numeric, plot= TRUE, auc = TRUE,  percent=TRUE, main= "ROC for NEURAL NETWORK")

roc.test(roc_c5.0, roc_cart, plot= TRUE)

roc.test(roc_nn, roc_polr, plot= TRUE)