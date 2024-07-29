# Load necessary libraries
library(readr)
library(dplyr)
library(caret)
library(pROC)
library(rpart)
library(rpart.plot)
library(e1071)

# Load the dataset
data <- read.csv("bank-additional-full.csv", sep = ";")

# Check the column names
colnames(data)

# Convert the target variable to a factor
data$y <- as.factor(ifelse(data$y == "yes", 1, 0))

# Handle categorical variables by converting them to factors
categorical_vars <- c('job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome')
data[categorical_vars] <- lapply(data[categorical_vars], as.factor)

# Split the data into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(data$y, p = .8, list = FALSE, times = 1)
dataTrain <- data[ trainIndex,]
dataTest  <- data[-trainIndex,]

# Logistic Regression model
logistic_model <- glm(y ~ ., data = dataTrain, family = binomial)
summary(logistic_model)

# Predict and evaluate Logistic Regression model
logistic_pred <- predict(logistic_model, newdata = dataTest, type = "response")
logistic_pred_class <- ifelse(logistic_pred > 0.5, 1, 0)

# Confusion matrix for Logistic Regression
logistic_conf_matrix <- confusionMatrix(as.factor(logistic_pred_class), dataTest$y)
logistic_conf_matrix

# AUC-ROC for Logistic Regression
logistic_roc <- roc(dataTest$y, logistic_pred)
plot(logistic_roc, col = "blue")
auc(logistic_roc)

# Decision Tree model
tree_model <- rpart(y ~ ., data = dataTrain, method = "class")
rpart.plot(tree_model)

# Predict and evaluate Decision Tree model
tree_pred <- predict(tree_model, newdata = dataTest, type = "class")
tree_conf_matrix <- confusionMatrix(tree_pred, dataTest$y)
tree_conf_matrix

# AUC-ROC for Decision Tree
tree_pred_prob <- predict(tree_model, newdata = dataTest, type = "prob")[,2]
tree_roc <- roc(dataTest$y, tree_pred_prob)
plot(tree_roc, col = "red")
auc(tree_roc)

# Visualization of Decision Tree structure
rpart.plot(tree_model)

# Display metrics for Logistic Regression
cat("Logistic Regression Metrics:\n")
cat("Accuracy: ", logistic_conf_matrix$overall['Accuracy'], "\n")
cat("Precision: ", logistic_conf_matrix$byClass['Pos Pred Value'], "\n")
cat("Recall: ", logistic_conf_matrix$byClass['Sensitivity'], "\n")
cat("F1 Score: ", logistic_conf_matrix$byClass['F1'], "\n")
cat("AUC: ", auc(logistic_roc), "\n")

# Display metrics for Decision Tree
cat("Decision Tree Metrics:\n")
cat("Accuracy: ", tree_conf_matrix$overall['Accuracy'], "\n")
cat("Precision: ", tree_conf_matrix$byClass['Pos Pred Value'], "\n")
cat("Recall: ", tree_conf_matrix$byClass['Sensitivity'], "\n")
cat("F1 Score: ", tree_conf_matrix$byClass['F1'], "\n")
cat("AUC: ", auc(tree_roc), "\n")

# Interpretation of Results

# Logistic Regression Coefficients
cat("Logistic Regression Coefficients:\n")
print(summary(logistic_model))
cat("Odds Ratios:\n")
print(exp(coef(logistic_model)))

# Decision Tree Structure
cat("Decision Tree Structure:\n")
print(tree_model)
cat("Variable Importance:\n")
print(varImp(tree_model))
