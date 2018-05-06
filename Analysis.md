---
title: "Machine Learning Coursera Project"
author: 'Yash Saxena'
date: "04-May-2018"
output: word_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

## Background
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity 
relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements 
about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that 
people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project,
our goal is to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants.They were asked to perform barbell-lifts
correctly and incorrectly in 5 different ways. More information is available from the website [here](http://groupware.les.inf.puc-rio.br/har) .

## Data
The training data for this project is available [here](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv) .
The test data is available [here](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv) .
The data for this project come from this source: [http://groupware.les.inf.puc-rio.br/har]. 


## Submission
The goal of our project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set.
I can use any of the other variables to predict with. You should create a report describing how I built this model, how I used 
cross validation, what I think the expected out of sample error is, and why I made the following choices. I will also use my prediction 
model to predict 20 different test cases.
My submission consists of a link to a Github repo with my R markdown and compiled HTML file describing my analysis.

## My Approach:
The outcome variable is classe, a factor variable. For this data set, "participants were asked to perform one set of 10 repetitions 
of the Unilateral Dumbbell Biceps Curl in 5 different fashions: - exactly according to the specification (Class A) - throwing the elbows 
to the front (Class B) - lifting the dumbbell only halfway (Class C) - lowering the dumbbell only halfway (Class D) - throwing the hips 
to the front (Class E)
I have tested two models, using decision tree and random forest. Eventually, I will choose the model with the highest accuracy.

## Cross Validation
Cross-validation will be done by subsampling given training data set randomly without replacement into 2 subsamples: 
TrainTrainingSet data (75% of the original Training data set) and TestTrainingSet data (25%). These models will be fitted on the 
TrainTrainingSet data set, and tested on the TestTrainingSet data. Once the most accurate model is choosen, it will be tested on 
the original Testing data set.

## Out of sample error
The expected out-of-sample error will correspond to the quantity: 1-accuracy in the cross-validation data. Accuracy is the proportion 
of correct classified observation over the total sample in the TestTrainingSet data set. Expected accuracy is the expected accuracy 
in the out-of-sample data set (i.e. original testing data set). Therefore, the expected value of the out-of-sample error will 
correspond to the expected number of missclassified observations/total observations in the Test data set, which is the 
quantity: 1-accuracy found from the cross-validation data set.

## Installed packages

```
install.packages("caret"); 
install.packages("randomForest"); 
install.packages("rpart"); 
install.packages("e1071");
library(lattice); 
library(ggplot2); 
library(caret); 
library(randomForest); 
library(rpart); 
library(rpart.plot);
```

## Code

```
set.seed(1234)

# data load and clean up
trainingset <- read.csv("pml-training.csv", na.strings=c("NA","#DIV/0!", ""))
testingset <- read.csv("pml-testing.csv", na.strings=c("NA","#DIV/0!", ""))

# Exploratory analysis - 
# dim(trainingset); dim(testingset); summary(trainingset); summary(testingset); str(trainingset); str(testingset); head(trainingset); head(testingset);               

# Delete columns with all missing values
trainingset<-trainingset[,colSums(is.na(trainingset)) == 0]
testingset <-testingset[,colSums(is.na(testingset)) == 0]

# Delete variables are irrelevant to our current project: user_name, raw_timestamp_part_1, raw_timestamp_part_,2 cvtd_timestamp, new_window, and  num_window (columns 1 to 7). 
trainingset   <-trainingset[,-c(1:7)]
testingset <-testingset[,-c(1:7)]

# partition the data so that 75% of the training dataset into training and the remaining 25% to testing
traintrainset <- createDataPartition(y=trainingset$classe, p=0.75, list=FALSE)
TrainTrainingSet <- trainingset[traintrainset, ] 
TestTrainingSet <- trainingset[-traintrainset, ]

# The variable "classe" contains 5 levels: A, B, C, D and E. A plot of the outcome variable will allow us to see the frequency of each levels in the TrainTrainingSet data set and # compare one another.

plot(TrainTrainingSet$classe, col="yellow", main="Plot of levels of variable classe within the TrainTrainingSet data set", xlab="classe", ylab="Frequency")
```

Based on the above graph Level A is most frequent while level D is less frequent.

### Model1:- Random Forest
```
model1 <- randomForest(classe ~. , data=TrainTrainingSet, method="class")

# Predicting:
prediction1 <- predict(model1, TestTrainingSet, type = "class")

#Test result
confusionMatrix(prediction2, TestTrainingSet$classe)
```

### Model2:- Decision tree

```
model2 <- rpart(classe ~ ., data=TrainTrainingSet, method="class")

prediction2 <- predict(model2, TestTrainingSet, type = "class")

# Plot the Decision Tree
rpart.plot(model2, main="Classification Tree", extra=102, under=TRUE, faclen=0)
# Test results
confusionMatrix(prediction1, TestTrainingSet$classe)
```
## Result
Random Forest algorithm performed better than Decision Trees. Accuracy for Random Forest model was 0.995 (95% CI: (0.993, 0.997)) 
compared to Decision Tree model with 0.739 (95% CI: (0.727, 0.752)).

We choose the Random Forests model. The expected out-of-sample error is estimated at 0.005, or 0.5%.

## Final Prediction
Here is the final outcome based on the Prediction Model 2 (Random Forest) applied against the Testing dataset

```
# predict outcome levels on the original Testing data set using Random Forest algorithm
predictfinal <- predict(model1, testingset, type="class")
predictfinal
```
