library(mlbench)
library(caret)
library(earth)
library(rpart)
library(e1071)
library(randomForest)
library(boot)

setwd('C:\\XXX\\')

FeatureSet   = read.csv('train_features.csv')
SalarySet    = read.csv('train_salaries.csv')
TestSet      = read.csv('test_features.csv')

########### merging feature set and salary set on jobIDs ###########
TrainSet     = merge(FeatureSet, SalarySet, by = 'jobId', all.x = T, all.y = T)
TrainSet     = TrainSet[,-c(1,2)]
TrainSet     = TrainSet[TrainSet$salary > 0,]

########## Exploratory Analysis ############
###### Do bi-variate analysis ########
###### check for multi-collinearity ###
###### check for correlation #########
attach(TrainSet)
boxplot(salary~jobType)
boxplot(salary~companyId)
boxplot(salary~degree)
boxplot(salary~major)
boxplot(salary~industry)

cor.test(TrainSet$yearsExperience    , TrainSet$salary)
cor.test(TrainSet$milesFromMetropolis, TrainSet$salary)


##### Modelling ########
##### cross-validation to check for over-fitting ######
PredErros = NULL
for(k in c(1:10)){
  inTraining   = createDataPartition(TrainSet$salary, p = 0.75, list = FALSE)
  training     = TrainSet[inTraining, ]
  testing      = TrainSet[-inTraining, ]
  
  #modelFit     = train( salary~.,data=training, method = 'rpart')
  #modelFit     = lm( salary~.,data=training)
  #modelFit     = earth( salary~.,data=training)
  modelFit      = glm( salary~.,data=training) #best so far
  
  #modelFit     = rpart( salary~.,data=training)
  #modelFit     = svm( salary~.,data=training) #too slow
  #modelFit     = naiveBayes( salary~.,data=training) can't be used because it's used for classifications
  #modelFit     = randomForest( salary~., data= training, ntree=10)
  testing$pred = predict(modelFit, testing )
  testing      = testing[testing$salary >0,]
  
  error        = abs(testing$salary - testing$pred)
  MAPE         = error/testing$salary*100
  
  errorTemp    = NULL
  errorTemp$MAE  = mean(error)
  errorTemp$MAPE = mean(MAPE)
  
  rmse <- function(error){
    sqrt(mean(error^2))
  }
  
  RMSE_rpart = rmse(error)
  errorTemp$RMSE = RMSE_rpart
  PredErros = rbind(PredErros, errorTemp)
}

########### Computing Variable Importance ###########
varImp(modelFit)

########### Training on entire data set ##############
modelFit      = glm( salary~.,data=TrainSet)

TestSetSS = TestSet[,-c(1,2)]
TestSetSS$salary = predict(modelFit,TestSetSS)

TestSet$salary = TestSetSS$salary

write.csv(TestSet[,c('jobId', 'salary')], 'test_salaries.csv')

