 args <- commandArgs(TRUE)
dataURL<-as.character(args[1])
header<-as.logical(args[2])
d<-read.csv(dataURL,header = header)
int<-as.integer(args[3])



library(rpart)
#library(e1071) 
library(class)
library("neuralnet")
library(mlbench)
library(ada)
library(randomForest)


# create 10 samples
set.seed(123)
for(i in 1:10) {
  cat("Running sample ",i,"\n")
  sampleInstances<-sample(1:nrow(d),size = 0.9*nrow(d))
  trainingData<-d[sampleInstances,]
  trainingData <-na.omit(trainingData)
  testData<-d[-sampleInstances,]
  testData <-na.omit(testData)
  
  # which one is the class attribute
  Class<-as.factor(as.integer(args[3]))
  # now create all the classifiers and output accuracy values:
  class1 <- as.integer(args[3])
  testClass <- testData[,as.integer(int)]
  
  # now create all the classifiers and output accuracy values:
  #Decision TREE
  method <-"DecisionTree"
  modeldecision <- rpart(as.formula(paste0("as.factor(", colnames(d)[int], ") ~ .")),data=trainingData,parms = list(split = "information"), method = "class", minsplit = 1)
  #printcp(modeldecison)
  prunedTree <- prune(modeldecision, cp = 0.010000)
  predictTree <- predict(prunedTree,testData,type="class")
  treeacctable <- table(predictTree,testClass)
  treeaccuValue <- sum(diag(treeacctable))/sum(treeacctable) *100
  cat("Method = ", method,", accuracy= ", treeaccuValue,"\n")
  
  #SVM
  method <- 'SVM'
  #modelsvm <- svm(classform, data = trainingData)
  modelsvm <- svm(as.formula(paste0("as.factor(", colnames(d)[int], ") ~ .")),data = trainingData)
  predsvm <- predict(modelsvm, testData, type = "class")
  #Acuuracy
  svmtable <- table(predsvm,testClass)
  svmaccuracy <- sum(diag(svmtable))/sum(svmtable) *100
  cat("Method = ", method,", accuracy= ", svmaccuracy,"\n")
  
  #NB
  method <- 'NaiveBayes'
  nbmodel <- naiveBayes(as.formula(paste0("as.factor(", colnames(d)[int], ") ~ .")), data = trainingData)
  prednb <- predict(nbmodel, testData, type = "class")
  nbacctable <- table(prednb,testClass)
  nbaccvalue <- sum(diag(nbacctable))/sum(nbacctable) *100
  cat("Method = ", method,", accuracy= ", nbaccvalue,"\n")
  
  
  #knn
  method <- 'KNN'
  trainClass<-trainingData[,as.integer(args[3])]

  knnModel <- knn(trainingData,testData,trainClass, k= 3, prob = TRUE)
  knnacctable<- table(knnModel,testClass)
  knnaccvalue <- sum(diag(knnacctable))/sum(knnacctable) *100
  cat("Method = ", method,", accuracy= ", knnaccvalue,"\n")
  
  
  #Logistic Regression
  method <- 'Logistic Regression'
  logisticModel <- glm(as.formula(paste0("as.factor(", colnames(d)[int], ") ~ .")), data = trainingData, family = "binomial")
  predlogistic<-predict(logisticModel, newdata=testData, type="response")
  threshold=0.65
  prediction<-sapply(predlogistic, FUN=function(x) if (x>threshold) 1 else 0)
  actual<-d[,as.integer(int)]
  LRaccvalue <- sum(actual==prediction)/length(actual) *100
  cat("Method = ", method,", accuracy= ", LRaccvalue,"\n")
  
  #NeuralNetworks
  
  #nnModel <- neuralnet(as.matrix(class1) ~  as.matrix(d[,as.integer(3)]) + as.matrix(d[,as.integer(2)]), trainingData, hidden = 4, lifesign = "minimal",linear.output = FALSE, threshold = 0.1)
  # test
  #temp_test <- subset(testData, select = c())
  #creditnet.results <- compute(creditnet, temp_test)
  
  #results <- data.frame(actual = testset$default10yr, prediction = creditnet.results$net.result)
  #results[100:115, ]
  # use rounding to approximat
  #results$prediction <- round(results$prediction)
  
  #results[100:115, ]
  
  
  #boosting
  method <- 'boosting'
  model <- ada(as.formula(paste0("as.factor(", colnames(d)[int], ") ~ .")), data = trainingData, iter=20, nu=1, type="discrete")
  p=predict(model,testData)
  # accuracy
  error <- sum(testData$default10yr==p)/length(p)
  
   cat("Method = ", method,", error= ", error,"\n")
   
   #random forest
   method <- "random"
   rfModel <- randomForest(as.formula(paste0("as.factor(", colnames(d)[int], ") ~ .")), data=trainingData, importance=TRUE, proximity=TRUE, ntree=500)
   RFpred <- predict(rfModel,testData,type='response')
   predTable <- table(RFpred,testClass)
   error <-accuracy <- sum(diag(predTable))/sum(predTable)*100
   
   cat("Method = ", method,", error= ", error,"\n")
}
