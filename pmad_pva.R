require(readxl)
cup98LRN <- as.data.frame(read_excel("C:/Users/Brandon/Desktop/pmad-pva/pmad_pva.xlsx"))
cup98LRN <- cup98LRN[!is.na(cup98LRN$TargetD),]

y <- cup98LRN$TargetD
X <- cup98LRN[, -c(1:3)]
require(caret)

preProcess.X <- preProcess(X, method = c("zv", "nzv", "corr", "BoxCox", "center", "scale", "medianImpute", "spatialSign"), outcome = y)
X <- predict(preProcess.X, X)

X$StatusCat96NK <- as.numeric(X$StatusCat96NK == "S")

BIN.DemCluster <- c("04", "08", "09", "16", "19", "20", "23", "25", "26", "27","28", 
                    "30", "32", "33", "36", "38", "39", "40", "41", "43", "45", "46", 
                    "47", "48", "49", "51", "52", "53")

X$DemCluster <- as.numeric(X$DemCluster %in% BIN.DemCluster)
X$DemGender <- as.numeric(X$DemGender == "F")
X$DemHomeOwner <- as.numeric(X$DemHomeOwner == "H")


list.model <- list()
list.model[["lm"]] <- train(y=y, x=X, 
                            trControl = trainControl(method = "repeatedcv", number = 5, repeats = 5), 
                            method = "lm")


for (j in c("forward", "both", "backward"))
{
    list.model[[j]] <- train(y=y, x=X, 
                    trControl = trainControl(method = "repeatedcv", number = 5, repeats = 5), 
                    method = "lmStepAIC", trace = 0, direction = j)
}

sapply(X = list.model, FUN = function(x) coef(x$finalModel))
sapply(X = list.model, FUN = getTrainPerf)

reduced.model <- list.model[["backward"]]$finalModel
full.model <- list.model[["lm"]]$finalModel

anova(reduced.model, full.model)
summary(reduced.model)
anova(reduced.model)

par(mfrow = c(2, 2))
plot(reduced.model, pch = 19, col = "#00000022")
?