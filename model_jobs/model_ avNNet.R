# Carregar pacotes ---------------------------------------------------------------------------

source("init.R")

# Carregar base -------------------------------------------------------------------------------

pp_data <- readRDS("pp_data_knnInput30.rds")

# avNNet -----------------------------------------------------------------------------------------
control <- trainControl(method="cv", 
                        number=3, 
                        classProbs=TRUE, 
                        summaryFunction=mnLogLoss,
                        search = "random",
                        verboseIter = TRUE)

set.seed(7)
model_avNNet <- train(target~.,
                   data=pp_data,
                   method="avNNet",
                   metric="logLoss",
                   tuneLength = 10,
                   trControl=control)
# display results
saveRDS(model_avNNet, "model_avNNet.rds")
model_avNNet %>% summary()
