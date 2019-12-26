# Carregar pacotes ---------------------------------------------------------------------------

source("init.R")

# Carregar base -------------------------------------------------------------------------------

pp_data <- readRDS("pp_data_knnInput30.rds")

# nnet -----------------------------------------------------------------------------------------
control <- trainControl(method="cv", 
                        number=3, 
                        classProbs=TRUE, 
                        summaryFunction=mnLogLoss,
                        search = "random",
                        verboseIter = TRUE)

set.seed(7)
model_nnet <- train(target~.,
                   data=pp_data,
                   method="nnet",
                   metric="logLoss",
                   tuneLength = 20,
                   trControl=control)
# display results
saveRDS(model_nnet, "model_nnet.rds")
model_nnet %>% summary()
