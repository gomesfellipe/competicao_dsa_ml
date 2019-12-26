# Carregar pacotes ---------------------------------------------------------------------------

source("init.R")

# Carregar base -------------------------------------------------------------------------------

pp_data <- readRDS("pp_data_knnInput30.rds")

# pcaNNet -----------------------------------------------------------------------------------------
control <- trainControl(method="cv", 
                        number=3, 
                        classProbs=TRUE, 
                        summaryFunction=mnLogLoss,
                        search = "random",
                        verboseIter = TRUE)

set.seed(7)
model_pcaNNet <- train(target~.,
                   data=pp_data,
                   method="pcaNNet",
                   metric="logLoss",
                   tuneLength = 20,
                   trControl=control)
# display results
saveRDS(model_pcaNNet, "model_pcaNNet.rds")
model_pcaNNet %>% summary()
