# Carregar pacotes ---------------------------------------------------------------------------

source("init.R")

# Carregar base -------------------------------------------------------------------------------

pp_data <- readRDS("pp_data_knnInput30.rds")

# xgbDART -------------------------------------------------------------------------------------
control <- trainControl(method="cv", 
                        number=3, 
                        classProbs=TRUE, 
                        summaryFunction=mnLogLoss,
                        search = "random",
                        verboseIter = TRUE)

set.seed(7)
model_xgbDART <- train(target~.,
                       data=pp_data,
                       method="xgbDART",
                       metric="logLoss",
                       tuneLength = 10,
                       trControl=control)

saveRDS(model_xgbDART, "model_xgbDART.rds")
# display results
model_xgbDART %>% summary()
