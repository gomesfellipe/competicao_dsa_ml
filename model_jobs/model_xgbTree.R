# Carregar pacotes ---------------------------------------------------------------------------

source("init.R")

# Carregar base -------------------------------------------------------------------------------

pp_data <- readRDS("pp_data_knnInput30.rds")

# xgbTree -------------------------------------------------------------------------------------
control <- trainControl(method="cv", 
                        number=10, 
                        classProbs=TRUE, 
                        summaryFunction=mnLogLoss,
                        # search = "random",
                        verboseIter = TRUE)

grid <- expand.grid(nrounds = c(200, 800, 1000),
                    max_depth = 3, 
                    eta = c(0.001,0.01, 0.2),
                    gamma = c(0, 1, 10, 100, 1000),
                    colsample_bytree = 0.5,
                    min_child_weight = c(1,3),
                    subsample = 0.5 
                    # lambda = c(0, 1e-2, 0.1, 1, 100, 1000, 10000),
                    # alpha = c(0, 1e-2, 0.1, 1, 100, 1000, 10000)
)

set.seed(7)
model_xgbTree <- train(target~.,
                       data=pp_data,
                       method="xgbTree",
                       metric="logLoss",
                       # tuneLength = 10,
                       tuneGrid = grid,
                       trControl=control)

saveRDS(model_xgbTree, "model_xgbTree_tun.rds")
# display results
model_xgbTree %>% summary()
