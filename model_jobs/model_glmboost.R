# Carregar pacotes ---------------------------------------------------------------------------

source("init.R")

# Carregar base -------------------------------------------------------------------------------

pp_data <- readRDS("pp_data_knnInput30.rds")

pp_data <- pp_data %>% select_at(c("target", readRDS("cols_to_select_rfe.rds")))

# GLMBOOST ------------------------------------------------------------------------------------
control <- trainControl(method="cv", 
                        number=3, 
                        classProbs=TRUE, 
                        summaryFunction=mnLogLoss,
                        # search = "random",
                        verboseIter = TRUE)

grid = expand.grid(prune = c('no'),
                   mstop = c(seq(100, 1000, 100)))

set.seed(7)
model_glmboost <- train(target~.,
                        data=pp_data,
                        method="glmboost",
                        metric="logLoss",
                        # tuneLength = 5,
                        tuneGrid = grid,
                        trControl=control)
# display results
saveRDS(model_glmboost, "model_glmboost_tun.rds")

# model_glmboost_tun <- readRDS("model_glmboost_tun.rds")





