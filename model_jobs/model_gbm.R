# Carregar pacotes ---------------------------------------------------------------------------

source("init.R")

# Carregar base -------------------------------------------------------------------------------

pp_data <- readRDS("pp_data_knnInput30.rds")

pp_data <- pp_data %>% select_at(c("target", readRDS("cols_to_select_rfe.rds")))

# GBM -----------------------------------------------------------------------------------------
control <- trainControl(method="cv", 
                        number=3, 
                        classProbs=TRUE, 
                        summaryFunction=mnLogLoss,
                        verboseIter = TRUE)

grid <-  expand.grid(interaction.depth = c(1, 5, 10), 
                        n.trees = (1:30)*50, 
                        shrinkage = c(0.01, 0.1),
                        n.minobsinnode = c(10, 20))

set.seed(7)
model_gbm <- train(target~.,
                   data=pp_data,
                   method="gbm",
                   metric="logLoss",
                   # tuneLength = 10,
                   tuneGrid = grid,
                   trControl=control,
                   verbose = FALSE)
# display results
saveRDS(model_gbm, "model_gbm_tun.rds")
# model_gbm %>% summary()

# model_gbm_tun <- readRDS("model_gbm.rds")
