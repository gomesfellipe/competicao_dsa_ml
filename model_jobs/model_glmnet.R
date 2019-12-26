# Carregar pacotes ---------------------------------------------------------------------------

source("init.R")

# Carregar base -------------------------------------------------------------------------------

pp_data <- readRDS("pp_data_knnInput20.rds")

# pp_data <- pp_data %>% select_at(c("target", readRDS("cols_to_select_rfe.rds")))

# GLMNET --------------------------------------------------------------------------------------
control <- trainControl(method="cv", 
                        number=3, 
                        classProbs=TRUE, 
                        summaryFunction=mnLogLoss,
                        # search = "random",
                        verboseIter = TRUE)

# grid <- expand.grid(alpha = c(0,1),
#                     lambda = seq(0.0000001, 0.00001, length = 100))

set.seed(7)
model_glmnet <- train(target~.,
                      data=pp_data,
                      method="glmnet",
                      metric="logLoss",
                      # tuneGrid=grid,
                      tuneLength = 200,
                      trControl=control)
# display results
saveRDS(model_glmnet, "model_glmnet.rds")

# model_glmnet <- readRDS("model_glmnet.rds")

# model_glmnet %>% summary()


# plot(model_glmnet)

