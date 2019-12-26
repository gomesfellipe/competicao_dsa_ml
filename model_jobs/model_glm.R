
# Carregar pacotes ---------------------------------------------------------------------------

source("init.R")

# Carregar base -------------------------------------------------------------------------------

pp_data <- readRDS("pp_data_knnInput20.rds") %>% 
  select("target", "v66", "v110", "v6", "v10", "v14", "v21", "v28", "v39", "v40", "v50", "v69",
         "v102", "v114", "v115", "v199", "v120")

# GLM -----------------------------------------------------------------------------------------
control <- trainControl(method="cv", 
                        number=3, 
                        classProbs=TRUE, 
                        summaryFunction=mnLogLoss,
                        verboseIter = TRUE)
set.seed(7)
model_glm <- caret::train(target~.,
                   data=pp_data,
                   method="glm",
                   metric="logLoss",
                   trControl=control)

saveRDS(model_glm, "model_glm.rds")

# display results
model_glm %>% summary()



# Subsets -------------------------------------------------------------------------------------

library(leaps)

models <- regsubsets(target~., data = pp_data, nvmax = 20, really.big=T)
summary(models)
