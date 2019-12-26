# Carregar pacotes ---------------------------------------------------------------------------

source("init.R")

# Carregar base -------------------------------------------------------------------------------

pp_data <- readRDS("pp_data_knnInput20.rds") %>% 
  select("target", "v66", "v110", "v6", "v10", "v14", "v21", "v28", "v39", "v40", "v50", "v69",
         "v102", "v114", "v115", "v120")

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
model_svm <- train(target~.,
                      data=pp_data,
                      metric="logLoss",
                      # tuneGrid=grid,
                      method = "svmRadial",               
                      preProcess = c("center", "scale"),
                      tuneLength = 10,
                      trControl=control)
# display results
saveRDS(model_svm, "model_svm.rds")

# model_svm <- readRDS("model_svm.rds")

# model_svm %>% summary()


# plot(model_svm)

