# Carregar pacotes ---------------------------------------------------------------------------

source("init.R")

# Carregar base -------------------------------------------------------------------------------

pp_data <- readRDS("pp_data_knnInput20.rds") 

# pp_data <- readRDS("pp_data.rds")

# RF -----------------------------------------------------------------------------------------
control <- trainControl(method="cv",
                        number=3,
                        classProbs=TRUE,
                        summaryFunction=mnLogLoss,
                        verboseIter = TRUE)

grid <- expand.grid(
  mtry = floor(ncol(pp_data) * c(.05, .15, .25, .333, .4)),
  min.node.size = c(1, 3, 5),
  splitrule = c("extratrees")
)

set.seed(7)
model_rf <- train(target~.,
                  data=pp_data,
                  method="ranger",
                  metric="logLoss",
                  tuneGrid = grid,
                  num.trees = ncol(pp_data)*10,
                  trControl=control,
                  importance = 'impurity')

saveRDS(model_rf, "model_rf_tun2.rds")

model_rf_tun <- readRDS("model_rf_tun2.rds")
# ggplot(model_rf_tun)

# grid <- expand.grid(
#   mtry = c(0,2,8,16,32,64),
#   splitrule = c("extratrees"),
#   min.node.size = c(12,8,16,32,64)
# )
# 
# set.seed(7)
# model_rf <- train(target~.,
#                   data=pp_data,
#                   method="ranger",
#                   metric="logLoss",
#                   tuneGrid = grid,
#                   # tuneLength = 10,
#                   num.trees = 200,
#                   trControl=control)
# display results

# saveRDS(model_rf, "rf_tune1.rds")

# rf_tune1 <- readRDS("rf_tune1.rds")
# 
# grid2 <- expand.grid(
#   mtry = rf_tune1$bestTune$mtry,
#   splitrule = c("extratrees"),
#   min.node.size = seq(2, 32 , 2)
# )
# 
# set.seed(7)
# model_rf2 <- train(target~.,
#                   data=pp_data,
#                   method="ranger",
#                   metric="logLoss",
#                   tuneGrid = grid2,
#                   # tuneLength = 10,
#                   num.trees = 200,
#                   trControl=control)
# # display results
# 
# saveRDS(model_rf2, "rf_tune2.rds")


# 2tun ----------------------------------------------------------------------------------------

# rf_tune2 <- readRDS("rf_tune2.rds")

# grid3 <- expand.grid(
#   mtry = c((rf_tune2$bestTune$mtry-2):(rf_tune2$bestTune$mtry+2), 64, 128, 256),
#   splitrule = c("extratrees", "gini"),
#   min.node.size = c(1,rf_tune2$bestTune$min.node.size)
# )

# set.seed(7)
# model_rf3 <- train(target~.,
#                    data=pp_data,
#                    method="ranger",
#                    metric="logLoss",
#                    tuneGrid = grid3,
#                    # tuneLength = 10,
#                    num.trees = 200,
#                    trControl=control)
# # display results
# 
# saveRDS(model_rf3, "rf_tune3.rds")

# rf_tune3 <- readRDS("rf_tune3.rds")

# set.seed(9560)
# pp_data <- downSample(x = pp_data %>% select(-target),
#                       y = pp_data$target) %>% 
#   rename(target = Class)

# grid3 <- expand.grid(
#   mtry = c(1:2, 20, 30),
#   splitrule = c("extratrees"),
#   min.node.size = 1:2
# )

# set.seed(7)
# model_rf3_bal <- train(target~.,
#                    data=pp_data,
#                    method="ranger",
#                    metric="logLoss",
#                    tuneGrid = grid3,
#                    # tuneLength = 10,
#                    num.trees = 100,
#                    trControl=control)
# display results

# saveRDS(model_rf3_bal, "rf_tune3_bal.rds")

# rf_tune3_bal <- readRDS("rf_tune3_bal.rds")
# 
# rf_tune3_bal

# ggplot(rf_tune3_bal)
