source("init.R")

pp_data <- readRDS("pp_data_knnInput30.rds")

# Melhor modelo -------------------------------------------------------------------------------

model_glm <- readRDS("model_glm.rds") 
model_glmnet <- readRDS("model_glmnet.rds") 
model_glmboost <- readRDS("model_glmboost.rds")
model_rf <- readRDS("model_rf_best.rds")
# model_gbm <- readRDS("model_gbm.rds")
model_xgbTree <- readRDS("model_xgbTree.rds")
# model_xgbDART <- readRDS("model_xgbDART.rds")
model_avNNET <- readRDS("model_avNNET.rds")
model_pcaNNet <- readRDS("model_pcaNNet.rds")
model_nnet <- readRDS("model_nnet.rds")

model_glmboost_tun <- readRDS("model_glmboost_tun.rds")
model_rf_tun <- readRDS("model_rf_tun.rds")


resamps <- resamples(list(model_glm = model_glm,
                          model_glmnet = model_glmnet,
                          model_glmboost = model_glmboost,
                          model_rf = model_rf,
                          # model_gbm = model_gbm,
                          model_xgbTree = model_xgbTree,
                          # model_xgbDART = model_xgbDART,
                          model_pcaNNet = model_pcaNNet,
                          # model_avNNET = model_avNNET,
                          model_nnet = model_nnet
))

summary(resamps) 
bwplot(resamps)


# Submissao -----------------------------------------------------------------------------------

test_id <- read_csv(file = "dataset_teste.csv")$ID

test <- 
  read_csv(file = "dataset_teste.csv")  %>%
  select(-ID,
         -v112, -v113, -v12, -v125, -v128, -v129, -v13, -v19, -v22, 
         -v23, -v25, -v3, -v32, -v33, -v34, -v38, -v41, -v43, -v46,
         -v49, -v52, -v53, -v54, -v55, -v56, -v60, -v63, -v64, -v65,
         -v67, -v74, -v75, -v76, -v77, -v8, -v82, -v83, -v86, -v89,
         -v95, -v96, -v97) %>% 
  mutate(v107 = ifelse(v107 %in% c("E", "C", "D", "B"), v107, "Z"),
         v24 = ifelse(v24 %in% c("E", "D", "C"), v24, "Z"),
         v30 = ifelse(v30 %in% c("C", "G", "D"), v30, "Z"),
         v31 = ifelse(is.na(v31), "D", v31),
         v47 = ifelse(v47 %in% c("C", "I"), v47, "Z"),
         v71 = ifelse(v71 %in% c("F", "B", "C"), v71, "Z"),
         v79 = ifelse(v79 %in% c("C", "B", "E"), v79, "Z"),
         v91 = ifelse(v91 %in% c("A", "G", "C", "B"), v91, "Z")
  )

model_recipe <- 
  recipe( ~ ., data= test) %>% 
  step_YeoJohnson(all_numeric()) %>% 
  step_knnimpute(all_predictors(), neighbors = 20) %>%
  # step_medianimpute(all_numeric()) %>%
  step_lincomb(all_numeric()) %>%
  step_nzv(all_predictors()) 
# step_modeimpute(all_nominal()) %>%
# step_center(all_numeric()) %>%
# step_scale(all_numeric()) %>%
# step_pca(all_numeric(),threshold = 0.80)

test_estimates <- prep(model_recipe, training = test, verbose = T)
test_data <- bake(test_estimates, test)
test_data %>% map_dbl(~ sum(is.na(.x)))
test_data

# GLM
model_glm
confusionMatrix(model_glm)
data.frame(
  ID = test_id,
  PredictedProb = predict(model_glm, test_data, type = 'prob')$sim) %>% 
  write.csv("sub_glm.csv", row.names = F)

# glmnet
model_glmnet
plot(model_glmnet) 
data.frame(
  ID = test_id,
  PredictedProb = predict(model_glmnet, test_data, type = 'prob')$sim) %>% 
  write.csv("sub_glmnet.csv", row.names = F)

# model_glmboost
model_glmboost
plot(model_glmboost)
data.frame(
  ID = test_id,
  PredictedProb = predict(model_glmboost, test_data, type = 'prob')$sim) %>% 
  write.csv("sub_glmboost.csv", row.names = F)

# model_rf
model_rf
confusionMatrix(model_rf) 
data.frame(
  ID = test_id,
  PredictedProb = predict(model_rf, test_data, type = 'prob')$sim) %>% 
  write.csv("sub_rf.csv", row.names = F)

# # model_gbm
# model_gbm
# data.frame(
#   ID = test_id,
#   PredictedProb = predict(model_gbm, test_data, type = 'prob')$sim) %>% 
#   write.csv("sub_gbm.csv", row.names = F)

# model_xgbTree
model_xgbTree
data.frame(
  ID = test_id,
  PredictedProb = predict(model_xgbTree, test_data, type = 'prob')$sim) %>% 
  write.csv("sub_xgbTree.csv", row.names = F)

# # model_xgbDART
# model_xgbDART
# data.frame(
#   ID = test_id,
#   PredictedProb = predict(model_xgbDART, test_data, type = 'prob')$sim) %>% 
#   write.csv("sub_xgbDART.csv", row.names = F)

# model_pcaNNet
model_pcaNNet
confusionMatrix(model_pcaNNet)
data.frame(
  ID = test_id,
  PredictedProb = predict(model_pcaNNet, test_data, type = 'prob')$sim) %>% 
  write.csv("sub_pcaNNet.csv", row.names = F)

# model_avNNET
model_avNNET
confusionMatrix(model_avNNET)
data.frame(
  ID = test_id,
  PredictedProb = predict(model_avNNET, test_data, type = 'prob')$sim) %>% 
  write.csv("sub_avNNET.csv", row.names = F)

# model_nnet
model_nnet
confusionMatrix(model_nnet)
data.frame(
  ID = test_id,
  PredictedProb = predict(model_nnet, test_data, type = 'prob')$sim) %>% 
  write.csv("sub_nnet.csv", row.names = F)

# model_rfe
model_rfe
confusionMatrix(model_rfe)
data.frame(
  ID = test_id,
  PredictedProb = predict(model_rfe, test_data, type = 'prob')$sim) %>% 
  write.csv("sub_rfe.csv", row.names = F)

# model_rfe
rf_tune2
ggplot(rf_tune2)
confusionMatrix(rf_tune2)
data.frame(
  ID = test_id,
  PredictedProb = predict(rf_tune2, test_data, type = 'prob')$sim) %>% 
  write.csv("sub_rf_tune2.csv", row.names = F)

# model_rf_tun
model_rf_tun
confusionMatrix(model_rf_tun) 
data.frame(
  ID = test_id,
  PredictedProb = predict(model_rf_tun, test_data, type = 'prob')$sim) %>% 
  write.csv("sub_rf_tun.csv", row.names = F)


# Rascunho ------------------------------------------------------------------------------------

pp_estimates_knnInput <- readRDS(pp_estimates_knnInput, "pp_estimates_knnInput.rds")


# # RFE
# set.seed(10)
# 
# ctrl <- rfeControl(functions = treebagFuncs,
#                    method = "cv",
#                    number = 3,
#                    verbose = T)
# 
# model_rfe <- rfe(pca_data %>% select(-target), 
#                  pca_data$target,
#                  sizes = seq(10,60, 10),
#                  rfeControl = ctrl)
# 
# fit <- predict(model_rfe, test_data, type = 'prob')
# 
# data.frame(
#   ID = test_id,
#   PredictedProb = fit$sim
# ) %>% 
#   write.csv("subrfe.csv", row.names = F)
# 
# 
# plot(model_rfe, type = c("g", "o"))