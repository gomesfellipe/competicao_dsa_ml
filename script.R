# Carregar pacotes ----------------------------------------------------------------------------
library(readr)
library(dplyr)
library(purrr)
library(caret)
library(recipes)
library(doParallel)

# Paralelizar 
cl <- makePSOCKcluster(16, outfile="")
registerDoParallel(cl)

# Leitura dados de treino
train <- read_csv(file = "dataset_treino.csv")

# AED -----------------------------------------------------------------------------------------

# summary(train$v50)
# hist(train$v50)
# hist(log(train$v50))
# boxplot(train$v50)
# qqnorm(train$v99)

# Data wrangling  ---------------------------------------------------------

train <- 
  train %>% 
  mutate(target = ifelse(target == 1, "sim", "nao") %>% as.factor()) %>% 
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
  ) %>% 
  filter(v1 < 20, v10 < 15, v101 < 20, v102 < 15, v103 < 15, v104 < 15, v106 > 0, 
         v108 < 15, v109 < 15, v11 > 0, v111 < 15, v114 > 0, v115 > 0, v116 < 15, 
         v117 < 20, v117 > 0, v118 < 20, v118 > 0, v119 < 20, v120 < 6, v121 < 20,
         v122 < 20, v123 < 20, v124 < 15, v126 < 15, v127 < 15, v130 < 15, v131 < 15,
         v14 > 5, v15 > 0, v15 < 15, v16 < 20, v16 > 0, v17 < 20, v18 < 20, v20 > 5,
         v21 < 20, v26 < 10, v27 < 10, v28 < 20, v29 < 20, v29 > 0, v35 < 20, v35 > 0, 
         v36 < 20, v36 > 0, v37 < 10, v39 < 14, v4 < 15, v42 > 0, v42 < 20, v48 > 0,
         v50 < 20, v57 < 11, v59 < 20, v59 > 0, v6 < 15, v61 > 0, v69 > 0, v69 < 20,
         v7 < 15, v70 > 0, v72 < 10, v78 > 0, v78 < 20, v80 < 15, v81 < 15, v84 < 15,
         v85 < 15, v88 < 15, v9 > 0,  v9 < 20, v90 > 0,  v90 < 2.5, v92 < 4, v93 > 0, 
         v93 < 15, v94 > 0, v94 < 15, v98 < 19, v98 > 0, v99 < 6)

# Pre processamento ---------------------------------------------------------------------------

model_recipe <- 
  recipe(target ~ ., data= train) %>% 
  step_YeoJohnson(all_numeric()) %>% 
  step_knnimpute(all_predictors(), neighbors = 20) %>%
  step_lincomb(all_numeric()) %>%
  step_nzv(all_predictors()) 
# step_medianimpute(all_numeric()) %>%
# step_modeimpute(all_nominal()) %>%
# step_center(all_numeric()) %>%
# step_scale(all_numeric()) %>%
# step_pca(all_numeric(),threshold = 0.80)

pp_estimates <- prep(model_recipe, training = train, verbose = T)
pp_data <- bake(pp_estimates, mutate_if(train, is.character, as.factor) )

saveRDS(pp_data, "pp_data.rds")
# ML ------------------------------------------------------------------------------------------
control <- trainControl(method="cv", 
                        number=3, 
                        classProbs=TRUE, 
                        summaryFunction=mnLogLoss,
                        verboseIter = TRUE, 
                        allowParallel = TRUE)

# GLM
set.seed(7)
model_glm <- train(target~.,
                   data=pp_data,
                   method="glm",
                   metric="logLoss",
                   trControl=control
                   )
model_glm # logLoss 0.4991938
saveRDS(model_glm, "model_glm.rds")

# Random Forest (ranger)

# grid <- expand.grid(
#   mtry = floor(ncol(pp_data) * c(.05, .15, .25, .333, .4)),
#   min.node.size = c(1, 3, 5, 10),
#   splitrule = c("extratrees")
# )
# 
# set.seed(7)
# model_rf <- train(target~.,
#                    data=pp_data,
#                    method="ranger",
#                    metric="logLoss",
#                    tuneGrid=grid,
#                    trControl=control,verbose = T)
# 
# model_rf2
# ggplot(model_rf2)
# saveRDS(model_rf, "model_rf.rds")

# XGBoost

# Default

grid_default <- expand.grid(
  nrounds = 100,
  max_depth = 6,
  eta = 0.3,
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 1
)

model_xgb_baseline <- caret::train(
  target~.,
  data=pp_data,
  trControl = control,
  tuneGrid = grid_default,
  method = "xgbTree",
  verbose = TRUE
)

# Tunning
# nrounds: Numero de arvores, default: 100
# max_depth: Profundidade máxima da árvore, default: 6
# eta: Taxa de Aprendizagem, default: 0.3
# gamma: Ajustar a Regularização, default: 0
# colsample_bytree: Amostragem em coluna, default: 1
# min_child_weight: Peso mínimo das folhas, default: 1
# subsample: Amostragem de linha, default: 1

# 1 Tuning:
# - nrouns
# - eta
# - max_depth

model_xgb_baseline # logLoss 0.5046261
saveRDS(model_xgb_baseline, "model_xgb_baseline.rds")

nrounds <- 1000

tune_grid <- expand.grid(
  nrounds = seq(from = 200, to = nrounds, by = 50),
  eta = c(0.025, 0.05, 0.1, 0.3),
  max_depth = c(2, 3, 4, 5, 6),
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 1
)

model_xgb_tune <- caret::train(
  target~.,
  data=pp_data,
  trControl = control,
  tuneGrid = tune_grid,
  method = "xgbTree",
  verbose = TRUE
)

model_xgb_tune
ggplot(model_xgb_tune)
saveRDS(model_xgb_tune, "model_xgb_tune.rds")

# 2 Tunning
# - min_child_weight

tune_grid2 <- expand.grid(
  nrounds = seq(from = 50, to = nrounds, by = 50),
  eta = model_xgb_tune$bestTune$eta,
  max_depth = ifelse(model_xgb_tune$bestTune$max_depth == 2,
                     c(model_xgb_tune$bestTune$max_depth:4),
                     model_xgb_tune$bestTune$max_depth - 1:model_xgb_tune$bestTune$max_depth + 1),
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = c(1, 2, 3),
  subsample = 1
)

model_xgb_tune2 <- caret::train(
  target~.,
  data=pp_data,
  trControl = control,
  tuneGrid = tune_grid2,
  method = "xgbTree",
  verbose = TRUE
)

model_xgb_tune2
ggplot(model_xgb_tune2)
saveRDS(model_xgb_tune2, "model_xgb_tune2.rds")

# 3 Tunning
# - colsample_bytree
# - subsample

tune_grid3 <- expand.grid(
  nrounds = seq(from = 50, to = nrounds, by = 50),
  eta = model_xgb_tune$bestTune$eta,
  max_depth = model_xgb_tune2$bestTune$max_depth,
  gamma = 0,
  colsample_bytree = c(0.4, 0.6, 0.8, 1.0),
  min_child_weight = model_xgb_tune2$bestTune$min_child_weight,
  subsample = c(0.5, 0.75, 1.0)
)

model_xgb_tune3 <- caret::train(
  target~.,
  data=pp_data,
  trControl = control,
  tuneGrid = tune_grid3,
  method = "xgbTree",
  verbose = TRUE
)

model_xgb_tune3
ggplot(model_xgb_tune3)
saveRDS(model_xgb_tune3, "model_xgb_tune3.rds")

# 4 Tunning 
# - gamma

tune_grid4 <- expand.grid(
  nrounds = seq(from = 50, to = nrounds, by = 50),
  eta = model_xgb_tune$bestTune$eta,
  max_depth = model_xgb_tune2$bestTune$max_depth,
  gamma = c(0, 0.05, 0.1, 0.5, 0.7, 0.9, 1.0),
  colsample_bytree = model_xgb_tune3$bestTune$colsample_bytree,
  min_child_weight = model_xgb_tune2$bestTune$min_child_weight,
  subsample = model_xgb_tune3$bestTune$subsample
)

model_xgb_tune4 <- caret::train(
  target~.,
  data=pp_data,
  trControl = tune_control,
  tuneGrid = tune_grid4,
  method = "xgbTree",
  verbose = TRUE
)

model_xgb_tune4
ggplot(model_xgb_tune4)
saveRDS(model_xgb_tune4, "model_xgb_tune4.rds")

# 5 Tunning
# - eta

tune_grid5 <- expand.grid(
  nrounds = seq(from = 100, to = 10000, by = 100),
  eta = c(0.01, 0.015, 0.025, 0.05, 0.1),
  max_depth = model_xgb_tune2$bestTune$max_depth,
  gamma = model_xgb_tune4$bestTune$gamma,
  colsample_bytree = model_xgb_tune3$bestTune$colsample_bytree,
  min_child_weight = model_xgb_tune2$bestTune$min_child_weight,
  subsample = model_xgb_tune3$bestTune$subsample
)

model_xgb_tune5 <- caret::train(
  target~.,
  data=pp_data,
  trControl = tune_control,
  tuneGrid = tune_grid5,
  method = "xgbTree",
  verbose = TRUE
)

model_xgb_tune5
ggplot(model_xgb_tune5)
saveRDS(model_xgb_tune5, "model_xgb_tune5.rds")

# Final Model
final_grid <- expand.grid(
  nrounds = model_xgb_tune5$bestTune$nrounds,
  eta = model_xgb_tune5$bestTune$eta,
  max_depth = model_xgb_tune5$bestTune$max_depth,
  gamma = model_xgb_tune5$bestTune$gamma,
  colsample_bytree = model_xgb_tune5$bestTune$colsample_bytree,
  min_child_weight = model_xgb_tune5$bestTune$min_child_weight,
  subsample = model_xgb_tune5$bestTune$subsample
)

final_grid

model_xgb_final <- caret::train(
  target~.,
  data=pp_data,
  trControl = control,
  tuneGrid = final_grid,
  method = "xgbTree",
  verbose = TRUE
)

model_xgb_final
ggplot(model_xgb_final)
saveRDS(model_xgb_final, "model_xgb_final.rds")



# Test --------------------------------------------------------------------

test_id <- read_csv(file = "dataset_teste.csv")$ID

test <- read_csv(file = "dataset_teste.csv")

# Data wrangling
test <- 
  test %>%
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

# Pre processamento
model_recipe <- 
  recipe( ~ ., data= test) %>% 
  step_YeoJohnson(all_numeric()) %>% 
  step_knnimpute(all_predictors(), neighbors = 20) %>%
  step_lincomb(all_numeric()) %>%
  step_nzv(all_predictors()) 

test_estimates <- prep(model_recipe, training = test, verbose = T)
test_data <- bake(test_estimates, test)

# Sub
data.frame(
  ID = test_id,
  PredictedProb = predict(model_xgb, test_data, type = 'prob')$sim) %>% 
  write.csv("sub_rf_tun.csv", row.names = F)

