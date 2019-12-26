# Carregar pacotes ---------------------------------------------------------------------------

source("init.R")

# Carregar base -------------------------------------------------------------------------------

pp_data <- readRDS("pp_data_knnInput30.rds")

pp_data <- pp_data %>% select_at(c("target", readRDS("cols_to_select_rfe.rds")))

# 1tun ----------------------------------------------------------------------------------------

# note to start nrounds from 200, as smaller learning rates result in errors so
# big with lower starting points that they'll mess the scales
# tune_grid <- expand.grid(
#   nrounds = seq(from = 100, to = 500, by = 100),
#   max_depth = c(1, 2, 4, 6),
#   eta = c(0.01, 0.01, 0.05, 0.1, 0.3),
#   gamma = 0,
#   colsample_bytree = 1,
#   min_child_weight = 1,
#   subsample = 1
# )
# 
tune_control <- caret::trainControl(
  method="cv",
  number=3,
  classProbs=TRUE,
  summaryFunction=mnLogLoss,
  verboseIter = TRUE,
  allowParallel = TRUE # FALSE for reproducible results
)
# 
# xgb_tune <- caret::train(
#   target ~ ., data = pp_data,
#   trControl = tune_control,
#   tuneGrid = tune_grid,
#   method = "xgbTree",
#   verbose = TRUE
# )
# 
# saveRDS(xgb_tune, "xgb_tune1.rds")

# 2tun ----------------------------------------------------------------------------------------

xgb_tune1 <- readRDS("xgb_tune1.rds")

tune_grid2 <- expand.grid(
  nrounds = 200,
  eta = xgb_tune1$bestTune$eta,
  max_depth = xgb_tune1$bestTune$max_depth,
  gamma = 0.05,
  colsample_bytree = seq(0.1, 0.9, 0.1),
  min_child_weight = 1,
  subsample = c(0.25, 0.5, 0.75)
)

print(tune_grid2)

xgb_tune2 <- caret::train(
  target ~ ., data = pp_data,
  trControl = tune_control,
  tuneGrid = tune_grid2,
  method = "xgbTree",
  verbose = TRUE
)

saveRDS(xgb_tune2, "xgb_tune2.rds")

# xgb_tune2 <- readRDS("xgb_tune2.rds")

# xgb_tune2$bestTune
# 
# ggplot(xgb_tune2)

# 3tun ----------------------------------------------------------------------------------------

# tune_grid3 <- expand.grid(
#   nrounds = 200,
#   eta = xgb_tune2$bestTune$eta,
#   max_depth = xgb_tune2$bestTune$max_depth,
#   gamma = 0,
#   colsample_bytree = c(0.2, 0.4, 0.6),
#   min_child_weight = xgb_tune2$bestTune$min_child_weight,
#   subsample = c(0.5, 0.75, 1.0)
# )

# xgb_tune3 <- caret::train(
#   target ~ ., data = pp_data,
#   trControl = tune_control,
#   tuneGrid = tune_grid3,
#   method = "xgbTree",
#   verbose = TRUE
# )

# saveRDS(xgb_tune3, "xgb_tune3.rds")

# xgb_tune3 <- readRDS("xgb_tune3.rds")

# ggplot(xgb_tune3)

# 4tun -----------------------------------------------------------------------------------------

# tune_grid4 <- expand.grid(
#   nrounds = seq(from = 50, to = 200, by = 50),
#   eta = xgb_tune3$bestTune$eta,
#   max_depth = xgb_tune2$bestTune$max_depth,
#   gamma = c(0, 0.05, 0.1, 0.5, 0.7, 0.9, 1.0),
#   colsample_bytree = xgb_tune3$bestTune$colsample_bytree,
#   min_child_weight = xgb_tune3$bestTune$min_child_weight,
#   subsample = xgb_tune3$bestTune$subsample
# )
# 
# xgb_tune4 <- caret::train(
#   target ~ ., data = pp_data,
#   trControl = tune_control,
#   tuneGrid = tune_grid4,
#   method = "xgbTree",
#   verbose = TRUE
# )

# saveRDS(xgb_tune4, "xgb_tune4.rds")

# xgb_tune4 <- readRDS("xgb_tune4.rds")

# xgb_tune4$bestTune

# ggplot(xgb_tune4)

# 5tun -----------------------------------------------------------------------------------------

# xgb_tune4 <- readRDS("xgb_tune4.rds")
# 
# tune_grid5 <- expand.grid(
#   nrounds = seq(from = 200, to = 1000, by = 100),
#   eta = xgb_tune4$bestTune$eta,
#   max_depth = xgb_tune4$bestTune$max_depth,
#   gamma = xgb_tune4$bestTune$gamma,
#   colsample_bytree = xgb_tune4$bestTune$colsample_bytree,
#   min_child_weight = xgb_tune4$bestTune$min_child_weight,
#   subsample = xgb_tune4$bestTune$subsample
# )
# 
# xgb_tune5 <- caret::train(
#   target ~ ., data = pp_data,
#   trControl = tune_control,
#   tuneGrid = tune_grid5,
#   method = "xgbTree",
#   verbose = TRUE
# )
# 
# saveRDS(xgb_tune5, "xgb_tune5.rds")

# 6tun -----------------------------------------------------------------------------------------

# xgb_tune5 <- readRDS("xgb_tune5.rds")
# 
# tune_grid6 <- expand.grid(
#   nrounds = xgb_tune5$bestTune$nrounds,
#   eta = seq(xgb_tune5$bestTune$eta, 0.1, 0.025),
#   max_depth = (xgb_tune5$bestTune$max_depth-2):(xgb_tune5$bestTune$max_depth+2),
#   gamma = xgb_tune5$bestTune$gamma,
#   colsample_bytree = xgb_tune5$bestTune$colsample_bytree,
#   min_child_weight = xgb_tune5$bestTune$min_child_weight,
#   subsample = xgb_tune5$bestTune$subsample
# )
# 
# xgb_tune5 <- caret::train(
#   target ~ ., data = pp_data,
#   trControl = tune_control,
#   tuneGrid = tune_grid6,
#   method = "xgbTree",
#   verbose = TRUE
# )
# 
# saveRDS(xgb_tune6, "xgb_tune6.rds")
# 
# 
# xgb_tune6 <- readRDS("xgb_tune6.rds")
