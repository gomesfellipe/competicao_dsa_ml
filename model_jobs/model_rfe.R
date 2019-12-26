# Carregar pacotes ---------------------------------------------------------------------------

source("init.R")

# Carregar base -------------------------------------------------------------------------------

pp_data <- readRDS("pp_data_knnInput30.rds")

set.seed(9560)
pp_data <- downSample(x = pp_data %>% select(-target),
                      y = pp_data$target) %>% 
  rename(target = Class)
table(pp_data$target) 

set.seed(3456)
trainIndex <- createDataPartition(pp_data$target, p = .8, 
                                  list = FALSE, 
                                  times = 1)


pp_data <- pp_data[trainIndex, ]

# RFE -----------------------------------------------------------------------------------------

ctrl <- rfeControl(method="cv", 
                   number=3, 
                   # classProbs=TRUE, 
                   # summaryFunction=mnLogLoss,
                   # search = "random",
                   functions = treebagFuncs,
                   verbose = TRUE,
                   allowParallel = TRUE
                   )

set.seed(10)

model_rfe <- rfe(pp_data %>% select(-target), 
                 pp_data$target,
                 # sizes = seq(4,84,4),
                 sizes = c(23, 24, 25, 60),
                 rfeControl = ctrl)

saveRDS(model_rfe, "model_rfe.rds")

model_rfe <- readRDS("model_rfe.rds")

plot(model_rfe, type = c("g", "o"))

model_rfe$variables %>% 
  filter(Variables == 24) %>% 
  group_by(var) %>% 
  summarise(Overall_mean = mean(Overall)) %>% 
  arrange(-Overall_mean)

cols_to_select_rfe <- model_rfe$variables %>% filter(Variables == 24) %>% pull(var) %>% unique()
saveRDS(cols_to_select_rfe, "cols_to_select_rfe.rds")
# Selecioar as 60 colunasmais importantes e reajustar os modelos

