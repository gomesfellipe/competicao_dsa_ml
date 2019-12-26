
# Carregar pacotes ----------------------------------------------------------------------------

source("init.R")

# Carregar dataset e select features ----------------------------------------------------------

train <- 
  read_csv(file = "dataset_treino.csv")  

train <- 
  train %>% 
  select_features() %>% 
  mutate(target = as.character(target)) %>% 
  mutate(target = ifelse(target == 1, "sim", "nao")) %>% 
  mutate(target = as.factor(target)) %>% 
  filter(v6 != 20) %>% 
  filter(v7 != 20) %>% 
  filter(v11 != 0) %>% 
  filter(v16 != 20)%>% 
  filter(v17 != 20)%>% 
  filter(v18 != 20)%>% 
  filter(v20 != 0) %>% 
  filter(v26 != 20)%>% 
  filter(v27 != 20)%>% 
  filter(v37 != 20)%>% 
  filter(v57 != 20)%>% 
  filter(v80 != 20)%>% 
  filter(v81 != 20)%>% 
  filter(v84 != 20)%>% 
  filter(v88 < 15) %>% 
  filter(v92 < 8) 

train <- 
  bind_cols(
    train %>% select_if(~ !is.numeric(.x)),
    train %>% select_if(is.numeric)
  ) %>% 
  mutate_if(is.character, as.factor)

# Balanceamento -------------------------------------------------------------------------------

# set.seed(1)
# down_train <- 
#   downSample(x = train[,-1], y = train$target) %>% 
#   rename(target = Class)

# Pre processamento ---------------------------------------------------------------------------

model_recipe <- 
  recipe(target ~ ., data= train) %>% 
  step_YeoJohnson(all_numeric()) %>% 
  # step_bagimpute(all_predictors()) %>%
  step_meanimpute(all_numeric()) %>%
  step_modeimpute(all_nominal()) %>%
  step_nzv(all_predictors()) 
# step_center(all_numeric()) %>%
# step_scale(all_numeric()) %>%
# step_pca(all_numeric(),threshold = 0.80)

pp_estimates <- prep(model_recipe, training = train, verbose = T)
pp_data <- bake(pp_estimates, train)
pp_data %>% map_dbl(~ sum(is.na(.x)))

saveRDS(pp_data, "pp_data.rds")



