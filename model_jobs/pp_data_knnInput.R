# Carregar pacotes ----------------------------------------------------------------------------

library(readr)
library(dplyr)
library(purrr)
library(glue)
library(ggplot2)
library(gridExtra)
library(caret)
library(recipes)
source("select_features.R")
source("pre_process.R")

# Carregar dataset e select features ----------------------------------------------------------

train <- 
  read_csv(file = "dataset_treino.csv")  

train <- 
  train  %>% 
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
  filter(v1 < 20,
         v10 < 15,
         v101 < 20,
         v102 < 15,
         v103 < 15,
         v104 < 15,
         v106 > 0,
         v108 < 15,
         v109 < 15,
         v11 > 0,
         v100 < 20,
         v100 > 0,
         v111 < 15,
         v114 > 0,
         v115 > 0,
         v116 < 15,
         v117 < 20,
         v117 > 0,
         v118 < 20,
         v118 > 0,
         v119 < 20,
         v120 < 6,
         v121 < 20,
         v122 < 20,
         v123 < 20,
         v124 < 15,
         v126 < 15,
         v127 < 15,
         v130 < 15,
         v131 < 15,
         v14 > 5,
         v15 > 0,
         v15 < 15,
         v16 < 20,
         v16 > 0,
         v17 < 20,
         v18 < 20,
         v20 > 5,
         v21 < 20,
         v26 < 10,
         v27 < 10,
         v28 < 20,
         v29 < 20,
         v29 > 0,
         v35 < 20,
         v35 > 0,
         v36 < 20,
         v36 > 0,
         v37 < 10,
         v39 < 14,
         v4 < 15,
         v42 > 0,
         v42 < 20,
         v48 > 0,
         v50 < 20,
         v57 < 11,
         v58 > 0,
         v58 < 20,
         v59 < 20,
         v59 > 0,
         v6 < 15,
         v61 > 0,
         v69 > 0,
         v69 < 20,
         v7 < 15,
         v70 > 0,
         v72 < 10,
         v78 > 0,
         v78 < 20,
         v80 < 15,
         v81 < 15,
         v84 < 15,
         v85 < 15,
         v88 < 15,
         v9 > 0, 
         v9 < 20,
         v90 > 0, 
         v90 < 2.5,
         v92 < 4,
         v93 > 0,
         v93 < 15,
         v94 > 0,
         v94 < 15,
         v98 < 19,
         v98 > 0,
         v99 < 6)

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
#   rename(target = Class) %>%
#   mutate(target = as.factor(target))

# Pre processamento ---------------------------------------------------------------------------

model_recipe <- 
  recipe(target ~ ., data= train) %>% 
  step_YeoJohnson(all_numeric()) %>% 
  step_knnimpute(all_predictors(), neighbors = 20) %>%
  # step_medianimpute(all_numeric()) %>%
  step_lincomb(all_numeric()) %>%
  step_nzv(all_predictors()) 
# step_modeimpute(all_nominal()) %>%
# step_center(all_numeric()) %>%
# step_scale(all_numeric()) %>%
# step_pca(all_numeric(),threshold = 0.80)

pp_estimates <- prep(model_recipe, training = train, verbose = T)
pp_data <- bake(pp_estimates, train)

# Salvar --------------------------------------------------------------------------------------

saveRDS(pp_data, "pp_data_knnInput20.rds")

pp_data %>% map_dbl(~ sum(is.na(.x)))
