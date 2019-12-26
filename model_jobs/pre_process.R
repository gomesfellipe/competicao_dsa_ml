pre_process <- function(train, k = NULL){
  
  # Pre processamento:
  model_recipe <- 
    recipe(target ~ ., data= train) %>% 
    step_YeoJohnson(all_numeric()) %>% 
    # step_lincomb(all_numeric()) %>%
    step_knnimpute(all_predictors(), neighbors = k) %>% 
    step_nzv(all_predictors())
    # step_meanimpute(all_numeric()) %>%
    # step_modeimpute(all_nominal()) %>%
  # step_center(all_numeric()) %>%
  # step_scale(all_numeric()) %>%
  # step_pca(all_numeric(),threshold = 0.80)
  
  pp_estimates <- prep(model_recipe, training = train, verbose = T)
  bake(pp_estimates, train)
  
}