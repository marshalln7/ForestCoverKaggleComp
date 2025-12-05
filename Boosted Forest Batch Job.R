
library(vroom)
library(tidyverse)
library(tidymodels)
library(dplyr)
library(discrim)
library(doParallel)
library(parallel)

set.seed(42)

train_data <- vroom("train.csv") |>
  mutate(Cover_Type = as.factor(Cover_Type)) |>
  mutate(across(-Cover_Type, as.numeric),
         Cover_Type = as.factor(Cover_Type))
test_data <- vroom("test.csv") |>
  mutate(across(everything(), as.numeric))

recipe <- recipe(Cover_Type ~ ., data=train_data) |>
  step_mutate(Distance_to_Hydrology = sqrt((Horizontal_Distance_To_Hydrology ** 2) + Vertical_Distance_To_Hydrology ** 2)) |>
  step_rm(Id, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology)

# prepped_recipe <- prep(recipe)
# train_data_2 <- bake(prepped_recipe, new_data=train_data)
# vroom_write(x=train_data_2, file="Baked Train Data.csv", delim=",")

my_model <- boost_tree(
  trees = tune(),           # number of trees
  tree_depth = tune(),      # max depth of trees
  min_n = tune(),           # minimum observations in node
  loss_reduction = tune(),  # minimum loss reduction for split
  sample_size = tune(),     # proportion of data for each tree
  mtry = tune(),            # number of predictors randomly sampled
  learn_rate = tune()       # learning rate (eta)
) |>
  set_engine("xgboost") |>
  set_mode("classification")

wf <- workflow() %>%
  add_recipe(recipe) %>%
  add_model(my_model)

grid_of_tuning_params <- grid_space_filling(
  trees(), 
  tree_depth(), 
  min_n(), 
  loss_reduction(), 
  sample_prop(),  
  finalize(mtry(), train_data),  
  learn_rate(), 
  size = 20  # Only 30 combinations instead of 78k!
)

folds <- vfold_cv(train_data, v = 5, repeats=1)


# Pick number of cores
n_cores <- detectCores() - 1         
cl <- makeCluster(n_cores)
registerDoParallel(cl)

# Run the cross validation in parallel
CV_results <- wf %>%
  tune_grid(
    resamples = folds,
    grid = grid_of_tuning_params,
    metrics = metric_set(accuracy),
    control = control_grid(save_pred = TRUE, verbose = TRUE)
  )

stopCluster(cl)
registerDoSEQ()


bestTune <- CV_results %>%
  select_best(metric="accuracy")

final_wf <- wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=train_data)

## Predict
tree_predictions <- final_wf %>%
  predict(new_data = test_data)

kaggle_submission <- test_data |>
  bind_cols(tree_predictions) |>
  rename(Cover_Type = .pred_class) |>
  select(Id, Cover_Type)

vroom_write(x=kaggle_submission, file="Boosted Forest Predictions.csv", delim=",")

# Send me a notification over pushbullet

system(
  "curl -u o.RuFXlQ50Yz0RzfcMYVTIFsOSeszUpm5q: \
    -X POST https://api.pushbullet.com/v2/pushes \
    -d type=note \
    -d title=\"R Job Complete\" \
    -d body=\"Your batch job has finished running.\""
)