library(keras)
mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y
x_train <- array_reshape(x_train, c(nrow(x_train), c(28, 28), 1))
x_test <- array_reshape(x_test, c(nrow(x_test), c(28, 28), 1))
x_train <- x_train / 255
x_test <- x_test / 255
y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)
model <- keras_model_sequential() 
model %>% 
  layer_conv_2d(filters = 64, activation = "relu", input_shape = c(28, 28, 1), kernel_size = 3) %>% 
  layer_conv_2d(filters = 32, activation = "relu", kernel_size = 3) %>%
  layer_flatten() %>%
  layer_dense(units = 10, activation = 'softmax')
model %>% compile(loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(),
  metrics = c('accuracy'))
history <- model %>% fit(x_train, y_train, 
  epochs = 3, validation_data = list(x_test, y_test))
plot(history)
model %>% evaluate(x_test, y_test)
model %>% predict_classes(x_test)