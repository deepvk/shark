# Toxic content classificators

This repo contains pretrained models:

1. keras CNN model in ```models/cnn_model``` 
See ```toxic_test.ipynb``` for usage.
The model is trained on [jigsaw challenge data](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) and provided for educational purposes

2. tensorfow-js model, converted from 1.
```tfjs.converters.save_keras_model(keras_model, "./path/")```
See [VK Apps demo code](https://github.com/VKCOM/vk-apps-tensorflow-example) for usage
