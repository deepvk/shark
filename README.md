# Toxic content classificators

This repo contains pretrained models:

**keras CNN model in ```models/cnn_model```***

See ```toxic_test.ipynb``` for usage.

The model is trained on [jigsaw challenge data](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) and provided for educational purposes

***tensorfow-js CNN model.***

Converted from keras:
```tfjs.converters.save_keras_model(keras_model, "./path/")```

See [VK Apps demo code](https://github.com/VKCOM/vk-apps-tensorflow-example) for usage.
Live demo here: https://vk.com/app6759433
