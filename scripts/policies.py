

from stable_baselines.common.policies import LstmPolicy
from stable_baselines.common.tf_layers import conv, linear, conv_to_fc
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import FeedForwardPolicy

import numpy as np
import tensorflow as tf

# Policy for logical feature with regressor and dqn
class CustomMlpPolicy_2(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomMlpPolicy_2, self).__init__(*args, **kwargs,
                                           layers=[16, 16],
                                           layer_norm=True,
                                           feature_extraction="mlp")

# Policy for logical feature with regressor and ppo
class CustomLstmMlpPolicy_3(LstmPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomLstmMlpPolicy_3, self).__init__(*args, **kwargs,
                                                    feature_extraction="mlp",
                                                    layers=[16, 16],
                                                    n_lstm=64,
                                                    layer_norm=True)

# Policy for logical feature and dqn
class CustomMlpPolicy_1(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomMlpPolicy_1, self).__init__(*args, **kwargs,
                                           layers=[16, 16],
                                           layer_norm=True,
                                           feature_extraction="mlp")

# Policy for logical feature with ppo
class CustomLstmMlpPolicy_2(LstmPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomLstmMlpPolicy_2, self).__init__(*args, **kwargs,
                                                    feature_extraction="mlp",
                                                    layers=[32, 32],
                                                    n_lstm=64,
                                                    layer_norm=True)


def modified_cnn(scaled_images, **kwargs):
    """
    :param scaled_images: (TensorFlow Tensor) Image input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    activ = tf.nn.relu
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=16, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=16, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_2 = conv_to_fc(layer_2)
    return activ(linear(layer_2, 'fc1', n_hidden=32, init_scale=np.sqrt(2)))

# Policy for direct feature with dqn
class CustomCnnPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomCnnPolicy, self).__init__(*args, **kwargs, cnn_extractor = modified_cnn,
                                              layers = [16, 16], layer_norm = True,
                                              feature_extraction = "cnn")

# Policy for direct feature with ppo
class CustomLstmCnnPolicy(LstmPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomLstmCnnPolicy, self).__init__(*args, **kwargs,
                                                    feature_extraction="cnn",
                                                    cnn_extractor=modified_cnn,
                                                    n_lstm=32,
                                                    layer_norm=True)
