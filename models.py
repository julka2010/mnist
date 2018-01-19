from keras import(
    optimizers,
    regularizers,
)
from keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
)
import keras.backend as K
from keras.layers import (
    Activation,
    add,
    AlphaDropout,
    BatchNormalization,
    concatenate,
    Conv2D,
    Dense,
    dot,
    Dropout,
    Embedding,
    Flatten,
    Input,
    merge,
    Reshape,
)
from keras.models import (
    load_model,
    Model,
)
from keras.utils import Sequence
import numpy as np

def simple_conv(img_shape):
    input_images = Input(shape=img_shape)
    x = input_images
    x = Conv2D(12, (5, 5), strides=1, padding='same', activation='relu', use_bias=True, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = Conv2D(24, (5, 5), strides=1, padding='same', activation='relu', use_bias=True, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = Conv2D(48, (5, 5), strides=1, padding='same', activation='relu', use_bias=True, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = Flatten()(x)
    x = Dense(20, use_bias=True, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(10, use_bias=True, activation='softmax')(x)

    return Model(input_images, x)


class PseudoLabelingSequence(Sequence):
    def __init__(
        self,
        model,
        train_x, train_y,
        pseudo_x, initial_pseudo_y,
        batch_size,
        pseudo_labeling_proportion=0.25,
        ):
        self.model = model
        self.x, self.y = train_x.copy(), train_y.copy()
        self.pseudo_x, self.pseudo_y = pseudo_x.copy(), initial_pseudo_y.copy()
        self.batch_size = batch_size
        self.pseudo_share = pseudo_labeling_proportion

    def __len__(self):
        return int((1+self.pseudo_share) * len(self.x) / self.batch_size) + 1

    def __getitem__(self, idx):
        def bound(id_, share):
            return int(id_ * self.batch_size * share)
        real_x = self.x[
            bound(idx, 1 - self.pseudo_share):
            bound(idx + 1, 1 - self.pseudo_share)
        ]
        real_y = self.y[
            bound(idx, 1 - self.pseudo_share):
            bound(idx + 1, 1 - self.pseudo_share)
        ]
        pseudo_x = self.pseudo_x[
            bound(idx, self.pseudo_share):
            bound(idx + 1, self.pseudo_share)
        ]
        pseudo_y = self.pseudo_y[
            bound(idx, self.pseudo_share):
            bound(idx + 1, self.pseudo_share)
        ]
        shuffle_order= np.random.permutation(len(real_x) + len(pseudo_x))
        x = np.concatenate((real_x, pseudo_x))
        y = np.concatenate((real_y, pseudo_y))
        return x, y

    def on_epoch_end(self):
        shuffle_order = np.random.permutation(len(self.x))
        self.x = self.x[shuffle_order]
        self.y = self.y[shuffle_order]
        np.random.shuffle(self.pseudo_x)
        self.pseudo_y = self.model.predict(
            self.pseudo_x,
            batch_size=self.batch_size
        )
