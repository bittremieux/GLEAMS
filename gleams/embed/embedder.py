import os

import keras
import numpy as np
import pandas as pd
from keras import backend as K
from keras import Input
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.layers import concatenate, Conv1D, Dense, Flatten, Lambda,\
    MaxPooling1D, Reshape
from keras.models import Model
from keras.optimizers import Adam
from sklearn.metrics import auc, roc_curve

from gleams import config


def euclidean_distance(vects):
    """
    Euclidean distance between two vectors using Keras.

    Parameters
    ----------
    vects
        Two vectors between which to compute the Euclidean distance.

    Returns
    -------
    The Euclidean distance between the two given vectors.
    """
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    """
    Get the shape of the Euclidean distance output.

    Parameters
    ----------
    shapes
        Input shapes to the Euclidean distance calculation.

    Returns
    -------
    The shape of the Euclidean distance output.
    """
    shape1, shape2 = shapes
    return shape1[0], 1


def contrastive_loss(y_true, y_pred):
    """
    Contrastive loss function adapted from Hadsell et al. 2006.
    (http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf)

    The contrastive loss is modified so that it takes a certainty that labels
    are correct into account. This helps the neural network to overcome
    incorrectly labeled instances.

    Parameters
    ----------
    y_true
        The true class labels.
    y_pred
        The predicted class labels.

    Returns
    -------
    The contrastive loss between the true and predicted class labels.
    """
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(config.margin - y_pred, 0))
    return K.mean(y_true * config.loss_label_certainty * square_pred +
                  (1 - y_true * config.loss_label_certainty) * margin_square)


class Embedder:
    """
    A spectrum embedder formed by a Siamese neural network.

    Inspired by the Keras MNIST Siamese network example:
    https://keras.io/examples/mnist_siamese/
    """

    def __init__(self, num_precursor_features: int, num_fragment_features: int,
                 num_ref_spectra_features: int, lr: float,
                 filename: str = 'gleams.h5'):
        """
        Instantiate the Embbeder based on the given number of input features.

        Parameters
        ----------
        num_precursor_features : int
            The number of input precursor features.
        num_fragment_features : int
            The number of input fragment features.
        num_ref_spectra_features : int
            The number of input reference spectra features.
        lr : float
            The learning rate for the Adam optimizer.
        filename : str
            Filename to save the trained Keras model.
        """
        self.num_precursor_features = num_precursor_features
        self.num_fragment_features = num_fragment_features
        self.num_ref_spectra_features = num_ref_spectra_features
        self.lr = lr
        self.filename = filename

        self.model = None

    def save(self):
        """
        Save a model and its training status.
        """
        if self.model is None:
            raise ValueError('The model hasn\'t been constructed yet')
        else:
            self.model.save(self.filename)

    def load(self):
        """
        Load a saved model and its training status from the given file.
        """
        self.model = keras.models.load_model(
            self.filename,
            custom_objects={'contrastive_loss': contrastive_loss})

    def _build_base_model(self) -> Model:
        """
        Construct the model architecture of a single arm of the Siamese
        network.

        The embedder network consists of the following elements:
        - Precursor features are processed using two fully-connected layers of
          dimensions 32 and 5. SELU activation is used.
        - The fragment features and reference spectra features are both
          processed through a single convolutional layer consisting of 30
          filters and kernel size 3, followed by a max pooling layer. SELU
          activation is used.
        - The output of all three elements is concatenated and processed using
          a single fully-connected layer of dimension 32.

        Returns
        -------
        Model
            The Keras neural network model for a single arm of the Siamese
            network that takes as input the features specified for this
            Embedder.
        """
        # Precursor features are processed through two dense layers.
        precursor_input = Input((self.num_precursor_features,))
        precursor_dense32 = (Dense(32, activation='selu',
                                   kernel_initializer='he_uniform')
                             (precursor_input))
        precursor_dense5 = (Dense(5, activation='selu',
                                  kernel_initializer='he_uniform')
                            (precursor_dense32))

        filters = 30
        kernel_size = 3
        strides = 1
        pool_size = 1
        pool_strides = 2
        # Fragment features are processed through a single convolutional and
        # max pooling layer.
        fragment_input = Input((self.num_fragment_features,))
        fragment_input_reshape =\
            Reshape((self.num_fragment_features, 1))(fragment_input)
        fragment_conv = Conv1D(filters, kernel_size, strides=strides,
                               activation='selu')(fragment_input_reshape)
        fragment_maxpool = MaxPooling1D(pool_size, pool_strides)(fragment_conv)
        fragment_output = Flatten()(fragment_maxpool)

        # Reference spectra features are processed through a single
        # convolutional and max pooling layer.
        ref_spectra_input = Input((self.num_ref_spectra_features,))
        ref_spectra_input_reshape =\
            Reshape((self.num_ref_spectra_features, 1))(ref_spectra_input)
        ref_spectra_conv = Conv1D(filters, kernel_size, strides=strides,
                                  activation='selu')(ref_spectra_input_reshape)
        ref_spectra_maxpool = (MaxPooling1D(pool_size, pool_strides)
                               (ref_spectra_conv))
        ref_spectra_output = Flatten()(ref_spectra_maxpool)

        # Combine all outputs and add a final dense layer.
        output_layer = (Dense(32, activation='selu',
                              kernel_initializer='he_uniform')
                        (concatenate([precursor_dense5, fragment_output,
                                      ref_spectra_output])))

        return Model(inputs=[precursor_input, fragment_input,
                             ref_spectra_input],
                     outputs=[output_layer])

    def build_siamese_model(self):
        """
        Build the Embedder's Siamese network and compile it to optimize the
        contrastive loss using Adam.

        Both arms of the Siamese network will use the same base network, i.e.
        the weights are tied between both arms.
        """
        # Both arms of the Siamese network use the same model, i.e. the weights
        # are tied.
        base_model = self._build_base_model()
        input_left = [Input((self.num_precursor_features,)),
                      Input((self.num_fragment_features,)),
                      Input((self.num_ref_spectra_features,))]
        input_right = [Input((self.num_precursor_features,)),
                       Input((self.num_fragment_features,)),
                       Input((self.num_ref_spectra_features,))]
        output_left = base_model(input_left)
        output_right = base_model(input_right)

        # Euclidean distance between two embeddings.
        distance = (Lambda(euclidean_distance, eucl_dist_output_shape)
                    ([output_left, output_right]))

        # Train using Adam to optimize the contrastive loss.
        self.model = Model(inputs=[*input_left, *input_right],
                           outputs=distance)
        self.model.compile(Adam(self.lr), contrastive_loss)

    def train(self, x_train, y_train, batch_size: int, num_epochs: int,
              x_val=None, y_val=None):
        if self.model is None:
            raise ValueError("The model hasn't been constructed yet")

        filename, ext = os.path.splitext(self.filename)
        filename_log = f'{filename}.log'
        # CrocHistory has to be added after CSVLogger because it uses the same
        # log file.
        callbacks = [ModelCheckpoint(filename + '.epoch{epoch:03d}' + ext),
                     CSVLogger(filename_log),
                     CrocHistory(x_val, y_val, filename_log)]
        self.model.fit(
            x_train, y_train, batch_size, num_epochs, callbacks=callbacks,
            validation_data=((x_val, y_val)
                             if x_val is not None and y_val is not None
                             else None))

    def predict(self, x):
        return self.model.predict(x)


class CrocHistory(keras.callbacks.Callback):
    """
    Track the AUC CROC on the validation data after each epoch ends.
    """

    # Alpha = 14 maps x = 0.05 to 0.5.
    alpha = 14

    def __init__(self, x_val, y_val, log_filename=None):
        super().__init__()

        self.x_val = x_val
        self.y_val = y_val
        self.log_filename = log_filename
        self.croc_aucs = None

    def on_train_begin(self, logs=None):
        self.croc_aucs = []

    def on_epoch_end(self, epoch, logs=None):
        if self.x_val is not None:
            y_pred = self.model.predict(self.x_val)
            fpr, tpr, _ = roc_curve(self.y_val, 1 - y_pred / y_pred.max())
            # Expontial CROC transformation from Swamidass et al. 2010.
            croc_fpr = ((1 - np.exp(-self.alpha * fpr)) /
                        (1 - np.exp(-self.alpha)))
            self.croc_aucs.append(auc(croc_fpr, tpr))

    def on_train_end(self, logs=None):
        if self.log_filename is not None and len(self.croc_aucs) > 0:
            df_log = pd.read_csv(self.log_filename)
            df_log['croc_auc'] = self.croc_aucs
            df_log.to_csv(self.log_filename, index=False)
