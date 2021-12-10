import logging
import os
from typing import List

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import Input
from tensorflow.keras.callbacks import Callback, CSVLogger, ModelCheckpoint
from tensorflow.keras.layers import concatenate, Conv1D, Dense, Flatten, \
    Lambda, MaxPooling1D, Reshape
from tensorflow.keras.models import Model
from tensorflow_addons.optimizers import RectifiedAdam

from gleams import config
from gleams.nn import data_generator


logger = logging.getLogger('gleams')


def euclidean_distance(xy: List[tf.Tensor]):
    """
    Euclidean distance between two vectors using Keras.

    Parameters
    ----------
    xy : List[tf.Tensor]
        List of two vectors between which to compute the Euclidean distance.

    Returns
    -------
    The Euclidean distance between the two given vectors.
    """
    x, y = xy
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


def contrastive_loss(y_true: tf.Tensor, y_pred: tf.Tensor):
    """
    Contrastive loss function adapted from Hadsell et al. 2006.
    (http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf)

    The contrastive loss is modified so that it takes a certainty that labels
    are correct into account. This helps the neural network to overcome
    incorrectly labeled instances.

    Parameters
    ----------
    y_true : tf.Tensor
        The true class labels.
    y_pred : tf.Tensor
        The predicted embedded Euclidean distances.

    Returns
    -------
    The contrastive loss between the true and predicted class labels.
    """
    ramp_square = K.square(K.minimum(y_pred, config.margin))
    margin_square = K.square(K.maximum(config.margin - y_pred, 0))
    y_true = tf.cast(y_true, tf.float32)
    return K.mean(y_true * config.loss_label_certainty * ramp_square +
                  (1 - y_true * config.loss_label_certainty) * margin_square)


class Embedder:
    """
    A spectrum embedder formed by a Siamese neural network.

    Inspired by the Keras MNIST Siamese network example:
    https://keras.io/examples/mnist_siamese/
    """

    def __init__(self, num_precursor_features: int, num_fragment_features: int,
                 num_ref_spectra_features: int, lr: float,
                 filename: str = 'gleams.hdf5'):
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

        self.siamese_model = None
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            self.num_gpu = strategy.num_replicas_in_sync
            logger.debug('Running the embedder model on %d GPU(s)',
                         self.num_gpu)

    def _get_embedder_model(self) -> Model:
        """
        Get the base embedder model (i.e. a single arm of the Siamese model).

        Returns
        -------
        Model
            The embedder model.
        """
        if self.siamese_model is None:
            raise ValueError('The embedder model has not been constructed yet')
        else:
            return self.siamese_model.get_layer('embedder')

    def save(self) -> None:
        """
        Save the embedder model's weights.
        """
        if self.siamese_model is None:
            raise ValueError('The embedder model has not been constructed yet')
        else:
            self._get_embedder_model().save(self.filename)

    def load(self) -> None:
        """
        Load a previously trained Embedder model.
        """
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            custom_loss = {'contrastive_loss': contrastive_loss}
            with keras.utils.custom_object_scope(custom_loss):
                self.siamese_model = keras.models.load_model(self.filename)

    def _build_embedder_model(self) -> Model:
        """
        Construct the embedder model (i.e. a single arm of the Siamese model).

        The embedder model consists of the following elements:
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
            The embedder model that takes as input the features specified.
        """
        # Precursor features are processed through two dense layers.
        precursor_input = Input((self.num_precursor_features,),
                                name='input_precursor')
        precursor_dense32 = (Dense(32, activation='selu',
                                   kernel_initializer='lecun_normal',
                                   name='precursor_dense_32')
                             (precursor_input))
        precursor_dense5 = (Dense(5, activation='selu',
                                  kernel_initializer='lecun_normal',
                                  name='precursor_dense_5')
                            (precursor_dense32))

        filters = 30
        kernel_size = 3
        strides = 1
        pool_size = 1
        pool_strides = 2
        # Fragment features are processed through a single convolutional and
        # max pooling layer.
        fragment_input = Input((self.num_fragment_features,),
                               name='input_fragment')
        fragment_input_reshape =\
            Reshape((self.num_fragment_features, 1),
                    name='fragment_input_reshape')(fragment_input)
        fragment_layer = fragment_input_reshape
        # Block 1.
        fragment_layer = Conv1D(
            filters, kernel_size, strides=strides, activation='selu',
            name='fragment_block_1_conv_1')(fragment_layer)
        fragment_layer = Conv1D(
            filters, kernel_size, strides=strides, activation='selu',
            name='fragment_block_1_conv_2')(fragment_layer)
        fragment_layer = MaxPooling1D(
            pool_size, pool_strides,
            name='fragment_block_1_pool')(fragment_layer)
        # Block 2.
        fragment_layer = Conv1D(
            filters * 2, kernel_size, strides=strides, activation='selu',
            name='fragment_block_2_conv_1')(fragment_layer)
        fragment_layer = Conv1D(
            filters * 2, kernel_size, strides=strides, activation='selu',
            name='fragment_block_2_conv_2')(fragment_layer)
        fragment_layer = MaxPooling1D(
            pool_size, pool_strides,
            name='fragment_block_2_pool')(fragment_layer)
        # Block 3.
        fragment_layer = Conv1D(
            filters * 4, kernel_size, strides=strides, activation='selu',
            name='fragment_block_3_conv_1')(fragment_layer)
        fragment_layer = Conv1D(
            filters * 4, kernel_size, strides=strides, activation='selu',
            name='fragment_block_3_conv_2')(fragment_layer)
        fragment_layer = Conv1D(
            filters * 4, kernel_size, strides=strides, activation='selu',
            name='fragment_block_3_conv_3')(fragment_layer)
        fragment_layer = MaxPooling1D(
            pool_size, pool_strides,
            name='fragment_block_3_pool')(fragment_layer)
        # Block 4.
        fragment_layer = Conv1D(
            filters * 8, kernel_size, strides=strides, activation='selu',
            name='fragment_block_4_conv_1')(fragment_layer)
        fragment_layer = Conv1D(
            filters * 8, kernel_size, strides=strides, activation='selu',
            name='fragment_block_4_conv_2')(fragment_layer)
        fragment_layer = Conv1D(
            filters * 8, kernel_size, strides=strides, activation='selu',
            name='fragment_block_4_conv_3')(fragment_layer)
        fragment_layer = MaxPooling1D(
            pool_size, pool_strides,
            name='fragment_block_4_pool')(fragment_layer)
        # Block 5.
        fragment_layer = Conv1D(
            filters * 8, kernel_size, strides=strides, activation='selu',
            name='fragment_block_5_conv_1')(fragment_layer)
        fragment_layer = Conv1D(
            filters * 8, kernel_size, strides=strides, activation='selu',
            name='fragment_block_5_conv_2')(fragment_layer)
        fragment_layer = Conv1D(
            filters * 8, kernel_size, strides=strides, activation='selu',
            name='fragment_block_5_conv_3')(fragment_layer)
        fragment_layer = MaxPooling1D(
            pool_size, pool_strides,
            name='fragment_block_5_pool')(fragment_layer)
        fragment_output = Flatten(name='fragment_flatten')(fragment_layer)

        # Reference spectra features are processed through a single
        # convolutional and max pooling layer.
        ref_spectra_input = Input((self.num_ref_spectra_features,),
                                  name='input_ref_spectra')
        ref_spectra_dense750 = (Dense(750, activation='selu',
                                      kernel_initializer='lecun_normal',
                                      name='ref_spectra_dense_750')
                                (ref_spectra_input))
        ref_spectra_output = (Dense(250, activation='selu',
                                    kernel_initializer='lecun_normal',
                                    name='ref_spectra_output')
                              (ref_spectra_dense750))

        # Combine all outputs and add a final dense layer.
        output_layer = (Dense(config.embedding_size, activation='selu',
                              kernel_initializer='lecun_normal',
                              activity_regularizer='l2', name='output')
                        (concatenate([precursor_dense5, fragment_output,
                                      ref_spectra_output])))

        return Model(inputs=[precursor_input, fragment_input,
                             ref_spectra_input],
                     outputs=[output_layer], name='embedder')

    def _build_siamese_model(self) -> Model:
        """
        Construct the Siamese model.

        The Siamese model consists of two instances of the embedder model whose
        weights are tied.

        Returns
        -------
        Model
            The Siamese model.
        """
        embedder_model = self._build_embedder_model()
        input_left = [Input((self.num_precursor_features,),
                            name='input_precursor_left'),
                      Input((self.num_fragment_features,),
                            name='input_fragment_left'),
                      Input((self.num_ref_spectra_features,),
                            name='input_ref_spectra_left')]
        input_right = [Input((self.num_precursor_features,),
                             name='input_precursor_right'),
                       Input((self.num_fragment_features,),
                             name='input_fragment_right'),
                       Input((self.num_ref_spectra_features,),
                             name='input_ref_spectra_right')]
        output_left = embedder_model(input_left)
        output_right = embedder_model(input_right)

        # Euclidean distance between two embeddings.
        distance = (Lambda(euclidean_distance, eucl_dist_output_shape,
                           name='embedding_euclidean_distance')
                    ([output_left, output_right]))

        return Model(inputs=[*input_left, *input_right], outputs=distance,
                     name='siamese_model')

    def build(self) -> None:
        """
        Build the Siamese model and compile it to optimize the contrastive loss
        using Adam.

        Both arms of the Siamese network will use the same embedder model, i.e.
        the weights are tied between both arms.

        The model will be parallelized over all available GPUs if applicable.
        """
        # Both arms of the Siamese network use the same model,
        # i.e. the weights are tied.
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            self.siamese_model = self._build_siamese_model()
            # Train using Adam to optimize the contrastive loss.
            self.siamese_model.compile(RectifiedAdam(self.lr),
                                       contrastive_loss)

    def train(self, train_generator: data_generator.PairSequence,
              steps_per_epoch: int = None, num_epochs: int = 1,
              validators: List[data_generator.PairSequence] = None) -> None:
        """
        Train the Siamese model.

        Parameters
        ----------
        train_generator : data_generator.PairSequence
            The training data generator.
        steps_per_epoch : int
             Total number of in each epoch. Useful to record the validation
             loss at specific intervals.
        num_epochs : int
            The number of epochs for which training occurs.
        validators : List[data_generator.PairSequence]
            The validation data generators.
        """
        if self.siamese_model is None:
            raise ValueError("The Siamese model hasn't been constructed yet")

        filename, ext = os.path.splitext(self.filename)
        filename_log = f'{filename}.log'
        callbacks = [ModelCheckpoint(filename + '.epoch{epoch:03d}' + ext),
                     ValidationCallback(validators, filename_log),
                     CSVLogger(filename_log)]
        self.siamese_model.fit(
            train_generator, steps_per_epoch=steps_per_epoch,
            epochs=num_epochs, callbacks=callbacks)

    def embed(self, encodings_generator: data_generator.EncodingsSequence)\
            -> np.ndarray:
        """
        Transform samples using the embedder model.

        Parameters
        ----------
        encodings_generator: data_generator.EncodingsSequence
            A generator that gives the input samples as batches of a list of
            length three representing the precursor features, fragment
            features, and reference spectra features.

        Returns
        -------
        np.ndarray
            The embeddings of the given samples.
        """
        return self._get_embedder_model().predict(encodings_generator)


class ValidationCallback(Callback):
    """
    Track the loss on validation sets per charge at the end of each epoch.
    """

    def __init__(self, pair_generators: List[data_generator.PairSequence],
                 log_filename: str = None):
        super().__init__()

        self.pair_generators = pair_generators
        self.log_filename = log_filename

    def on_epoch_end(self, epoch, logs=None):
        for charge, pair_generator in zip(
                np.arange(config.charges[0], config.charges[1] + 1),
                self.pair_generators):
            logs[f'val_loss_{charge}'] = self.model.evaluate(pair_generator,
                                                             verbose=0)
