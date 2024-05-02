
from typing import NamedTuple, Dict, Text, Any
from keras_tuner.engine import base_tuner
from tensorflow.keras import layers
import tensorflow as tf
import tensorflow_transform as tft
import keras_tuner as kt

LABEL_KEY = "is_toxic"
FEATURE_KEY = "tweet"


def transformed_name(key):
    """Renaming transformed features"""
    return key + "_xf"


def gzip_reader_fn(filenames):
    """Loads compressed data"""
    return tf.data.TFRecordDataset(filenames, compression_type="GZIP")


def input_fn(
    file_pattern, tf_transform_output, num_epochs, batch_size=64
) -> tf.data.Dataset:
    """Get post_tranform feature & create batches of data"""

    # Get post_transform feature spec
    transform_feature_spec = tf_transform_output.transformed_feature_spec().copy()

    # Create batches of data
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=transformed_name(LABEL_KEY),
    )

    return dataset


# Vocabulary size and number of words in a sequence
VOCAB_SIZE = 10000
SEQUENCE_LENGTH = 50

vectorize_layer = layers.TextVectorization(
    standardize="lower_and_strip_punctuation",
    max_tokens=VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=SEQUENCE_LENGTH,
)


def model_builder(hp):
    """Builds a keras model to be tuned"""
    embedding_dim = hp.Int("embedding_dim", min_value=16, max_value=64, step=16)
    lstm_units = hp.Int("lstm_units", min_value=32, max_value=64, step=16)
    dense_units = hp.Int("dense_units", min_value=16, max_value=64, step=16)
    dropout_rate = hp.Float("dropout_rate", min_value=0.2, max_value=0.4, step=0.1)
    learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3])

    inputs = tf.keras.Input(
        shape=(1,), name=transformed_name(FEATURE_KEY), dtype=tf.string
    )
    reshaped_narrative = tf.reshape(inputs, [-1])
    x = vectorize_layer(reshaped_narrative)
    x = layers.Embedding(VOCAB_SIZE, embedding_dim)(x)
    x = layers.Bidirectional(layers.LSTM(lstm_units))(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        metrics=[tf.keras.metrics.BinaryAccuracy()],
    )

    return model


def tuner_fn(fn_args):
    """Build the tuner using the KerasTuner API.
    Args:
      fn_args: Holds args used to tune models as name/value pairs.

    Returns:
      A namedtuple contains the following:
        - tuner: A BaseTuner that will be used for tuning.
        - fit_kwargs: Args to pass to tuner"s run_trial function for fitting the
                      model , e.g., the training and validation dataset. Required
                      args depend on the above tuner"s implementation.
    """
    # Load the transform output
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    train_set = input_fn(fn_args.train_files[0], tf_transform_output, num_epochs=10)
    val_set = input_fn(fn_args.eval_files[0], tf_transform_output, num_epochs=10)

    vectorize_layer.adapt(
        [
            j[0].numpy()[0]
            for j in [i[0][transformed_name(FEATURE_KEY)] for i in list(train_set)]
        ]
    )

    # Define the tuner
    tuner = kt.Hyperband(
        lambda hp: model_builder(hp),
        objective="val_binary_accuracy",
        max_epochs=5,
        factor=3,
        directory=fn_args.working_dir,
        project_name="toxic_tweets_tuner",
    )

    # Define early stopping callback
    stop_early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=1)

    TunerFnResult = NamedTuple(
        "TunerFnResult",
        [("tuner", base_tuner.BaseTuner), ("fit_kwargs", Dict[Text, Any])],
    )

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            "callbacks": [stop_early],
            "x": train_set,
            "validation_data": val_set,
            "steps_per_epoch": fn_args.train_steps,
            "validation_steps": fn_args.eval_steps,
        },
    )
