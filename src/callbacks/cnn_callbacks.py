import tensorflow as tf


def cnn_callbacks(model, config, experiment):
    histogram_callback = tf.keras.callbacks.LambdaCallback(
        on_epoch_end=lambda batch,
        logs: experiment.log_histogram_3d(
            values=model.layers[0].get_weights()[-1],
            name='conv1d_1 weights'
        )
    )
    return [histogram_callback]
