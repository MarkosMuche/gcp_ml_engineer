import tensorflow as tf
import tensorflow_datasets as tfds
import argparse
import hypertune

NUM_EPOCHS = 10
BATCH_SIZE = 64

def get_args():
  '''Parses args. Must include all hyperparameters you want to tune.'''

  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--learning_rate',
      required=True,
      type=float,
      help='learning rate')
  parser.add_argument(
      '--momentum',
      required=True,
      type=float,
      help='SGD momentum value')
  parser.add_argument(
      '--num_units',
      required=True,
      type=int,
      help='number of units in last hidden layer')
  args = parser.parse_args()
  return args


def preprocess_data(image, label):
  '''Resizes and scales images.'''

  image = tf.image.resize(image, (150,150))
  return tf.cast(image, tf.float32) / 255., label


def create_dataset(batch_size):
  '''Loads Horses Or Humans dataset and preprocesses data.'''

  data, info = tfds.load(name='horses_or_humans', as_supervised=True, with_info=True)

  # Create train dataset
  train_data = data['train'].map(preprocess_data)
  train_data  = train_data.shuffle(1000)
  train_data  = train_data.batch(batch_size)

  # Create validation dataset
  validation_data = data['test'].map(preprocess_data)
  validation_data  = validation_data.batch(batch_size)

  return train_data, validation_data


def create_model(num_units, learning_rate, momentum):
  '''Defines and compiles model.'''

  inputs = tf.keras.Input(shape=(150, 150, 3))
  x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu')(inputs)
  x = tf.keras.layers.MaxPooling2D((2, 2))(x)
  x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(x)
  x = tf.keras.layers.MaxPooling2D((2, 2))(x)
  x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
  x = tf.keras.layers.MaxPooling2D((2, 2))(x)
  x = tf.keras.layers.Flatten()(x)
  x = tf.keras.layers.Dense(num_units, activation='relu')(x)
  outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
  model = tf.keras.Model(inputs, outputs)
  model.compile(
      loss='binary_crossentropy',
      optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum),
      metrics=['accuracy'])
  return model


def main():
  args = get_args()

  # Create distribution strategy
  strategy = tf.distribute.MirroredStrategy()

  # Get data
  GLOBAL_BATCH_SIZE = BATCH_SIZE * strategy.num_replicas_in_sync
  train_data, validation_data = create_dataset(GLOBAL_BATCH_SIZE)

  # Wrap variable creation within strategy scope
  with strategy.scope():
    model = create_model(args.num_units, args.learning_rate, args.momentum)

  # Train model
  history = model.fit(train_data, epochs=NUM_EPOCHS, validation_data=validation_data)

  # Define metric
  hp_metric = history.history['val_accuracy'][-1]

  hpt = hypertune.HyperTune()
  hpt.report_hyperparameter_tuning_metric(
      hyperparameter_metric_tag='accuracy',
      metric_value=hp_metric,
      global_step=NUM_EPOCHS)


if __name__ == "__main__":
    main()