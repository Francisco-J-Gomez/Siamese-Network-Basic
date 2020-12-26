import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
import time

NUM_OUTPUTS = 11


def format_output(data):
    is_red = data.pop('is_red')
    is_red = np.array(is_red)
    quality = data.pop('quality')
    quality = np.array(quality)
    return quality, is_red


def norm(x):
    return (x - train_stats['mean']) / train_stats['std']


def base_model(inputs):
    """
    Define the base model
    :param inputs: keras input
    :return: base model
    """
    x = Dense(256, activation="relu")(inputs)
    x = Dropout(0.2)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.2)(x)
    return x


def final_model(inputs):
    """
    Siamese model created from base
    :param inputs: input keras
    :return: siamese model
    """

    # get the base model
    x = base_model(inputs)

    wine_quality = Dense(units='1', name='wine_quality')(x)  # connect the output Dense layer for regression (no activation)
    wine_type = Dense(units='1', activation="sigmoid", name='wine_type')(x) # connect the output Dense layer for classification

    # define the model using the input and output layers
    model = Model(inputs=inputs, outputs=[wine_quality, wine_type])

    return model


def plot_metrics(metric_name, title):
    ylim = np.max(history.history[metric_name])
    plt.clf()
    plt.title(title)
    plt.ylim(0, ylim)
    plt.plot(history.history[metric_name], color='blue', label=metric_name)
    plt.plot(history.history['val_' + metric_name], color='green', label='val_' + metric_name)
    plt.show()


def plot_confusion_matrix(y_true, y_pred, title='', labels=[0,1]):
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="black" if cm[i, j] > thresh else "white")
    plt.show()


if __name__ == "__main__":
    start = time.time()
    # Preprocess dataset
    url_white = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'
    white_df = pd.read_csv(url_white, sep=";")
    white_df["is_red"] = 0  # fill the `is_red` column with zeros.
    white_df = white_df.drop_duplicates(keep='first')  # keep only the first of duplicate items

    # URL of the red wine dataset
    url_red = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    # load the dataset from the URL
    red_df = pd.read_csv(url_red, sep=";")
    red_df["is_red"] = 1  # fill the `is_red` column with ones.
    red_df = red_df.drop_duplicates(keep='first')  # keep only the first of duplicate items

    # concatenate and shuffle dataframes
    df = pd.concat([red_df, white_df], ignore_index=True)
    df = df.iloc[np.random.permutation(len(df))]

    # Plot the quality of the wines
    df['quality'].hist(bins=20)

    # We can see from the plot  that the dataset is imbalanced.
    # We drop the observations with quality 3.4. 8 and 9
    df = df[(df['quality'] > 4) & (df['quality'] < 8)]
    df = df.reset_index(drop=True)  # reset index

    # Split the dataset into training, test and validation .
    train, test = train_test_split(df, test_size=0.2, random_state=1)
    train, val = train_test_split(train, test_size=0.2, random_state=1)

    # Explore the training data
    train_stats = train.describe()
    train_stats.pop('is_red')
    train_stats.pop('quality')
    train_stats = train_stats.transpose()
    print(train_stats)

    # Get the output
    train_Y = format_output(train)
    val_Y = format_output(val)
    test_Y = format_output(test)

    # Normalize the data
    norm_train_X = norm(train)
    norm_val_X = norm(val)
    norm_test_X = norm(test)

    # Create and compile the model
    inputs = tf.keras.layers.Input(shape=(NUM_OUTPUTS,))
    rms = tf.keras.optimizers.RMSprop(lr=0.0001)
    model = final_model(inputs)

    model.compile(optimizer=rms,
                  loss={'wine_type': 'binary_crossentropy',
                        'wine_quality': "mean_squared_error"
                        },
                  metrics={'wine_type': 'accuracy',
                           'wine_quality': tf.keras.metrics.RootMeanSquaredError()
                           }
                  )
    model.summary()

    # Callbacks
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=5, min_lr=0.001)

    # Train the model
    history = model.fit(norm_train_X, train_Y, batch_size=128, steps_per_epoch=27,
                        epochs=180, validation_data=(norm_val_X, val_Y),
                        callbacks=[reduce_lr])

    # Show training metrics
    loss, wine_quality_loss, wine_type_loss, wine_quality_rmse, wine_type_accuracy = model.evaluate(x=norm_val_X, y=val_Y)

    print(f'loss: {loss}')
    print(f'wine_quality_loss: {wine_quality_loss}')
    print(f'wine_type_loss: {wine_type_loss}')
    print(f'wine_quality_rmse: {wine_quality_rmse}')
    print(f'wine_type_accuracy: {wine_type_accuracy}')

    # Analyzing the model
    predictions = model.predict(norm_test_X)
    quality_pred = predictions[0]
    type_pred = predictions[1]

    print("It took {} s to train the model".format(time.time()-start))

    # Plot metrics
    plot_metrics('wine_quality_root_mean_squared_error', 'RMSE')
    plot_metrics('wine_type_loss', 'Wine Type Loss')

    # Plot confusion matrix for wine type
    plot_confusion_matrix(test_Y[1], np.round(type_pred), title='Wine Type', labels=[0, 1])