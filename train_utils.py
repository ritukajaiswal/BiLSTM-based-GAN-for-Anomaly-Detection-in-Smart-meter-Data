from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from tensorflow.keras.optimizers import Adam

from network import *

# Performs training on a given dataset


def train(x_train, batch_size, epochs, generator, discriminator, gan, progress=False):
    # Calculating the number of batches based on the batch size
    batch_count = x_train.shape[0] // batch_size
    if progress:
        pbar = tqdm(total=epochs * batch_count)
    gan_loss = []
    discriminator_loss = []

    for epoch in range(epochs):
        for index in range(batch_count):
            if progress:
                pbar.update(1)
            # Creating a random set of input noise and images
            noise = np.random.normal(0, 1, size=[batch_size, 8])

            generated_data = generator.predict_on_batch(noise)

            # Obtain a batch of normal network packets
            image_batch = x_train[index * batch_size: (index + 1) * batch_size]

            X = np.vstack((generated_data, image_batch))
            y_dis = np.ones(2*batch_size)
            y_dis[:batch_size] = 0

            # Train discriminator
            discriminator.trainable = True
            d_loss = discriminator.train_on_batch(X, y_dis)

            # Train generator
            noise = np.random.uniform(0, 1, size=[batch_size, 8])
            y_gen = np.ones(batch_size)
            discriminator.trainable = False
            g_loss = gan.train_on_batch(noise, y_gen)

            # Record the losses
            discriminator_loss.append(d_loss)
            gan_loss.append(g_loss)

        # print("Epoch %d Batch %d/%d [D loss: %f] [G loss:%f]" % (epoch,index,batch_count, d_loss, g_loss))

    return discriminator_loss, gan_loss, discriminator


# Predictions on the test set
def anomaly_detection(x_test, y_test, batch_size, discriminator):
    nr_batches_test = np.ceil(x_test.shape[0] // batch_size).astype(np.int32)

    results = []

    for t in range(nr_batches_test + 1):
        ran_from = t * batch_size
        ran_to = (t + 1) * batch_size
        image_batch = x_test[ran_from:ran_to]
        tmp_rslt = discriminator.predict(
            x=image_batch, batch_size=128, verbose=0)
        results = np.append(results, tmp_rslt)

    pd.options.display.float_format = '{:20,.7f}'.format
    results_df = pd.concat(
        [pd.DataFrame(results), pd.DataFrame(y_test)], axis=1)
    results_df.columns = ['results', 'y_test']
    # print ('Mean score for normal packets :', results_df.loc[results_df['y_test'] == 0, 'results'].mean() )
    # print ('Mean score for anomalous packets :', results_df.loc[results_df['y_test'] == 1, 'results'].mean())

    # Obtaining the lowest 1% score
    per = np.percentile(results, 1)
    y_pred = results.copy()
    y_pred = np.array(y_pred)

    # Thresholding based on the score
    inds = (y_pred > per)
    inds_comp = (y_pred <= per)
    y_pred[inds] = 0
    y_pred[inds_comp] = 1

    return y_pred, results_df

# Predictions on the test set


def anomaly_detection_test(x_test, batch_size, discriminator):
    nr_batches_test = np.ceil(x_test.shape[0] // batch_size).astype(np.int32)

    results = []

    for t in range(nr_batches_test + 1):
        ran_from = t * batch_size
        ran_to = (t + 1) * batch_size
        image_batch = x_test[ran_from:ran_to]
        tmp_rslt = discriminator.predict(
            x=image_batch, batch_size=128, verbose=0)
        results = np.append(results, tmp_rslt)

    pd.options.display.float_format = '{:20,.7f}'.format

    # Obtaining the lowest 1% score
    per = np.percentile(results, 1)
    y_pred = results.copy()
    y_pred = np.array(y_pred)

    # Thresholding based on the score
    inds = (y_pred > per)
    inds_comp = (y_pred <= per)
    y_pred[inds] = 0
    y_pred[inds_comp] = 1

    return y_pred

# Evaluates the performance of the model


def evaluation(y_test, y_pred, print_results=False):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='binary')
    acc_score = accuracy_score(y_test, y_pred)
    if print_results:
        print('Accuracy Score :', acc_score)
        print('Precision :', precision)
        print('Recall :', recall)
        print('F1 :', f1)
    return acc_score, precision, recall, f1


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    target_names = ['normal', 'anomaly']
    # plt.figure(figsize=(10,10),)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()

    width, height = cm.shape

    for x in range(width):
        for y in range(height):
            plt.annotate(str(cm[x][y]), xy=(y, x),
                         horizontalalignment='center',
                         verticalalignment='center')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_roc_curve(y_test, y_pred):
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred)
    auc_keras = auc(fpr_keras, tpr_keras)
    # plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras,
             label='Keras (area = {:.2f})'.format(auc_keras))

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()

# Runs the entire process, performs training, anomaly detection and plots the relevant results


def run_and_plot(dataset, learning_rate, batch_size, epochs, progress=False):
    x_train, y_train, x_test, y_test = dataset['x_train'], dataset[
        'y_train'], dataset['x_test'], dataset['y_test']
    adam = Adam(learning_rate=learning_rate)
    generator = get_generator(adam)
    discriminator = get_discriminator(adam)
    gan = get_gan_network(discriminator, generator, adam, input_dim=8)
    discriminator_loss, gan_loss, trained_discriminator = train(
        x_train, batch_size, epochs, generator, discriminator, gan, progress)
    y_pred = anomaly_detection(
        x_test, y_test, batch_size, trained_discriminator)
    print(len(y_pred), len(y_test))
    # acc_score, precision, recall, f1 = evaluation(y_test, y_pred)

    plt.figure(figsize=(20, 20))

    plt.subplot(2, 1, 1)
    plt.plot(discriminator_loss, label='Discriminator')
    plt.plot(gan_loss, label='Generator')
    plt.title("Training Losses")
    plt.legend()

    plt.subplot(2, 2, 3)
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm)

    plt.subplot(2, 2, 4)
    plot_roc_curve(y_test, y_pred)

    plt.show()

    return y_pred, trained_discriminator

# Performs only the anomaly detection given a test set a batch size and a discriminator


def test(x_test, batch_size, discriminator):
    y_pred = anomaly_detection_test(x_test, batch_size, discriminator)
    return y_pred
