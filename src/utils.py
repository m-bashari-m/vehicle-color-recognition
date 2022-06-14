import tensorflow as tf
from tensorflow import keras
import os
import tensorflow_hub as hub
import numpy as np
from tqdm.notebook import tqdm_notebook as tqdn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

class ModelCreator():

    def __init__(self, hub_module_url, model_name):
        self.bit_module = hub.KerasLayer(hub_module_url)
        self.model_name = model_name

        self.metrics = [
            keras.metrics.AUC(name='auc', curve='PR', num_thresholds=100),
            'accuracy'
        ]


    def make_model(self,
                    n_classes=16,
                    img_size=(256,256),
                    decay_step=533,
                    initial_lr=1e-2):

        # Define the model
        model = tf.keras.Sequential([
            keras.Input(shape=img_size+(3,)),
            self.bit_module,
            keras.layers.Dense(n_classes, activation='softmax')
        ])
        
        model._name = self.model_name
        
        # Compiling
        loss_fn = keras.losses.CategoricalCrossentropy()
        lr_schedule =tf.keras.optimizers.schedules.ExponentialDecay(initial_lr, decay_step, .9)
        model.compile(loss=loss_fn,
                    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
                    metrics=self.metrics)

        model.summary()
        
        return model


    # Returns EarlyStopping and ModelCheckpoint callbacks
    def get_callbacks(self):
        early_stopping = keras.callbacks.EarlyStopping(
                                                    monitor='auc', 
                                                    verbose=1,
                                                    patience=3,
                                                    restore_best_weights=True,
                                                    mode='max')

        check_point_path = os.path.join('./logs/checkpoints', self.model_name+"-{epoch:02d}.h5")
        check_point = keras.callbacks.ModelCheckpoint(
                                                    filepath=check_point_path,
                                                    monitor='auc',
                                                    save_best_only=True,
                                                    mode='max')
        
        callbacks = [early_stopping, check_point]
        return callbacks



class ErrorAnalyzer():
    
    def __init__(self, model, ds, classes, model_name):
        self.model = model
        self.ds = ds
        self.file_paths = np.array(ds.file_paths)
        self.classes = classes
        self.model_name = model_name
        
        # Path to store images, csv files and confusion matrix array
        os.makedirs('./logs/statistic', exist_ok=True)
        self.log_file = os.path.join('./logs/statistic')

        # labels and predictions from validation dataset
        self.lbls = tf.Variable([], dtype=tf.int16)
        self.preds = tf.Variable([], dtype=tf.int16)
        
        self.conf_mat = self.__calc_confusion_mat()


    # Calculates and saves confusion matrix in log file
    def __calc_confusion_mat(self):
        print("Making confusion matrix:")
        for img_batch, lbl_batch in tqdn(self.ds):
            pred = tf.argmax(self.model(img_batch), axis=-1)
            self.preds = tf.concat([self.preds, tf.cast(pred, tf.int16)], axis=0)
            self.lbls = tf.concat([self.lbls, tf.cast(tf.argmax(lbl_batch, axis=-1), tf.int16)], axis=0)

        conf_mat = tf.math.confusion_matrix(self.lbls, self.preds).numpy()
        
        print('Confusion matrix is saved')
        with open(os.path.join(self.log_file, self.model_name+'-conf-mat.npy'), 'wb') as f:
            np.save(f, conf_mat)

        return conf_mat
            

    # Calcualtes the percentage of confusion matrix and plot and save
    def plot_confusion_mat(self):
        conf_mat = self.conf_mat.astype('float') / self.conf_mat.sum(axis=1)[:, tf.newaxis] * 100

        fig, ax = plt.subplots(figsize=(10,10))
        sns.heatmap(conf_mat, annot=True, fmt='.1f', xticklabels=self.classes, yticklabels=self.classes)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')

        images_dir = os.path.join(self.log_file, 'images')
        os.makedirs(images_dir, exist_ok=True)
        dest = os.path.join(images_dir, self.model_name+'.jpg')
        plt.savefig(dest, dpi=200)
        
        plt.show(block=False)


    # Saves some metrics and save them in log file
    # Metrics are:
    # Accuracy, Precision, Recall, True/False Positive True/False Negative
    def evaluate_model(self):
        print("Calculateing error types...")
        conf_stats = ConfusionStatistic(self.conf_mat, self.classes)
        
        print("Writing in log file...")
        with open(os.path.join(self.log_file, self.model_name+'.csv'), 'a') as f:
            f.write('color,accuracy,precision,recall,TP,TN,FP,FN\n')

            precision_sum, recall_sum = 0, 0
            for i, class_ in enumerate(self.classes):
                precision, recall = self.get_precision_recall(i)
                precision_sum += precision
                recall_sum += recall
                f.write("{},{},{},{},{},{},{},{}\n".format(class_,
                                                        round(conf_stats.accuracy, 3),
                                                        precision,
                                                        recall,
                                                        conf_stats.tp[class_],
                                                        conf_stats.tn[class_],
                                                        conf_stats.fp[class_],
                                                        conf_stats.fn[class_]))
                                                        
        print("\033[1;32mAll done. Check log file => {}".format(self.model_name+'.csv'))

        print('Accuracy: %{}'.format(round(conf_stats.accuracy, 4)* 100))
        print('Precision mean: {}'.format(round(precision_sum / len(self.classes), 4)))
        print('Recall mean: {}'.format(round(recall_sum / len(self.classes), 4)))
    
            
    def get_precision_recall(self, class_num):
        precision = self.conf_mat[class_num, class_num] / self.conf_mat.sum(axis=0)[class_num]
        recall = self.conf_mat[class_num, class_num] / self.conf_mat.sum(axis=1)[class_num]
        return round(precision, 3), round(recall, 3)

    
    def plot_missclassified(self, base_class, n_cols=4, n_rows=4):
        row, col = 0, 0
        plt.figure(figsize=(17,17))
        for class_ in self.classes:
            if col == n_cols:
                col = 0
                row += 1

            frame = self.__a_predicted_as_b(base_class, class_)
            plt.subplot(n_rows, n_cols, row*n_cols + col + 1)
            plt.title(f'{base_class} predicted as {class_}')
            plt.imshow(frame)
            plt.axis('off')

            col += 1

    def __make_frame(self, paths, size=200, n_cols=3, n_rows=3):
        frame = np.zeros([n_rows*size, n_cols*size, 3])
        row, col = 0, 0
        for path in paths:
            image = tf.image.decode_image(tf.io.read_file(path), expand_animations=False)
            image = tf.image.resize(image, (size, size))

            if col == n_cols:
                row += 1
                col = 0

            try:
                frame[row*size:(row+1)*size, col*size:(col+1)*size, :] = tf.cast(image, tf.float32) / 255.0
            except Exception as ex:
                return frame

            col +=1

        return frame

    def __a_predicted_as_b(self, class_a, class_b, images_in_frame=9):
        class_a_num = self.classes.index(class_a)
        class_b_num = self.classes.index(class_b)
        target_paths = self.file_paths[(self.lbls == class_a_num) & (self.preds == class_b_num)]
        if len(target_paths) < images_in_frame:
            frame = self.__make_frame(target_paths)
        else:
            frame = self.__make_frame(target_paths[:images_in_frame])

        return frame

    
            
# Calculate some metrics for using in ErrorAnalyzer
# Metrics are:
# Accuracy, True/False Positive True/False Negative
class ConfusionStatistic():
    def __init__(self, confusion_mat, classes):
        self.confusion_mat = confusion_mat
        self.classes = classes
        self._tp = self.__true_positive()
        self._fp = self.__false_positive()
        self._fn = self.__false_negative()
        self._tn = self.__true_negative()
        self._accuracy= self.__accuracy()

    @property
    def tp(self):
        return self._tp

    @property
    def fp(self):
        return self._fp

    @property
    def fn(self):
        return self._fn

    @property
    def tn(self):
        return self._tn

    @property
    def accuracy(self):
        return self._accuracy

    def __true_positive(self):
        result = dict().fromkeys(self.classes)
        for i, class_ in enumerate(self.classes):
            result[class_] = self.confusion_mat[i, i]

        return result

    def __false_positive(self):
        result = dict().fromkeys(self.classes)
        for i, class_ in enumerate(self.classes):
            result[class_] = np.sum(self.confusion_mat[:, i]) - self.tp[class_]

        return result

    def __false_negative(self):
        result = dict().fromkeys(self.classes)
        for i, class_ in enumerate(self.classes):
            result[class_] = np.sum(self.confusion_mat[i, :]) - self.tp[class_]

        return result

    def __true_negative(self):
        result = dict().fromkeys(self.classes)
        for i, class_ in enumerate(self.classes):
            result[class_] = np.sum(self.confusion_mat) - self.fp[class_] - self.fn[class_] + self.tp[class_]

        return result

    def __accuracy(self):
        return np.sum(np.diagonal(self.confusion_mat)) / np.sum(self.confusion_mat)
         

# Returns train and validation dataset

def get_train_val_ds(train_dir, val_dir,batch_size=32, img_size=(256,256), seed=42, shuffle=True):
    train_ds = keras.utils.image_dataset_from_directory(train_dir,
                                                    image_size=img_size,
                                                    shuffle=shuffle,
                                                    label_mode='categorical',
                                                    batch_size=batch_size,
                                                    seed=seed)

    val_ds = keras.utils.image_dataset_from_directory(val_dir,
                                                  image_size=img_size,
                                                  shuffle=False,
                                                  label_mode='categorical',
                                                  batch_size=batch_size,
                                                  seed=seed)

    return train_ds, val_ds


# Download classes information and alculate class weights 
# Returns classe names in a list and class weights in a dict with 
#   kyes from 0 to 15
def get_class_weight():
    url = 'https://raw.githubusercontent.com/m-bashari-m/vehicle-color-recognition/main/logs/dataset-info.csv'
    df = pd.read_csv(url, index_col=0)

    n_train = np.sum(df['train'])
    class_weight = {key:value for
                key, value in zip(df['color'].index, (n_train/df['train']).round(2))}

    return df['color'], class_weight


def get_model(model_name):
    base_path = './drive/MyDrive/checkpoints'
    path = os.path.join(base_path, model_name)
    model = tf.keras.models.load_model(path, custom_objects={'KerasLayer': hub.KerasLayer})
    return model
