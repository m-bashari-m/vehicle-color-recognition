from abc import abstractmethod
from pydoc import classname
import tensorflow as tf
from tensorflow import keras
import os
import tensorflow_hub as hub
import numpy as np
from tqdm.notebook import tqdm_notebook as tqdn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import keras.backend as K

class ModelCreator():
    def __init__(self, hub_module_url, model_name):
        self.bit_module = hub.KerasLayer(hub_module_url)
        self.model_name = model_name
        self.metrics = [
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name='auc', curve='PR', multi_label=True),
            keras.metrics.Accuracy()
        ]

    def make_model(self,
                    n_classes=16,
                    img_size=(512,512),
                    n_channels=3,
                    init_lr=1e-2,
                    decay_steps=200):


        model = tf.keras.Sequential([
            keras.Input(shape=img_size+(n_channels,)),
            self.bit_module,
            keras.layers.Dense(600),
            keras.layers.Dropout(.3),
            keras.layers.Dense(n_classes, activation='softmax')
        ])
        
        model._name = self.model_name
        
        loss_fn = keras.losses.CategoricalCrossentropy()
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                                                            initial_learning_rate=init_lr,
                                                            decay_steps=decay_steps,
                                                            decay_rate=.9)

        model.compile(loss=loss_fn,
                    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
                    metrics=self.metrics)
        
        return model

    def get_callbacks(self):
        early_stopping = keras.callbacks.EarlyStopping(
                                                    monitor='auc', 
                                                    verbose=1,
                                                    patience=5,
                                                    mode='max')

        check_point_path = os.path.join('./logs/checkpoints', self.model_name+"-{epoch:02d}.h5")
        check_point = keras.callbacks.ModelCheckpoint(
                                                    filepath=check_point_path,
                                                    monitor='auc',
                                                    mode='max')
        
        callbacks = [early_stopping, check_point]
        return callbacks



class ErrorAnalyzer():
    def __init__(self, model, ds, classes, model_name):
        self.model = model
        self.ds = ds
        self.classes = classes
        self.model_name = model_name
        self.log_file = os.path.join('./logs/statistic')
        self.conf_mat = self._calc_confusion_mat()

    def _calc_confusion_mat(self):
        self.lbls = np.array([])
        self.preds = np.array([])

        print("Making confusion matrix:")
        for img_batch, lbl_batch in tqdn(self.ds):
            pred = tf.argmax(self.model(img_batch), axis=-1)
            self.preds = np.concatenate([self.preds, pred.numpy()])
            self.lbls = np.concatenate([self.lbls, tf.argmax(lbl_batch, axis=-1).numpy()]) 
        
        conf_mat = tf.math.confusion_matrix(self.lbls, self.preds).numpy()

        return conf_mat

    
    def plot_confusion_mat(self):
        conf_mat = self.conf_mat.astype('float') / self.conf_mat.sum(axis=1)[:, tf.newaxis]

        fig, ax = plt.subplots(figsize=(10,10))
        sns.heatmap(conf_mat, annot=True, fmt='.2f', xticklabels=self.classes, yticklabels=self.classes)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')

        images_dir = os.path.join(self.log_file, 'images')
        os.makedirs(images_dir, exist_ok=True)
        dest = os.path.join(images_dir, self.model_name+'.jpg')
        plt.savefig(dest, dpi=150)
        plt.show(block=False)


    def evaluate_model(self):
        print("Calculating error type...")
        conf_stat = ConfusionStatistic(self.conf_mat, self.classes)
        
        csv_output = os.path.join(self.log_file, self.model_name+'.csv')
        print("Writing in log file...")
        with open(csv_output, 'a') as f:
            f.write('color,precision,recall,TP,TN,FP,FN\n')
            for i, class_ in enumerate(self.classes):
                precision, recall = self.get_precision_recall(i)
                f.write("{},{},{},{},{},{},{}\n".format(class_,
                                                        precision,
                                                        recall,
                                                        conf_stat.tp[class_],
                                                        conf_stat.tn[class_],
                                                        conf_stat.fp[class_],
                                                        conf_stat.fn[class_]))
                                                        
        print("All done. Check log file {}.".format(self.model_name+'.csv'))

            
    def get_precision_recall(self, class_num):
        precision = self.conf_mat[class_num, class_num] / self.conf_mat.sum(axis=0)[class_num]
        recall = self.conf_mat[class_num, class_num] / self.conf_mat.sum(axis=1)[class_num]
        return precision, recall


class ConfusionStatistic():
    def __init__(self, confusion_mat, classes):
        self.confusion_mat = confusion_mat
        self.classes = classes
        self._tp = self.__true_positive()
        self._fp = self.__false_positive()
        self._fn = self.__false_negative()
        self._tn = self.__true_negative()

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


    
def get_train_val_ds(train_dir, val_dir,batch_size=32, img_size=(512,512), seed=42, shuffle=True):
    train_ds = keras.utils.image_dataset_from_directory(train_dir,
                                                    image_size=img_size,
                                                    shuffle=shuffle,
                                                    label_mode='categorical',
                                                    batch_size=batch_size,
                                                    seed=seed)

    val_ds = keras.utils.image_dataset_from_directory(val_dir,
                                                  image_size=img_size,
                                                  shuffle=shuffle,
                                                  label_mode='categorical',
                                                  batch_size=batch_size,
                                                  seed=seed)

    return train_ds, val_ds



def get_class_weight():
    url = 'https://raw.githubusercontent.com/m-bashari-m/vehicle-color-recognition/main/logs/dataset-info.csv'
    df = pd.read_csv(url, index_col=0)

    n_train = np.sum(df['train'])
    class_weight = {key:value for
                key, value in zip(df['color'].index, (n_train/df['train']).round(2))}

    return df['color'], class_weight



