# Ryan Theurer 6/24/2022

# import libraries
import logging
import numpy as np
import numpy.ma as ma
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
import time
from xgboost import XGBClassifier


'''Preprocess'''


def import_data():
    img_dir = 'images'
    metadata_fpath = 'image_metadata.csv'
    # testing data for models
    gems_plant_id_test = [16324, 27965, 13361, 15570]

    img_metadata = pd.read_csv(metadata_fpath)
    img_metadata['image_fpath'] = img_metadata.apply(
        lambda row, img_dir=img_dir: os.path.join(img_dir, str(row.gems_plant_id), row.image_fn),
        axis='columns')
    plant_training_data = img_metadata[~img_metadata["gems_plant_id"].isin(gems_plant_id_test)].reset_index()
    plant_testing_data = img_metadata[img_metadata["gems_plant_id"].isin(gems_plant_id_test)].reset_index()

    return img_metadata, plant_training_data, plant_testing_data


def image_to_pixels(image):
    # flattens a three-dimensional image into a two-dimensional array of pixels
    return image.reshape((image.shape[0] * image.shape[1], image.shape[2]))


def get_rgb_from_array(row, data, global_feature_flag):
    data['filename'].append(row.image_fn)

    arr_fpath = row.image_fpath
    arr = np.load(arr_fpath)
    arr = np.nan_to_num(arr)

    # outputs the rgb array from the input array with shape of [rows, columns, bands]
    arr_t = np.transpose(arr, axes=(1, 2, 0))
    rgb = arr_t[:, :, [3, 2, 1]]
    shape = rgb.shape
    rgb = MinMaxScaler().fit_transform(image_to_pixels(rgb))
    rgb = np.reshape(rgb, shape)

    # if global feature flag is on we extract a feature to test
    if global_feature_flag:
        rgb = global_feature_extraction(rgb)

    data['data'].append(rgb)
    return data


def get_selected_band_from_array(row, data, band):
    data['filename'].append(row.image_fn)

    arr_fpath = row.image_fpath
    arr = np.load(arr_fpath)
    arr = np.nan_to_num(arr)

    # outputs the selected band array from the input array with shape of [rows, columns, bands]
    arr_t = np.transpose(arr, axes=(1, 2, 0))
    selected_band = arr_t[:, :, [band]]
    shape = selected_band.shape
    selected_band = MinMaxScaler().fit_transform(image_to_pixels(selected_band))
    selected_band = np.reshape(selected_band, shape)
    data['data'].append(selected_band)
    return data


def create_labels(row, data):
    # create labels out of power generation data

    if row.generation > 0:
        data['label'].append(1)
    else:
        data['label'].append(0)
    return data


def create_weights(row, data):
    # create weights

    if float(row.capacity_factor) <= .1 and float(row.capacity_factor) != 0.0:
        data['weights'].append(.5)
    else:
        data['weights'].append(1)
    return data


def create_ndvi_mask(row, data):
    arr_fpath = row.image_fn
    arr = np.load(arr_fpath)
    arr = np.nan_to_num(arr)

    arr_t = np.transpose(arr, axes=(1, 2, 0))
    # Calculate NDVI layer
    NDVI = (arr_t[:, :, [7]] - arr_t[:, :, [3]])/(arr_t[:, :, [7]] + arr_t[:, :, [3]])
    shape = NDVI.shape
    # Normalize and reshape NDVI layer
    NDVI = MinMaxScaler().fit_transform(image_to_pixels(NDVI))
    NDVI = np.reshape(NDVI, shape)
    # outputs the rgb image from the input array with shape of [bands, rows, columns]
    arr_t = np.transpose(arr, axes=(1, 2, 0))
    rgb = arr_t[:, :, [3, 2, 1]]
    shape = rgb.shape
    rgb = MinMaxScaler().fit_transform(image_to_pixels(rgb))
    rgb = np.reshape(rgb, shape)

    # match NDVI dimension to RGB dimension to apply mask
    NDVI = np.broadcast_to(NDVI, (40, 40, 3))

    # apply mask where NDVI is greater than .4 (thick vegetation) and less than 0 (snow, dirt, water)
    ma_rgb = ma.masked_where(NDVI > .4, rgb)
    ma_rgb = ma.masked_where(NDVI < 0, ma_rgb)
    data['data'].append(ma_rgb)
    return data


def global_feature_extraction(rgb):
    feature_matrix = np.zeros((40, 40))

    for i in range(0, rgb.shape[0]):
        for j in range(0, rgb.shape[1]):
            feature_matrix[i][j] = ((float(rgb[i, j, 0]) + float(rgb[i, j, 1]) + float(rgb[i, j, 2])) / 3)
    # calculate grey scale feature
    grey = np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
    return grey


def retrieve_and_store_data(dataset, rgb_flag, band, global_feature_flag):
    # creates dictionary for storing labels, image data, filename, and weights
    data = dict()
    data['label'] = []
    data['filename'] = []
    data['data'] = []
    data['weights'] = []

    for i, row in dataset.iterrows():
        # if the rgb flag is off, the a user selected band is processed instead
        if rgb_flag is True:
            data = get_rgb_from_array(row, data, global_feature_flag)
        else:
            data = get_selected_band_from_array(row, data, band)
        data = create_labels(row, data)
        data = create_weights(row, data)
        # creates NDVI mask for and applies to image
        # data = create_ndvi_mask(row, data)
    return data

# method to put data into its proper shape for modeling
def create_train_test_data(training, testing, global_feature_flag):
    X_training = np.array(training['data'])
    y_training = np.array(training['label'])

    # global feature is already 3 dimensional so it needs a different process step than bands are RGB
    if global_feature_flag:
        X_training = X_training.reshape(X_training.shape[0],
                                        X_training.shape[1] * X_training.shape[2])
    else:
        X_training = X_training.reshape(X_training.shape[0],
                                        X_training.shape[1] * X_training.shape[2] * X_training.shape[3])

    X_testing = np.array(testing['data'])
    y_testing = np.array(testing['label'])
    if global_feature_flag:
        X_testing = X_testing.reshape(X_testing.shape[0], X_testing.shape[1] * X_testing.shape[2])
    else:
        X_testing = X_testing.reshape(X_testing.shape[0], X_testing.shape[1] * X_testing.shape[2] * X_testing.shape[3])

    return X_training, X_testing, y_training, y_testing


'''Data Modeling'''


def xgboost_model(X_train, y_train):
    classifier = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    training_start = time.perf_counter()
    classifier.fit(X_train, y_train)
    training_end = time.perf_counter()
    return classifier, training_start, training_end


def knn_model(X_train, y_train):
    knn = KNeighborsClassifier()
    training_start = time.perf_counter()
    knn.fit(X_train, y_train)
    training_end = time.perf_counter()
    return knn, training_start, training_end


def random_forest_model(X_train, y_train, weights):
    weights = np.array(weights)

    rfc = RandomForestClassifier(n_estimators=10)
    training_start = time.perf_counter()
    rfc.fit(X_train, y_train, sample_weight=weights)
    training_end = time.perf_counter()
    return rfc, training_start, training_end


def svc_model(X_train, y_train):
    svc = SVC()
    training_start = time.perf_counter()
    svc.fit(X_train, y_train)
    training_end = time.perf_counter()
    return svc, training_start, training_end


'''Postprocessing'''


def xgboost_post_processing(classifier, X_test, y_test, X_train, y_train, training_start, training_end):
    prediction_start = time.perf_counter()
    preds = classifier.predict(X_test)
    prediction_end = time.perf_counter()
    logging.info("xgboost Model")
    logging.info(confusion_matrix(y_test, preds))
    tn, fp, fn, tp = confusion_matrix(
        y_test, preds,
        labels=[0, 1]).ravel()
    logging.info("Accuracy:" + str(round(accuracy_score(y_test, preds), 2)))
    precision = (tp/(tp + fp))
    recall = (tp/(tp + fn))
    logging.info("Precision: " + str(round(precision, 2)))
    logging.info("Recall: " + str(round(recall, 2)))
    logging.info("f1_score: " + str(round(2 * ((precision * recall)/(precision + recall)), 2)))

    xgb_train_time = training_end - training_start
    xgb_prediction_time = prediction_end - prediction_start
    # accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
    # logging.info("Cross Val Accuracy: {:.2f} %".format(accuracies.mean()*100))
    # logging.info("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

    logging.info("Time for training: %4.3f seconds" % (xgb_train_time))
    logging.info("Time for prediction: %6.3f seconds" % (xgb_prediction_time))
    logging.info("------------------------------------------------------------------------------------------------")
    logging.info(" ")


def knn_post_processing(knn, X_test, y_test, X_train, y_train, training_start, training_end):
    prediction_start = time.perf_counter()
    preds = knn.predict(X_test)
    prediction_end = time.perf_counter()
    logging.info("knn Model")
    logging.info(confusion_matrix(y_test, preds))
    tn, fp, fn, tp = confusion_matrix(
        y_test, preds,
        labels=[0, 1]).ravel()
    logging.info("Accuracy:" + str(round(accuracy_score(y_test, preds), 2)))
    precision = (tp / (tp + fp))
    recall = (tp / (tp + fn))
    logging.info("Precision: " + str(round(precision, 2)))
    logging.info("Recall: " + str(round(recall, 2)))
    logging.info("f1_score: " + str(round(2 * ((precision * recall) / (precision + recall)), 2)))

    acc_knn = (preds == y_test).sum().astype(float) / len(preds) * 100
    knn_train_time = training_end - training_start
    knn_prediction_time = prediction_end - prediction_start
    logging.info("Time for training: %4.3f seconds" % (knn_train_time))
    logging.info("Time for prediction: %6.3f seconds" % (knn_prediction_time))
    logging.info("------------------------------------------------------------------------------------------------")
    logging.info(" ")


def random_forest_post_processing(rfc, X_test, y_test, X_train, y_train, training_start, training_end):
    prediction_start = time.perf_counter()
    preds = rfc.predict(X_test)
    prediction_end = time.perf_counter()
    logging.info("random forest Model")
    logging.info(confusion_matrix(y_test, preds))
    tn, fp, fn, tp = confusion_matrix(
        y_test, preds,
        labels=[0, 1]).ravel()
    logging.info("Accuracy:" + str(round(accuracy_score(y_test, preds), 2)))
    precision = (tp / (tp + fp))
    recall = (tp / (tp + fn))
    logging.info("Precision: " + str(round(precision, 2)))
    logging.info("Recall: " + str(round(recall, 2)))
    logging.info("f1_score: " + str(round(2 * ((precision * recall) / (precision + recall)), 2)))

    acc_rfc = (preds == y_test).sum().astype(float) / len(preds) * 100
    rfc_train_time = training_end - training_start
    rfc_prediction_time = prediction_end - prediction_start
    logging.info("Time for training: %4.3f seconds" % (rfc_train_time))
    logging.info("Time for prediction: %6.3f seconds" % (rfc_prediction_time))

    # rfc_cv = RandomForestClassifier(n_estimators=100)
    # scores = cross_val_score(rfc_cv, X_train, y_train, cv=10, scoring="accuracy")
    # logging.info("Scores:", scores)
    # logging.info("Mean:", scores.mean())
    # logging.info("Standard Deviation:", scores.std())
    logging.info("------------------------------------------------------------------------------------------------")
    logging.info(" ")


def svc_post_processing(svc, X_test, y_test, X_train, y_train, training_start, training_end):
    prediction_start = time.perf_counter()
    preds = svc.predict(X_test)
    prediction_end = time.perf_counter()
    logging.info("SVC Model")
    logging.info(confusion_matrix(y_test, preds))
    tn, fp, fn, tp = confusion_matrix(
        y_test, preds,
        labels=[0, 1]).ravel()
    logging.info("Accuracy:" + str(round(accuracy_score(y_test, preds), 2)))
    precision = (tp / (tp + fp))
    recall = (tp / (tp + fn))
    logging.info("Precision: " + str(round(precision, 2)))
    logging.info("Recall: " + str(round(recall, 2)))
    logging.info("f1_score: " + str(round(2 * ((precision * recall) / (precision + recall)), 2)))

    acc_svc = (preds == y_test).sum().astype(float) / len(preds) * 100
    svc_train_time = training_end - training_start
    svc_prediction_time = prediction_end - prediction_start
    logging.info("Time for training: %4.3f seconds" % (svc_train_time))
    logging.info("Time for prediction: %6.3f seconds" % (svc_prediction_time))


'''Start Program'''
if __name__ == '__main__':
    logging.basicConfig(filename='console_output.log', format='%(message)s', level=logging.INFO)
    logging.info("Start")
    # flag to determine if we are processing rgb images or one of the 12 bands alone
    rgb_flag = False
    # flag to determine whether to collect a grey scale feature
    global_feature_flag = False
    # band selected, must set rgb_flag to False to use
    band = 12

    # Preprocessing
    img_metadata, plant_training_data, plant_testing_data = import_data()
    processed_training_data = retrieve_and_store_data(plant_training_data, rgb_flag, band, global_feature_flag)
    processed_testing_data = retrieve_and_store_data(plant_testing_data, rgb_flag, band, global_feature_flag)
    X_tra, X_test, y_tra, y_test = create_train_test_data(processed_training_data, processed_testing_data,
                                                          global_feature_flag)

    # Data Modeling
    active_xgboost_model, xgb_start_time, xgb_end_time = xgboost_model(X_tra, y_tra)
    active_knn_model, knn_start_time, knn_end_time = knn_model(X_tra, y_tra)
    active_random_forest_model, rf_start_time, rf_end_time = random_forest_model(X_tra, y_tra,
                                                                                     processed_training_data['weights'])
    active_svc_model, svc_start_time, svc_end_time = svc_model(X_tra, y_tra)

    # Postprocessing
    xgboost_post_processing(active_xgboost_model, X_test, y_test, X_tra, y_tra, xgb_start_time, xgb_end_time)
    knn_post_processing(active_knn_model, X_test, y_test, X_tra, y_tra, knn_start_time, knn_end_time)
    random_forest_post_processing(active_random_forest_model, X_test, y_test, X_tra, y_tra, rf_start_time, rf_end_time)
    svc_post_processing(active_svc_model, X_test, y_test, X_tra, y_tra, svc_start_time, svc_end_time)
