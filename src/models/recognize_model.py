from sklearn.svm import SVC 
import pickle
import cv2
from cv2 import resize
import os
import numpy as np
from pandas import DataFrame
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.preprocessing.extract_feature import extract_lbp_features, extract_hog_features


def load_model(path:str):
    with open(path, 'rb') as file:
        return pickle.load(file)

def save_model(model, path:str):
    with open(path, 'wb') as file:
        pickle.dump(model, file)


class Face_Recognition(SVC):

    def __init__(self) -> None:
        super().__init__(kernel="linear", probability=True, C=10, gamma='auto', degree=0)
        self.pca_transf = PCA(n_components=None, whiten=True)
        self.std_scl = StandardScaler()
        self.input_img = (64, 64)
        self.threshold = 0.4
    
    def _reshape_img(self, img):
        return resize(img, self.input_img)

    def recognize(self, frame, faces):
        recog_faces = []
        for (x1, y1, x2, y2) in faces:
            face = frame[y1:y2, x1:x2]


            name = self.predict([face])[0]
            name_prob = self.predict_proba([face])[0]
            name_prob = np.max(name_prob)

            if name_prob < self.threshold:
                name = "Unknown"
                name_prob = 0
            recog_faces.append([[x1, y1, x2-x1, y2 - y1], f"{name_prob:.2f}", name])

        return recog_faces
    
    def _cvt_to_gray(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    def _get_features(self, img):
        lbp_features = extract_lbp_features(img, radius=16, n_points=48)
        hog_features = extract_hog_features(img)
        features = np.concatenate((lbp_features, hog_features))
        return features


    def _convert_img_to_feature_for_predict(self, img):
        img = self._reshape_img(img)
        X = self._get_features(img)

        X = self.std_scl.transform([X])[0]
        X = self.pca_transf.transform([X])[0]
        return X

    def _get_data_from_db(self, face_db):
        X = []
        y = []

        names = os.listdir(path=face_db)
        for name in names:
            imgs = os.listdir(f"{face_db}/{name}")
            for img in imgs:
                _img = cv2.imread(f"{face_db}/{name}/{img}")
                _img = self._cvt_to_gray(_img)
                _img = self._reshape_img(_img)
                X.append(_img)
                y.append(name)
        return np.array(X), np.array(y)


    def fit_model(self, face_db):        
        X, y = self._get_data_from_db(face_db=face_db)
        self.fit(X, y)

    def fit(self, X: np.ndarray | DataFrame, y , sample_weight=None):
        X_train = []

        for x in X:
            feature = self._get_features(x)
            X_train.append(feature)

        X_train = np.array(X_train)

        X_train = self.std_scl.fit_transform(X_train)
        X_train = self.pca_transf.fit_transform(X_train)
        print("X shape", X_train.shape)

        return super().fit(X_train, y)
    
    def predict(self, imgs:np.ndarray):
        X = []
        for im in imgs:
            feature = self._convert_img_to_feature_for_predict(im)
            X.append(feature)
        X = np.array(X)
        return super().predict(X)
    
    def predict_proba(self, imgs) -> np.ndarray:
        X = []
        for im in imgs:
            feature = self._convert_img_to_feature_for_predict(im)
            X.append(feature)
        X = np.array(X)
        return super().predict_proba(X)
    



