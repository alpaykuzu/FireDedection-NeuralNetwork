import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.preprocessing import LabelEncoder

base_path = 'Data'
categories = ['Fire', 'NoFire']

color_spaces = ['RGB', 'HSV']

def extract_color_distribution(image):
    r, g, b = cv2.split(image)
    hist_r, _ = np.histogram(r, bins=256, range=(0, 256))
    hist_g, _ = np.histogram(g, bins=256, range=(0, 256))
    hist_b, _ = np.histogram(b, bins=256, range=(0, 256))
    return np.concatenate([hist_r, hist_g, hist_b])

def extract_texture_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, 8, 1, method='uniform')
    hist_lbp, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
    return hist_lbp

def extract_features(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None

    image = cv2.resize(image, (512, 512))
    
    # Merkez 128x128 bölgeyi seç
    center_x = image.shape[1] // 2
    center_y = image.shape[0] // 2
    half_size = 64
    center_region = image[center_y - half_size:center_y + half_size, center_x - half_size:center_x + half_size]

    features = []

    for color_space in color_spaces:
        if color_space == 'HSV':
            center_region = cv2.cvtColor(center_region, cv2.COLOR_BGR2HSV)

        hist = cv2.calcHist([center_region], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]).flatten()
        features.extend(hist)

    color_distribution_features = extract_color_distribution(center_region)
    texture_features = extract_texture_features(center_region)

    mean_color_val = np.mean(center_region, axis=(0, 1))
    gray = cv2.cvtColor(center_region, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)
    glcm_features = []
    glcm_props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    for prop in glcm_props:
        glcm_props_val = graycoprops(glcm, prop)[0, 0]
        glcm_features.append(glcm_props_val)
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        moments = cv2.moments(contour)
        hu_moments = cv2.HuMoments(moments).flatten()
    else:
        area = 0
        perimeter = 0
        hu_moments = np.zeros(7)
    shape_features = [area, perimeter] + list(hu_moments)
    mean_val = np.mean(gray)
    std_val = np.std(gray)
    stat_features = [mean_val, std_val]

    features += list(mean_color_val) + glcm_features + shape_features + stat_features + list(color_distribution_features) + list(texture_features)
    
    return features

data = []
labels = []

for category in categories:
    folder_path = os.path.join(base_path, category)
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        features = extract_features(file_path)
        if features is not None:
            data.append(features)
            labels.append(category)

num_color_bins = 8 * 8 * 8
num_glcm_features = 6
num_shape_features = 9
num_stat_features = 2

color_columns_rgb = [f'color_bin_rgb_{i}' for i in range(num_color_bins)]
color_columns_hsv = [f'color_bin_hsv_{i}' for i in range(num_color_bins)]
mean_color_columns = ['mean_color_red', 'mean_color_green', 'mean_color_blue']
glcm_columns = ['glcm_contrast', 'glcm_dissimilarity', 'glcm_homogeneity', 'glcm_energy', 'glcm_correlation', 'glcm_asm']
shape_columns = ['shape_area', 'shape_perimeter'] + [f'hu_moment_{i}' for i in range(7)]
stat_columns = ['mean_val', 'std_val']
color_distribution_columns = [f'color_distribution_{i}' for i in range(768)]
texture_columns = [f'texture_{i}' for i in range(256)]
columns = color_columns_rgb + color_columns_hsv + mean_color_columns + glcm_columns + shape_columns + stat_columns + color_distribution_columns + texture_columns

df = pd.DataFrame(data, columns=columns)
df['label'] = labels

label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

df.to_csv('Data_image_features.csv', index=False)
