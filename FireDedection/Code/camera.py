import cv2
import numpy as np
from joblib import load
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

# Pretrained model yükleme
model_classifier = load('trained_model.joblib')

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

def extract_features(frame):
    if frame is None:
        return None

    image = cv2.resize(frame, (512, 512))
    
    # Merkez 128x128 bölgeyi seç
    center_x = image.shape[1] // 2
    center_y = image.shape[0] // 2
    half_size = 64
    center_region = image[center_y - half_size:center_y + half_size, center_x - half_size:center_x + half_size]

    features = []
    color_spaces = ['RGB', 'HSV']
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
    
    return np.array(features)

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # Dikdörtgeni çizmek için koordinatları hesapla
    original_height, original_width = frame.shape[:2]
    center_x = original_width // 2
    center_y = original_height // 2
    half_size = 64  # 64x64 bölgenin yarısı
    top_left = (center_x - half_size, center_y - half_size)
    bottom_right = (center_x + half_size, center_y + half_size)
    
    # Dikdörtgeni siyah ile çiz
    cv2.rectangle(frame, top_left, bottom_right, (0, 0, 0), 2)

    features = extract_features(frame)
    if features is not None:
        features_scaled = np.expand_dims(features, axis=0)  # Features array'ini 2D hale getirme
        prediction = model_classifier.predict(features_scaled)
        
        if prediction == 0:
            prediction_text = "ATES VAR"
        elif prediction == 1:
            prediction_text = "ATES YOK"
        else:
            prediction_text = "Tahmin Hatası"
        
        print("Predicted Class: " + prediction_text)
        cv2.putText(frame, prediction_text, (150, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Object Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
