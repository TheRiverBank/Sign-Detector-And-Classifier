import numpy as np

import config
import data_processor as dp
import sign_detector
import image_processor as ip
import cv2

def resize_img(img):
    img_resized = cv2.resize(img, dsize=(32, 32), interpolation=cv2.INTER_CUBIC)
    img_resized = img_resized.reshape(-1, 32, 32, 3)
    #img_resized = cv2.blur(img_resized, (3,3), 0)
    return img_resized

if __name__ == '__main__':
    X, y = dp.get_data()

    model = sign_detector.Sign_detector(X, y, n_classes=config.CLASSES)
    model = model.train(epochs=25)

    test_img = cv2.imread("./test_images/40_test.png")
    side_bars = ip.get_side_bars(test_img)
    rects = ip.get_rects((side_bars[0], side_bars[1]))
    print(np.shape(rects))

    speed_sign_names = {0: '20', 1: '30', 2: '50', 3: '60', 4: '70', 5: '80', 6: '80 rem', 7: '100', 8: '120'}
    img = cv2.imread("./data/test/00179.png")
    img = resize_img(img)/255

    pred = model.predict(img)
    print(pred)
    print(speed_sign_names.get(np.argmax(pred)))
