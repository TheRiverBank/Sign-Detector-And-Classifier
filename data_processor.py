import numpy as np
import cv2
import os
import config

def process_data():
    data = []
    labels = []
    for i in range(config.CLASSES):
        path_to_class = os.path.join("./data/train/", str(i))
        img_dirs = os.listdir(path_to_class)
        for img_name in img_dirs:
            img = cv2.imread(path_to_class + '/' + img_name)
            img = cv2.resize(img, (32, 32))
            data.append(img)
            labels.append(i)
        data = np.array(data)
        labels = np.array(labels)

        np.save('./data/data.npy', data)
        np.save('./data/labels.npy', labels)

        return (data, labels)

def get_data():
    try:
        data = np.load('./data/data.npy')
        labels = np.load('./data/labels.npy')
    except:
        print("Data not processed\nProcessing data\n")
        data, labels = process_data()

    return data, labels


if __name__ == '__main__':
    process_data()
    data, labels = get_data()

    print(np.shape(data))