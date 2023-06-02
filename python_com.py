import ctypes
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import os


def image_transform(image_dir):
    image_vector = []

    for image_name in os.listdir(image_dir):
        if image_name.endswith(".jpg") or image_name.endswith(".png"):
            image_path = os.path.join(image_dir, image_name)
            image = Image.open(image_path)
            image = image.resize((5, 5))
            pixel_values = list(image.getdata())
            image_vector.append(pixel_values)
    for vector in image_vector:
        print(f"{vector}\n {len(vector)}")


'''def expand_dims(liste, axis):
    liste = np.array(liste)  # convert input to NumPy array
    shape = list(liste.shape)
    axis = axis if axis >= 0 else len(shape) + 1 + axis
    shape.insert(axis, 1)
    return liste.reshape(shape)'''


def projections(test_samples, actual_label, predicted_label, output_classes):
    ko_cases_indices = np.where(actual_label != predicted_label)[0]

    fig, axes = plt.subplots(nrows=1, ncols=len(ko_cases_indices), figsize=(15, 5))

    for i, idx in enumerate(ko_cases_indices):
        image = test_samples[idx]
        true_label = output_classes[actual_label[idx]]
        predicted_label = output_classes[predicted_label[idx]]

        axes[i].imshow(image, cmap='gray')
        axes[i].set_title(f'True: {true_label}\nPredicted: {predicted_label}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # clib = ctypes.CDLL("D:\school\PROJET_ANNUEL_IABD3\PMC\pmc_ml_lib\linear_model.dll")
    # clibrary.function() a chaque fois que tu modifie le script C++ tu doit re-exporter le .cpp vers .dll, b"Johb" il
    # faut passer des binary string au fonctions du C++ parce que en python les string sont immutable tandis qu'en
    # C++ ils sont mutable

    # clib.my_func.argtypes = [ctypes.c_int32, ctypes.c_int32]
    # clib.my_func.restype = ctypes.c_int32

    # from_buffered()fonction retrun ctype objects

    '''arr = np.array([1, 2, 3])  # 1-dimensional array
    expanded_arr = expand_dims(arr, axis=-16)
    print(expanded_arr)'''

    # image_transform("D:\school\PROJET_ANNUEL_IABD3\PMC\dataset1")

    test_samples = [[[1, 0], [0, 1], [0, 0], [1, 1]]]  # Les données de test (images)
    actual_label = [[1, 1, -1, -1]]  # Les vraies étiquettes correspondantes
    predicted_label = 0  # Les étiquettes prédites par le modèle
    output_classes = [[1], [-1]]  # Les noms de classe correspondants

    projections(test_samples, actual_label, predicted_label, output_classes)
