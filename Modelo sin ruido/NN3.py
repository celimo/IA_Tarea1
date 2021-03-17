# Se importan librerias utilizadas en el código
from tensorflow import keras     # Realizar el modelo de la NN
import pandas as pd              # Abrir los datos
import matplotlib.pyplot as plt  # Graficar curvas de validación
import numpy as np               # Para determinar las clases
from mlxtend.plotting import plot_confusion_matrix  # Para matriz confusion
from sklearn.metrics import confusion_matrix        # Para matriz confusion

# ================ Cargar datos a utilizar ================

datos_train = pd.read_csv('datos/train.csv', header=0)        # Entrenamiento
datos_test = pd.read_csv('datos/test.csv', header=0)          # Validacion

# =============== Dar formato a los datos de entrenamiento  ===============

train_labels = datos_train.pop('Class')  # Separar etiquetes
train_data = datos_train                 # Asignar datos de entrenamiento

# =============== Dar formato a los datos de validación  ===============

test_labels = datos_test.pop('Class')  # Separar etiquetas
test_data = datos_test                 # Asignar datos de validación

# =============== Ajustar el valor de los datos ===============
# Los datos deben estar entre 0 y 1, ya que se trabaja con probabilidades

train_data = train_data/5  # Se ajustan los datos de entrenamiento
test_data = test_data/5    # Se ajustan los datos de validación

# ================ clasificaciones  ================

class_names = ['Move-F', 'Slight-RT',
               'Sharp-RT', 'Slight-LT']

model = keras.models.load_model("final/modeloFinal.h5")

model.load_weights("final/pesosFInal.h5")

prob_matrix = model.predict(test_data)
pred_labels = np.argmax(prob_matrix, axis=-1)

mat = confusion_matrix(pred_labels, test_labels)
plot_confusion_matrix(conf_mat=mat,
                      figsize=(9, 9),
                      class_names=class_names,
                      show_normed=True,
                      cmap=plt.cm.Blues)
plt.xlabel("Valor predicho")
plt.ylabel("Valor real")
plt.savefig("confMatrix/MatrizConf.png")

del model                        # Se borra el modelo utilizado
