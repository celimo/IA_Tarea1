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

# ================ Se crea el modelo ================
# Se trabaja con 3 capas: 1 de entrada, 1 oculta y 1 de salida
# 4 entradas
# 26 percetrones en la capa oculta
# 4 clasificaciones en la salida

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(4,)),       # input layers
    keras.layers.Dense(4, activation='sigmoid'),  # hidden layers
    keras.layers.Dense(4, activation='softmax')   # output layers
    ])

# ================ Se compila el modelo ================
# Se trabaja con el optimizador "adaptive moment estimation (ADAM)"
# La función de pérdida se configura para trabajar en clasificación
# Se va a medir la precisión de los datos
# El objeto optimizador indica el tipo con su tasa de aprendizaje

#optimizador = keras.optimizers.RMSprop(learning_rate=0.01,
#                                       momentum=0.0)

optimizador = keras.optimizers.Adam(learning_rate=0.1,
                                    beta_1=0.8)

# Se compila el modelo
# optimizer: Optimizador a usar
# loss: tipo de función de pérdida
# metrics: la métrica a evaluar durante el entrenamiento y el testing

model.compile(optimizer=optimizador,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ================ Se entrena el modelo con los datos ================

iteration = 400  # Cantidad de iteraciones para el entrenamiento
CantVal = int(iteration * 0.1)  # Cantidad de validaciones
freqVal = int(iteration / CantVal)  # Frecuencia de las validaciones

# Se inicia  el entrenamiento de la red
# Los primeros parámetros son los datos de entrenamiento y las clasificaciones
# epochs: iteraciones del entrenamiento
# validation_split: Utiliza un 30% de los datos para validación
# validation_freq: cada cierta freqVal de iteraciones se hace la validación
# verbose: No se muestra todas las iteraciones del entrenamiento

print("Realizando entrenamiento...")

training = model.fit(train_data, train_labels.to_numpy(),
                     epochs=iteration,
                     validation_split=0.3,
                     validation_freq=freqVal,
                     verbose=0)

print("Fin entrenamiento")

# ================ Configuración para realizar la gráfica ================

loss = training.history['loss']  # Datos de la perida de entrenamiento
val_loss = training.history['val_loss']  # Datos de la perdida de validación
x_loss = []  # Datos eje x para la curva de etrenamiento
x_val = []  # Datos eje x para la curva de validación

for i in range(iteration):  # Se agregan datos al eje x de entrenamiento
    x_loss.append(i + 1)
for i in range(CantVal):  # Se agregan datos al eje x de validación
    x_val.append((i + 1) * freqVal)

plt.xlabel("Iteración")
plt.ylabel("Error")
plt.plot(x_loss, loss, label="Pérdida de entrenamiento")
plt.plot(x_val, val_loss, label="Pérdida de validación")
plt.legend()
plt.savefig("Graph/CurvasEntrenamiento.png")

# ================ Guardar curvas de périda ================

file = open("curvas/LossTrain.txt", "w")
for i in range(len(loss)):
    file.write(str(x_loss[i]) + " " + str(loss[i]) + "\n")
file.close()

file = open("curvas/ValidTrain.txt", "w")
for i in range(len(val_loss)):
    file.write(str(x_val[i]) + " " + str(val_loss[i]) + "\n")
file.close()

# ================ Crear matriz de confusión ================
# prob_matrix: Se calcula la probabilidad que el dato de entrada corresponda
#               a cierta clasificación
# pred_labels: Se obtiene el indice donde se encuentra la mayor probabilidad
#              este array es el que es comparado con el test_labels para
#              la matriz de confusión

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

# ================ Guardar los pesos del entrenamiento ================
# Se guardan los pesos para comparar si mejoraron el resultado

model.save_weights('model/weights.h5',
                   overwrite=True)

# ================ Guardar el modelo utilizado ================
# El modelo se guarda para comparar los resultados y en el cas que Sea
# mejor al anterior se guarda

model.save('model/my_model.h5')  # Se guarda con una extensión h.5
del model                        # Se borra el modelo utilizado
