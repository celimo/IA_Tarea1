# Se importan librerias utilizadas en el código
from tensorflow import keras     # Realizar el modelo de la NN
import pandas as pd              # Abrir los datos
import matplotlib.pyplot as plt  # Graficar curvas de validación

# ================ Cargar datos a utilizar ================

datos_train = pd.read_csv('train.csv', header=0)  # Entrenamiento
datos_test = pd.read_csv('test.csv', header=0)    # Validacion

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

class_names = ['Move-Forward', 'Slight-Right-Turn',
               'Sharp-Right-Turn', 'Slight-Left-Turn']

# ================ Se crea el modelo ================
# Se trabaja con 3 capas: 1 de entrada, 1 oculta y 1 de salida
# 4 entradas
# 26 percetrones en la capa oculta
# 4 clasificaciones en la salida

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(1, 4)),      # input layers
    keras.layers.Dense(26, activation='sigmoid'),  # hidden layers
    keras.layers.Dense(4, activation='softmax')    # output layers
    ])

# ================ Se compila el modelo ================
# Se trabaja con el optimizador "adaptive moment estimation (ADAM)"
# La función de pérdida se configura para trabajar en clasificación
# Se va a medir la precisión de los datos
# El objeto optimizador indica el tipo con su tasa de aprendizaje

optimizador = keras.optimizers.Adam(learning_rate=0.01)

# Se compila el modelo
# optimizer: Optimizador a usar
# loss: tipo de función de pérdida
# metrics: la métrica a evaluar durante el entrenamiento y el testing

model.compile(optimizer=optimizador,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ================ Se entrena el modelo con los datos ================

iteration = 200  # Cantidad de iteraciones para el entrenamiento
CantVal = int(iteration * 0.1)  # Cantidad de validaciones
freqVal = int(iteration / CantVal)  # Frecuencia de las validaciones

# Se inicia  el entrenamiento de la red
# Los primeros parámetros son los datos de entrenamiento y las clasificaciones
# epochs: iteraciones del entrenamiento
# validation_data: datos de validación (datos, clasificación)
# validation_freq: cada cierta freqVal de iteraciones se hace la validación

training = model.fit(train_data, train_labels,
                     epochs=iteration,
                     validation_data=(test_data, test_labels),
                     validation_freq=freqVal)

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
plt.show()
