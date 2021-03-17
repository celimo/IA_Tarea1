# Se importan librerias utilizadas en el código
from tensorflow import keras     # Realizar el modelo de la NN
import pandas as pd              # Abrir los datos
import matplotlib.pyplot as plt  # Graficar curvas de validación
from mpl_toolkits.mplot3d import axes3d
from matplotlib import style
import numpy as np               # Para determinar las clases
from mlxtend.plotting import plot_confusion_matrix  # Para matriz confusion
from sklearn.metrics import confusion_matrix        # Para matriz confusion
import math

# ================ Cargar datos a utilizar ================

datos_cte0 = pd.read_csv('datos/region/cte0.csv', header=0)
datos_cte03 = pd.read_csv('datos/region/cte03.csv', header=0)
datos_cte06 = pd.read_csv('datos/region/cte06.csv', header=0)
datos_cte1 = pd.read_csv('datos/region/cte1.csv', header=0)

# ================ clasificaciones  ================

class_names = ['Move-F', 'Slight-RT',
               'Sharp-RT', 'Slight-LT']

model = keras.models.load_model("final/modeloFinal.h5")

model.load_weights("final/pesosFinal.h5")

prob_matrix = model.predict(datos_cte0)
pred_labels = np.argmax(prob_matrix, axis=-1)

class_0x = []
class_0y = []
class_0z = []

class_1x = []
class_1y = []
class_1z = []

class_2x = []
class_2y = []
class_2z = []

class_3x = []
class_3y = []
class_3z = []

sensor1 = 0.2

steps = []
for i in range(10):
    steps.append(round(i*0.1, 1))

for i in steps:
    for j in steps:
        for k in steps:
            aux = [[sensor1,i,j,k]]
            prob_matrix = model.predict(aux)
            pred_labels = np.argmax(prob_matrix, axis=-1)
            if pred_labels[0] == 0:
                class_0x.append(i)
                class_0y.append(j)
                class_0z.append(k)
                break
            elif pred_labels[0] == 1:
                class_1x.append(i)
                class_1y.append(j)
                class_1z.append(k)
                break
            elif pred_labels[0] == 2:
                class_2x.append(i)
                class_2y.append(j)
                class_2z.append(k)
                break
            elif pred_labels[0] == 3:
                class_3x.append(i)
                class_3y.append(j)
                class_3z.append(k)
                break

# Grafica 3D
fig = plt.figure()
# Creamos el plano 3D
ax1 = fig.add_subplot(111, projection='3d')
ax1.scatter(class_0x, class_0y, class_0z, c='g', marker='o')
ax1.scatter(class_1x, class_1y, class_1z, c='b', marker='o')
ax1.scatter(class_2x, class_2y, class_2z, c='r', marker='o')
ax1.scatter(class_3x, class_3y, class_3z, c='y', marker='o')
plt.savefig("regiones/regSD_F0.png")

file = open("datos/region/planoClass0Cte"+str(sensor1)+".txt", "w")
for i in range(len(class_0x)):
    file.write(str(class_0x[i])+" "+str(class_0y[i])+" "+str(class_0z[i])+"\n")
    if (i % 3) == 2:
        file.write("\n")

file = open("datos/region/planoClass1Cte"+str(sensor1)+".txt", "w")
for i in range(len(class_1x)):
    file.write(str(class_1x[i])+" "+str(class_1y[i])+" "+str(class_1z[i])+"\n")
    if (i % 3) == 2:
        file.write("\n")

file = open("datos/region/planoClass2Cte"+str(sensor1)+".txt", "w")
for i in range(len(class_2x)):
    file.write(str(class_2x[i])+" "+str(class_2y[i])+" "+str(class_2z[i])+"\n")
    if (i % 3) == 2:
        file.write("\n")

file = open("datos/region/planoClass3Cte"+str(sensor1)+".txt", "w")
for i in range(len(class_3x)):
    file.write(str(class_3x[i])+" "+str(class_3y[i])+" "+str(class_3z[i])+"\n")
    if (i % 3) == 2:
        file.write("\n")

del model                        # Se borra el modelo utilizado
