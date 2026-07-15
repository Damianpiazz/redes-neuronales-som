# redes-neuronales-som

Implementación *from scratch* de un **Mapa Autoorganizado de Kohonen (SOM - Self-Organizing Map)**, desarrollado como proyecto académico de Redes Neuronales y Aprendizaje No Supervisado.

---

# Objetivos del Proyecto

Este repositorio tiene como objetivo:

* Implementar un SOM desde cero.
* Comprender el funcionamiento del aprendizaje no supervisado.
* Agrupar patrones similares mediante clustering.
* Reducir la dimensionalidad de los datos.
* Preservar las relaciones topológicas entre las observaciones.
* Visualizar la organización de los datos en una grilla bidimensional.
* Evaluar la calidad del entrenamiento mediante métricas específicas.

---

# ¿Qué es un SOM?

Los **Self-Organizing Maps (SOM)**, también conocidos como **Mapas Autoorganizados de Kohonen**, son una red neuronal de aprendizaje **no supervisado** propuesta por **Teuvo Kohonen**.

Su objetivo es proyectar datos de alta dimensión sobre una grilla bidimensional, preservando la vecindad entre los patrones originales. De esta manera, observaciones similares quedan representadas en neuronas cercanas dentro del mapa.

---

# Algoritmo Implementado

## SOM (Self-Organizing Map)

El SOM es una red neuronal que organiza datos de alta dimensión en una grilla bidimensional de neuronas.

Características:

* Aprendizaje no supervisado.
* Preserva la topología de los datos.
* Cada neurona posee un vector de pesos con la misma dimensión que los datos de entrada.
* Utiliza la distancia euclídea para encontrar la neurona más cercana.
* Actualiza los pesos de la BMU y sus vecinas de forma iterativa.

---

# Fundamentos Teóricos

## Neurona

Cada neurona del mapa posee un vector de pesos con la misma dimensión que los datos de entrada.

$$
W_i=(w_1,w_2,\ldots,w_n)
$$

---

## Best Matching Unit (BMU)

Para cada patrón de entrada se busca la neurona cuyos pesos sean los más similares al vector de entrada.

La neurona ganadora se obtiene mediante:

$$
BMU=\arg\min_i ||X-W_i||
$$

Generalmente se utiliza la distancia euclídea:

$$
d(X,W_i)=
\sqrt{
\sum_{j=1}^{n}
(x_j-w_j)^2
}
$$

---

## Función de Vecindad

Además de actualizar la neurona ganadora, también se modifican las neuronas cercanas.

La función de vecindad gaussiana es:

$$
h_{ci}(t)=
\exp
\left(
-\frac{
d_{grid}(c,i)^2
}{
2\sigma(t)^2
}
\right)
$$

donde:

* \(c\) representa la BMU.
* \(i\) representa una neurona vecina.
* \(\sigma(t)\) es el radio de vecindad.

---

## Actualización de Pesos

Los pesos se actualizan según la siguiente expresión:

$$
W_i(t+1)=
W_i(t)
+
\alpha(t)
\,
h_{ci}(t)
\,
(X-W_i(t))
$$

donde:

* \(\alpha(t)\) es la tasa de aprendizaje.
* \(h_{ci}(t)\) es la función de vecindad.
* \(X\) es el patrón de entrada.

---

## Normalización

Antes del entrenamiento los datos deben normalizarse.

Una normalización Min-Max puede expresarse como:

$$
X'=
\frac{
X-X_{min}
}{
X_{max}-X_{min}
}
$$

---

# Datasets Utilizados

## Breast Cancer Wisconsin

Clasificación de tumores como benignos o malignos basándose en características celulares.

Archivo:

```text
data/breast.txt
```

### Atributos

| Atributo    | Tipo    |
| ----------- | ------- |
| clump       | Numérico |
| ucellsize   | Numérico |
| ucellshape  | Numérico |
| mgadhesion  | Numérico |
| sepics      | Numérico |
| bnuclei     | Numérico |
| bchromatin  | Numérico |
| normnucl    | Numérico |
| mitoses     | Numérico |
| class       | Clase objetivo (begnin / malignant) |

---

## Cars

Predicción de millas por galón (MPG) según características del vehículo.

Archivo:

```text
data/cars.xls
```

### Atributos

| Atributo      | Tipo        |
| ------------- | ----------- |
| Country       | Categórico  |
| Car           | Categórico  |
| MPG           | Continua    |
| Weight        | Continua    |
| Drive_Ratio   | Continua    |
| Horsepower    | Continua    |
| Displacement  | Continua    |
| Cylinders     | Numérico    |

---

# Entrenamiento del SOM

## Inicialización

1. Crear la grilla de neuronas.
2. Inicializar los pesos de forma aleatoria.

---

## Iteración por Época

Para cada época:

1. Seleccionar un patrón de entrada.
2. Encontrar la Best Matching Unit (BMU).
3. Calcular la función de vecindad.
4. Actualizar los pesos de las neuronas.
5. Reducir la tasa de aprendizaje.
6. Reducir el radio de vecindad.

---

# Pseudocódigo

```text
Inicializar pesos aleatorios

Para cada época:

    Para cada patrón X:

        Encontrar la BMU

        Para cada neurona:

            Calcular vecindad

            Actualizar pesos

    Reducir learning rate

    Reducir radio de vecindad
```

---

# Arquitectura de la Red

```text
Breast Cancer:
9 atributos, 699 instancias
Mapa: 10 x 10 neuronas
Total: 100 neuronas

Cars:
6 atributos, 38 instancias
Mapa: 6 x 6 neuronas
Total: 36 neuronas
```

---

# Métricas de Evaluación

## Quantization Error

Mide qué tan bien representan los vectores de pesos a los datos originales.

$$
QE=
\frac{1}{N}
\sum
||X-BMU||
$$

Un menor valor indica una mejor representación.

---

## Topographic Error

Evalúa si el mapa preserva correctamente la topología de los datos.

$$
TE=
\frac{
Errores\ Topológicos
}{
N
}
$$

Valores cercanos a cero indican una buena preservación de la estructura del conjunto de datos.

---

# Visualizaciones

## U-Matrix

Representa las distancias entre neuronas vecinas y permite identificar agrupamientos naturales. Valores altos indican fronteras entre clusters.

## Hit Map

Muestra la cantidad de observaciones asignadas a cada neurona del mapa. Permite identificar zonas de alta densidad y neuronas muertas.

## Cluster Map

Proyecta las observaciones sobre la grilla coloreadas por su clase real. Permite ver cómo el SOM separa los grupos.

## Component Planes

Un heatmap por cada atributo mostrando cómo se distribuyen los pesos en la grilla. Permite ver qué zona del mapa responde a qué特征a.

## Network Graph

Visualización como grafo donde cada nodo es una neurona y las aristas conectan vecinas. El color del nodo indica la clase dominante y el grosor/color de las aristas refleja la distancia entre pesos (U-Matrix como grafo).

---

# Resultado Esperado

## Breast Cancer

```text
Grilla: 10 x 10
Epocas: 200
Quantization Error: 0.3527
Topographic Error: 0.0544
```

## Cars

```text
Grilla: 6 x 6
Epocas: 200
Quantization Error: 0.1684
Topographic Error: 0.1579
```

---

# Descubrimientos

## Breast Cancer Wisconsin

* El SOM logra separar correctamente tumores benignos de malignos. Las neuronas en las esquinas superiores del mapa representan mayoritariamente casos malignos, mientras que las inferiores corresponden a benignos.
* El **Quantization Error de 0.3527** indica que los pesos representan razonablemente bien los datos, pero hay variabilidad dentro de los clusters.
* El **Topographic Error de 0.0544** (5.4%) es bajo, lo que confirma que el mapa preserva la topología: observaciones similares quedan en neuronas cercanas.
* El atributo **bnuclei** (núcleos anormales) es el que más influye en la separación, seguido por **ucellsize** y **ucellshape**. Esto es consistente con lo que se espera médicamente.
* La U-Matrix muestra una frontera clara entre las dos clases, con una zona de transición en el centro del mapa.

## Cars

* El SOM agrupa los vehículos por origen geográfico: autos americanos (U.S.) tienden a concentrarse en una zona del mapa, europeos en otra y japoneses en otra.
* El **Quantization Error de 0.1684** es bajo, buena representación de los datos.
* El **Topographic Error de 0.1579** (15.8%) es más alto que en Breast Cancer, lo que indica que la topología no se preserva tan bien. Esto se debe a que el dataset es pequeño (38 instancias) y tiene atributos con escalas muy diferentes.
* El atributo **Weight** y **Horsepower** son los que más diferencian los grupos. Los autos americanos tienden a ser más pesados y con más caballos.
* El dataset de Cars es más difícil de clustering porque la frontera entre orígenes no es tan clara como entre benigno/maligno.

## Comparación entre datasets

* El SOM funciona mejor con datasets grandes y con clases bien separadas (Breast Cancer: 699 instancias, 2 clases claras).
* Con datasets pequeños (Cars: 38 instancias) y muchas dimensiones, el SOM tiende a sobreajustar y el topographic error sube.
* La normalización Min-Max es crítica: sin ella, atributos con escalas grandes (como Weight) dominan la distancia euclídea.

---

# Aplicaciones

* Segmentación de clientes.
* Detección de anomalías.
* Bioinformática.
* Minería de datos.
* Reducción de dimensionalidad.
* Visualización de datos de alta dimensión.
* Descubrimiento de patrones.

---

# Estructura del Proyecto

```text
redes-neuronales-som/
│
├── data/
│   ├── breast.txt
│   └── cars.xls
│
├── src/
│   ├── neuron.py
│   ├── som.py
│   ├── distance.py
│   ├── neighborhood.py
│   ├── utils.py
│   └── visualization.py
│
├── tests/
│   ├── test_bmu.py
│   ├── test_training.py
│   └── test_quantization.py
│
├── plots/
│   ├── breast_*.png
│   └── cars_*.png
│
├── main.py
├── requirements.txt
├── pytest.ini
└── README.md
```

---

# Instalación

## 1. Clonar repositorio

```bash
git clone <repo-url>
```

---

## 2. Entrar al proyecto

```bash
cd redes-neuronales-som
```

---

## 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

# Ejecución

Ejecutar todos los datasets:

```bash
python main.py
```

Ejecutar un dataset específico:

```bash
python main.py breast
python main.py cars
```

---

# Complejidad

Sea:

* \(N\): cantidad de patrones.
* \(M\): cantidad de neuronas del mapa.
* \(D\): dimensión de cada patrón.

La búsqueda de la **Best Matching Unit** posee una complejidad aproximada de:

$$
O(N \cdot M \cdot D)
$$

---

# Testing

Ejecutar:

```bash
pytest
```

---

# Referencias

* Kohonen, T. (1982). *Self-Organized Formation of Topologically Correct Feature Maps.*
* Kohonen, T. (2001). *Self-Organizing Maps.*
* Haykin, S. *Neural Networks and Learning Machines.*
