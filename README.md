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

# Redes Neuronales Artificiales

Las **redes neuronales artificiales (RNA)** intentan reproducir el comportamiento del cerebro. El modelo realiza una simplificación averiguando cuáles son los elementos más relevantes del sistema.

---

## El modelo biológico

Las neuronas son células que contienen los mismos elementos que las células biológicas pero tiene características que las diferencian:

* Miden aproximadamente de 5 a 10 micras.
* Cuentan con una rama principal o **axón** el cual puede ramificarse en su extremo.
* Se desprenden varias ramas más cortas llamadas **dendritas**.

### Capacidad de comunicación

Las neuronas tienen la capacidad de comunicarse: reciben información de miles de neuronas y envían a otras más. En el cerebro humano se calculan 10¹⁵ conexiones.

Las señales que usan son **eléctricas** y **químicas**:

* Las generadas en la neurona y transmitidas por el axón son **eléctricas**.
* Las que se realizan entre una terminal axónica y las dendritas son **químicas**.

### Potencial de reposo y de acción

El medio externo tiene una mayor concentración de iones de sodio y el medio interno tiene una mayor concentración de iones de potasio. La diferencia es del orden de 10 veces mayor. Esto genera una diferencia de potencial del orden de los **70 mV** (negativo en el interior) llamado **potencial de reposo** de la célula nerviosa.

Las señales que llegan a las dendritas actúan acumulativamente (bajando el potencial de reposo). Cuando llega a un valor crítico comienza la entrada de iones de sodio que invierten la polaridad. La inversión del voltaje llamada **potencial de acción** cierra el paso a los iones de potasio hasta que se restablece el equilibrio en reposo.

La inversión del voltaje se propaga a lo largo del axón provocando la emisión de los terminales axónicos.

### Tipos de sinapsis

* **Sinapsis excitadoras**: los neurotransmisores facilitan la generación de impulsos.
* **Sinapsis inhibidoras**: los neurotransmisores dificultan la emisión de impulsos.

---

## Red neuronal artificial

Tiene dispositivos elementales de proceso que llamamos **neuronas**. Hay tres tipos de neuronas:

* **Entrada**: reciben estímulos externos.
* **Oculta (hidden)**: se ocupan del procesado interno.
* **Salida**: son las que dan la respuesta del sistema.

Se llama **capa o nivel** a un conjunto de neuronas cuyas entradas provienen de la misma fuente.

### Estado de activación

Además del conjunto de unidades se necesitan los estados del sistema en un tiempo *t*. Cada elemento del vector representa la activación de una unidad Uᵢ en el tiempo *t*:

$$
A(t) = (a_1(t), a_2(t), \ldots, a_N(t))
$$

El procesamiento en la red es una evolución de un patrón de activación en el conjunto de unidades a través del tiempo.

### Conexión entre neuronas

Las conexiones que unen una red de neuronas tiene asociado un **peso** que hace que la red adquiera conocimiento. Cada conexión (sinapsis) entre la neurona *i* y *j* está ponderada por un peso **wᵢⱼ**.

Como el efecto de cada señal es aditivo, la **entrada neta** que recibe una neurona (potencial postsináptico) Netⱼ es la suma de cada valor individual por el valor de la sinapsis:

$$
Net_j = \sum w_{ij} \cdot y_i
$$

* Si **wᵢⱼ > 0**: la interacción entre las neuronas es **excitadora**. Siempre que *i* esté activada intentará activar a la neurona *j*.
* Si **wᵢⱼ < 0**: la sinapsis es **inhibidora**.
* Si **wᵢⱼ = 0**: no hay conexión entre las neuronas.

### Función de transferencia

Asociada con cada neurona hay una **función de activación o transferencia** *f(Netᵢ)* que transforma la entrada neta de la neurona en una salida:

$$
Y(t) = (f_1(Net_1(t)), f_2(Net_2(t)), \ldots, f_N(Net_N(t)))
$$

Funciones de activación comunes:

* **Función escalón**: la salida es binaria (0 o 1).
* **Función lineal**: *f(x) = x*.
* **Sigmoidal**: función continua, rango (0, 1).
* **Gaussiana**: más fácil de adaptar.

### Reglas de aprendizaje

Durante el proceso de aprendizaje se producen cambios en las conexiones de las neuronas artificiales. Se considera que el conocimiento se encuentra representado en los pesos de las conexiones entre neuronas.

**Por supervisión:**

* **Supervisado**: entrenamiento supervisado por un agente externo.
    * Aprendizaje por corrección de errores.
    * Aprendizaje por refuerzo.
    * Aprendizaje estocástico.
* **No supervisado o auto supervisado**: no existe un agente externo.
    * Aprendizaje Hebbiano.
    * Aprendizaje competitivo y cooperativo.
* **Reforzado**: solo se indica la salida (ej. redes de Hopfield).

**Por estado de la red:**

* **On-line**: la red puede aprender durante su funcionamiento habitual.
* **Off-line**: hay dos fases — una de entrenamiento/aprendizaje y otra de operación.

### Formas de conexión

Las conexiones en una red están relacionadas de manera que la salida de una neurona se canaliza para convertirse en la entrada de otra:

* **Propagación hacia adelante (feedforward)**.
* **Propagación hacia adelante y atrás (feedforward/feedback)**.

---

## Modelos de redes neuronales

### El perceptrón

Red con una sola capa de neuronas de salida. Es el modelo más simple de RNA.

### La red backpropagation

Es un método para que una RNA aprenda la asociación que existe entre los patrones de entrada y las clases correspondientes. Funciona con una etapa de entrenamiento donde aprende de un conjunto predefinido de observaciones de entrada y salida, y una segunda etapa de puesta en marcha.

### El modelo de Hopfield

Consiste en asociar algo con una característica notoria. Funciona como una **memoria asociativa**. Es una red monocapa con N neuronas donde sus valores de salida son binarios (0 y 1, o -1 y 1). Tiene limitaciones como la cantidad máxima de patrones que puede almacenar: si son muchos, los resultados pueden converger en salidas diferentes.

---

# El modelo de Kohonen

Es un modelo de RNA con la capacidad de formar **mapas de características** como los topológicos en el cerebro.

Tiene dos variantes:

* **Learning Vector Quantization (LVQ)**.
* **Topology Preserving Map / Self Organizing Map (SOM)**.

## Estructura de la red Kohonen

Red de **dos capas**:

* **N neuronas de entrada**: reciben los atributos del patrón.
* **M neuronas de salida**: forman la grilla bidimensional del mapa.

Cada una de las N neuronas de entrada se conecta a las M neuronas de salida a través de una **conexión hacia delante** (feedforward). Entre las neuronas de la capa de salida hay **conexiones laterales de inhibición** de manera que van a tener cierta influencia en sus vecinas.

### Conexiones laterales

Las conexiones laterales entre las neuronas de salida permiten la **cooperación**: cuando una neurona es la BMU, sus vecinas también se acercan al patrón de entrada. Este mecanismo es lo que permite preservar la topología del espacio de entrada en el mapa.

---

## Aplicaciones de la red Kohonen

### Reconocimiento de voz

Un prototipo para convertir en tiempo real el habla en texto escrito, a partir de un vocabulario limitado. La red recibe 15 neuronas a través de las cuales recibe las componentes de frecuencia que caracterizan a los diferentes fonemas. Ante fonemas parecidos se activan en la red neuronas próximas de salida, creando un **mapa fonotópico**.

En la fase de funcionamiento, al escuchar una palabra la debe dividir secuencialmente en fonemas que se presentan a la red que los recibe y activa una u otra neurona de salida, finalizando con la transcripción fonética de la palabra.

### Reconocimiento de texto manuscrito

Se construye un **mapa grafotópico**, que convierte texto manuscrito en un texto estándar con caracteres en ASCII. La red recibe la representación paramétrica de un *stroke* (porción de trazo de escritura manuscrita). Se entrena con diferentes strokes (red de 15x15 neuronas de salida y 10 de entrada) generando un mapa grafotópico. Después del aprendizaje, cuando se analiza una palabra durante el funcionamiento se la divide en strokes que la red analiza secuencialmente y responde activando en orden las correctas neuronas de la capa de salida.

### Resolución de problemas de optimización

Como el **problema del viajante** (Traveling Salesman Problem): se establece el camino más corto entre una serie de ciudades sin pasar dos veces por la misma ciudad. La red debe constar de tantas neuronas de salida como ciudades del recorrido, y como entrada necesitará dos neuronas para representar las coordenadas en el plano. Lo que se hace es entrenar la red para que aprenda la situación geográfica de cada una de las ciudades.

---

## Fundamentos Teóricos del SOM

---

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

Un heatmap por cada atributo mostrando cómo se distribuyen los pesos en la grilla. Permite ver qué zona del mapa responde a qué característica.

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

# Aplicaciones del SOM

## Aplicaciones generales

* Segmentación de clientes.
* Detección de anomalías.
* Bioinformática.
* Minería de datos.
* Reducción de dimensionalidad.
* Visualización de datos de alta dimensión.
* Descubrimiento de patrones.

## Aplicaciones específicas de la red Kohonen

### Reconocimiento de voz (mapa fonotópico)

Un prototipo para convertir en tiempo real el habla en texto escrito a partir de un vocabulario limitado. La red recibe las componentes de frecuencia que caracterizan los fonemas. Ante fonemas parecidos se activan neuronas próximas de salida, creando un mapa fonotópico.

### Reconocimiento de texto manuscrito (mapa grafotópico)

Convierte texto manuscrito en texto estándar ASCII. Se entrena con *strokes* (porciones de trazo) usando una red de 15x15 neuronas de salida y 10 de entrada.

### Problema del viajante (TSP)

Resolución del problema del viajante (Traveling Salesman Problem): encontrar el camino más corto entre ciudades sin pasar dos veces por la misma. La red tiene tantas neuronas de salida como ciudades y 2 neuronas de entrada para las coordenadas geográficas.

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
