# Implementación de la arquitectura RCAN de Super-Resolución en Tensorflow 2.4

El desarrollo de este proyecto se ha realizado apoyándose en la implementación original de la arquitectura en PyTorch [<a href='https://github.com/yulunzhang/RCAN'>ver</a>] y, en mayor medida, en esta adaptación realizada a Tensorflow 1.13 [<a href='https://github.com/dongheehand/RCAN-tf'>ver</a>].

La red diseñada ha sido entrenada utilizando los mismos datasets de entrenamiento y validación, <a href='https://data.vision.ee.ethz.ch/cvl/DIV2K/'>DIV2K</a>, que en las dos implementaciones mencionadas. Como datasets de test, se han empleado cuatro conjuntos de imágenes reconocidos en la literatura: Set5, Set14, Urban100 y Manga109, disponibles públicamente <a href='https://cvnote.ddlee.cc/2019/09/22/image-super-resolution-datasets'>aquí</a>. Las imágenes de los datasets se almacenan en formato .bmp para evitar pérdidas por compresión (como las originadas por el formato .jpg).

El entrenamiento se ha realizado de la forma más similar posible al entrenamiento original de la red con el objetivo de replicar sus resultados. Por tanto, los parámetros del entrenamiento son:
- <u>Número de iteraciones (training steps)</u>: 1.000.000.
- <u>Optimizador</u>: Adam con parámetros por defecto.
- <u>Learning rate</u>: 1e-4, que cada 200.000 iteraciones se reduce a la mitad.
- <u>Validación</u>: cada 20.000 training steps.
- <u>Data augmentation</u>: 
    - Volteos aleatorios tanto en el eje X como en el eje Y.
    - Rotaciones aleatorias en múltiplos de 90: 0, 90, 180 o 270.

Las métricas empleadas para la medición de la calidad de las reconstrucciones son: PSNR y SSIM, obteniendo unos valores parejos a los otros dos trabajos:

<table style="width:100%">
    <tr>
        <th>Arquitectura</th>
        <th>Métrica</th>
        <th>Set5</th>
        <th>Set14</th>
        <th>Urban100</th>
        <th>Manga109</th>
    </tr>
    <tr>
        <th rowspan='2'>RCAN (Original)</th>
        <th>PSNR</th>
        <th>34,74</th>
        <th>30,65</th>
        <th>29,09</th>
        <th>34,44</th>
    </tr>
    <tr>
        <th>SSIM</th>
        <th>0,9299</th>
        <th>0,8482</th>
        <th>0,8702</th>
        <th>0,9499</th>
    </tr>
    <tr>
        <th rowspan='2'>RCAN (TF 1.13)</th>
        <th>PSNR</th>
        <th>34,75</th>
        <th>30,61</th>
        <th>29,03</th>
        <th>34,49</th>
    </tr>
    <tr>
        <th>SSIM</th>
        <th>0,9302</th>
        <th>0,8470</th>
        <th>0,8693</th>
        <th>0,9500</th>
    </tr>
    <tr>
        <th rowspan='2'>RCAN (TF 2.4 - nuestro)</th>
        <th>PSNR</th>
        <th>34,68</th>
        <th>30,31</th>
        <th>28,90</th>
        <th>34,31</th>
    </tr>
    <tr>
        <th>SSIM</th>
        <th>0,9300</th>
        <th>0,8474</th>
        <th>0,8679</th>
        <th>0,9494</th>
    </tr>
</table>

### <u>Requisitos</u>

Los módulos necesarios para ejecutar el proyecto son los que se indican a continuación:

<font face='Courier New'>

- python == 3.7.9
- tensorflow-gpu == 2.4
- open-cv == 4.4.0
- pandas == 1.1.4
- numpy == 1.18.3
- scikit-learn == 0.23.2
- imgaug == 0.4.0
- pillow (PIL) == 8.0.1

</font>

## <u>Distribución del proyecto</u>

El proyecto se encuentra estructurado en las siguientes carpetas:

### <u>data</u>

En este directorio se encuentran todos los datasets empleados en el entrenamiento y testeo de las redes elaboradas, además de otros datasets no utilizados pero también disponibles (General-100, 91-images, BSDS200 y openImages_light).

Cada dataset posee su propio directorio en el cual se encuentran todas sus imágenes.

Todos estos datasets poseen una copia en menor resolución (factor de reducción igual a 3) en la carpeta 'data/lr_images/'. Estas imágenes fueron generadas a través de MATLAB (interpolación bicúbica) y ubicadas en esa dirección de forma manual. Por cada dataset que se almacene, se deberá añadir a 'data/lr_images/' su versión en baja resolución.

En la carpeta 'data/0_csvs/' se podrá encontrar un fichero .csv por cada dataset disponible. Estos ficheros contienen una única columna con identificador 'path' e indica la ruta hacia cada una de las imágenes del dataset, partiendo siempre desde el directorio 'data/'.

Se incluye además un script, a2_create_csv_files.py, que, al ser ejecutado, crea un fichero csv para cada dataset (directorio en la carpeta 'data' que no se llame '0_csvs' o 'lr_images') y lo almacena en 'data/0_csvs/'.

### <u>lib</u>

Directorio donde se almacenan los módulos Python implementados para el desarrollo del proyecto. Éstos son:
- <u>constants</u>: en este módulo se definen una serie de valores constantes o identificadores que son posteriormente utilizados por el resto de módulos.
- <u>custom_callbacks</u>: en este módulo se encuentran los Callbacks y funciones que han sido necesarias implementar para realizar el seguimiento de los entrenamientos.
- <u>custom_loss_functions</u>: contiene las implementaciones de las distintas <i>loss functions</i> disponibles para entrenar (mae, mse y sobel_loss). Incluye además una función, 'get_mix_loss_function', que permite combinar de forma ponderada dos <i>loss functions</i> en una sola, y otra función que permite seleccionar una <i>loss function</i> a partir de su identificador en el módulo 'constants'.
- <u>custom_metric_functions</u>: contiene las implementaciones de las distintas <i>loss functions</i> disponibles para entrenar (mae, mse y sobel_loss). Incluye además una función que permite seleccionar una <i>metric function</i> a partir de su identificador en el módulo 'constants'.
- <u>data_processing_functions</u>: contiene las funciones de mapeado necesarias para la carga de los datasets en memoria y su transformaciones.
- <u>PrepareDataset</u>: implementa la clase PrepareDataset, que permite realizar la carga de los datasets de una forma más sencilla al automatizar el proceso mediante parámetros en su constructor. Espera recibir inicialmente los datasets como un único fichero .csv (como los almacenados en 'data/0_csvs/'). Este csv contendrá las rutas hacia las imágenes en alta resolución y, a partir de éstas, y conociendo que las imágenes en baja resolución deben estar almacenadas en 'data/lr_images/', generará las rutas hacia las imágenes de baja resolución.
- <u>psnr_ssim</u>: implementa las métricas PSNR y SSIM correctas para la evaluación. Las proporcionadas por Tensorflow son útiles para entrenar, pero no proporcionan los resultados correctos que permitan comparar el rendimiento de la red con el de otras redes en la literatura.
- <u>RCAN</u>: implementación de la arquitectura RCAN, basada en su mayoría en la implementación hecha para Tensorflow 1.13. Contiene la función 'get_RCAN', que genera una red con el número de grupos y bloques residuales que se especifiquen por parámetros. Además, permite especificar el modo en el que la red se entrenará, es decir, lo que ocurre en cada <i>training step</i>.
- <u>RCAN_inception y RCAN_lkrelu</u>: implementaciones de pequeñas modificaciones sobre la arquitectura original que no mostraron diferencia en los resultados (en pruebas sobre redes de menor tamaño, las RCAN-SCALE3-SHORT).
- <u>training_loops</u>: contiene la definición de los bucles de entrenamiento para las redes RCAN. Estos bucles se definen como clases que heredan de 'keras.Model' y se sobreescriben únicamente los métodos 'train_step' y 'test_step', especificando la forma en la que se desee que se calcule y optimice el valor del <i>loss</i>. Al implementarse de esta manera y no como un bucle a secas, se aprovechan las optimizaciones de código que Tensorflow aplica de forma interna.

### <u>TRAINED_MODELS_NEW</u>

En este directorio se almacenan los ficheros de configuración de las redes entrenadas. Éstos son los ficheros que contienen la estructura de la red (.json) y los que contienen los valores de los pesos de las redes (.h5).

Habrá tantas carpetas como estructuras de redes se hayan probado en entrenamientos. La forma de definir si una red pertenece a una estructura u otra se realiza a partir del nombre del script o notebook donde se implementa el entrenamiento. Los nombres de estos archivos seguirán el siguiente formato: identificador numérico, 'train', estructura de la red, características del entrenamiento. Estos cuatro atributos deberán estar separados por '\_', siendo siempre los tres primeros elementos el identificador, 'train' y la estructura. En las 'características del entrenamiento' pueden haber más caracteres '\_'. Un ejemplo de esta nomenclatura es:

    nb05_train_rcan-scale3_loss-computed-on-lr.ipynb

En cada carpeta correspondiente a cada estructura definida, habrá tantos ficheros .json como configuraciones de entrenamiento probadas por cada estructura. Además, por cada entrenamiento habrá una serie de ficheros .h5 que contendrán las configuraciones de pesos que mejores valores obtuvieron en validación para las métricas PSNR y SSIM.

### <u>TRAINING_METRIC_EVOLUTIONS_NEW</u>

En este directorio se guardan los ficheros .csv generados por el Callback 'Save_Training_Evolution' y contienen la evolución de todas las métricas en cada entrenamiento.

Mediante el script see_metrics_evolution_in_training.py se podrá visualizar gráficamente la evolución de los entrenamientos, además de poder comparar varios entrenamientos entre sí.

### <u>entrenamientos_anteriores</u>

En esta carpeta se guardan los scripts que se emplearon originariamente para realizar los primeros entrenamientos en los que la estructura de la red no estaba completamente pulida o los hiper-parámetros del entrenamiento no eran iguales a los de los trabajos mencionados.

Las redes resultantes de estos entrenamientos se pueden ver en el directorio 'TRAINED_MODELS_NEW/', en las estructuras 'RCAN-BASE-SCALE3' y 'RCAN-SCALE3-SHORT'.

Estos scripts no han sido reestructurados (se eliminó y reorganizó gran parte del código) y, por tanto, no pueden ser ejecutados porque darán error de importación o asignación de parámetros.

### <u>superresolved_datasets</u>

En este directorio se almacenan las imágenes de los datasets (una carpeta por dataset) reconstruidas (a partir de las imágenes en baja resolución) por la red que se esté evaluando una vez haya sido entrenada.

## <u>Notebooks y scripts</u>

### <u>Script see_metrics_evolution_in_training.py</u>

Este script, mencionado anteriormente, permite visualizar y/o comparar la evolución que muestra un entrenamiento, ya sea mientras éste se ejecuta o una vez finalizado, mediante la consulta del fichero .csv que se va generando conforme el entrenamiento progresa (para que se genere dicho .csv es necesario que en el entrenamiento se incluya el Callback Save_Training_Evolution).

Para más información acerca de los parámetros de este script, consultar su descripción inicial o ejecutar:

    python see_metrics_evolution_in_training.py -h

### <u>Notebooks nb04 y nb04b</u>

Estos dos archivos contienen la definición del entrenamiendo de la red RCAN propuestas. La diferencia entre ambos se trata de que, al producirse un corte en el suministro eléctrico durante el entrenamiento, se creó un nuevo notebook desde el cual se reanudó el entrenamiento, modificando los parámtros correspondientes a la duración e inicialización del mismo.

En el entrenamiento base, se realizan los siguientes pasos:

- Se construyen unos datasets de entrenamiento y validación empleando los ficheros .csv correspondientes a los datasets Train y Val de DIV2K. Se establece un <i>batch size</i> de 16 y un tamaño de recorte de 48x48 en el dataset de entrenamiento (se aplican recortes a las imágenes con el objetivo de reducir la carga de trabajo de la GPU en el entrenamiento y además, aumentar virtualmente, aparte del data-augmentation, el número de imágenes para entrenar). En el dataset de validación no se aplica ningún recorte y el <i>batch size</i> es igual a 1.
- Se inicializa la red RCAN con el mismo número de grupos y bloques residuales que se indica en la implementación original.
- Se establecen los demás parámetros del entrenamiento:
    - Optimizador Adam.
    - Número de épocas acorde para que se realicen 1.000.000 de <i>training steps</i>.
    - Asignación del <i>learning rate</i> inicial a 1e-4, con una reducción en 1/2 cada 200.000 <i>training steps</i>.

### <u>Notebooks nb05 y nb05b</u>

Estos dos notebooks poseen una estructura idéntica a los que se acaban de mencionar, únicamente cambiando la creación/inicialización de la red RCAN. En estos entrenamientos se aplican otros bucles de entrenamiento que hacen que la <i>loss function</i> se pueda calcular sobre sólo la imagen en baja resolución, o sobre imagen de baja y alta resolución simultáneamente.

### <u>Notebook testing</u>

Este notebook permite evaluar sobre los datasets de test disponibles el rendimiento de la red seleccionada.

La evaluación se realiza con las métricas PSNR y SSIM implementadas en el módulo 'psnr_ssim.py' ya que tienen un funcionamiento más preciso.