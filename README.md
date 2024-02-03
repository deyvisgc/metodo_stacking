<!-- # Analisis de sentimiento aplicando el metodo Stacking Emsamble
## Subtítulo -->

[![License](https://img.shields.io/badge/License-BSD%202--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)
[![GitHub forks](https://img.shields.io/github/forks/tirthajyoti/Machine-Learning-with-Python.svg)](https://github.com/tirthajyoti/Machine-Learning-with-Python/network)
[![GitHub stars](https://img.shields.io/github/stars/tirthajyoti/Machine-Learning-with-Python.svg)](https://github.com/tirthajyoti/Machine-Learning-with-Python/stargazers)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/tirthajyoti/Machine-Learning-with-Python/pulls)

# Analisis de sentimiento aplicando el metodo Stacking Emsamble

### Bach. Garcia Cercado Deyvis Ronald, Lima, Peru  ([No dude en conectarse en LinkedIn aquí](https://www.linkedin.com/in/deyvis-garcia-a503ba163/))

![ml-ds](https://raw.githubusercontent.com/tirthajyoti/Machine-Learning-with-Python/master/Images/ML-DS-cycle-1.png)

---
Análisis de sentimientos en Twitter con stacking ensemble para el índice de aprobación política de la presidencia del estado peruano. Segmentación geográfica para Hispanoamérica, usando Ciencia de Datos.
## Librerias
* **Python 3.6+**
* **NumPy (`pip install numpy`)**
* **Pandas (`pip install pandas`)**
* **Scikit-learn (`pip install scikit-learn`)**
* **MatplotLib (`pip install matplotlib`)**
* **Seaborn (`pip install seaborn`)**
* **Sympy (`pip install sympy`)**
---

<!-- ## Essential tutorial-type notebooks on Pandas and Numpy
Jupyter notebooks covering a wide range of functions and operations on the topics of NumPy, Pandans, Seaborn, Matplotlib etc.

* [Detailed Numpy operations](https://github.com/tirthajyoti/Machine-Learning-with-Python/blob/master/Pandas%20and%20Numpy/Numpy_operations.ipynb)
* [Detailed Pandas operations](https://github.com/tirthajyoti/Machine-Learning-with-Python/blob/master/Pandas%20and%20Numpy/Pandas_Operations.ipynb)
* [Numpy and Pandas quick basics](https://github.com/tirthajyoti/Machine-Learning-with-Python/blob/master/Pandas%20and%20Numpy/Numpy_Pandas_Quick.ipynb)
* [Matplotlib and Seaborn quick basics](https://github.com/tirthajyoti/Machine-Learning-with-Python/blob/master/Pandas%20and%20Numpy/Matplotlib_Seaborn_basics.ipynb)
* [Advanced Pandas operations](https://github.com/tirthajyoti/Machine-Learning-with-Python/blob/master/Pandas%20and%20Numpy/Advanced%20Pandas%20Operations.ipynb)
* [How to read various data sources](https://github.com/tirthajyoti/Machine-Learning-with-Python/blob/master/Pandas%20and%20Numpy/Read_data_various_sources/How%20to%20read%20various%20sources%20in%20a%20DataFrame.ipynb)
* [PDF reading and table processing demo](https://github.com/tirthajyoti/Machine-Learning-with-Python/blob/master/Pandas%20and%20Numpy/Read_data_various_sources/PDF%20table%20reading%20and%20processing%20demo.ipynb)
* [How fast are Numpy operations compared to pure Python code?](https://github.com/tirthajyoti/Machine-Learning-with-Python/blob/master/Pandas%20and%20Numpy/How%20fast%20are%20NumPy%20ops.ipynb) (Read my [article](https://towardsdatascience.com/why-you-should-forget-for-loop-for-data-science-code-and-embrace-vectorization-696632622d5f) on Medium related to this topic)
* [Fast reading from Numpy using .npy file format](https://github.com/tirthajyoti/Machine-Learning-with-Python/blob/master/Pandas%20and%20Numpy/Numpy_Reading.ipynb) (Read my [article](https://towardsdatascience.com/why-you-should-start-using-npy-file-more-often-df2a13cc0161) on Medium on this topic) -->

<!-- ## Modelos de clasificacion Utilizados

### Regression
* Simple linear regression with t-statistic generation
<img src="https://slideplayer.com/slide/6053182/20/images/10/Simple+Linear+Regression+Model.jpg" width="400" height="300"/>
-----
### Classification
* Logistic regression/classification ([Here is the Notebook](https://github.com/tirthajyoti/Machine-Learning-with-Python/blob/master/Classification/Logistic_Regression_Classification.ipynb))
<img src="https://qph.fs.quoracdn.net/main-qimg-914b29e777e78b44b67246b66a4d6d71"/>

* _k_-nearest neighbor classification ([Here is the Notebook](https://github.com/tirthajyoti/Machine-Learning-with-Python/blob/master/Classification/KNN_Classification.ipynb))

* Decision trees and Random Forest Classification ([Here is the Notebook](https://github.com/tirthajyoti/Machine-Learning-with-Python/blob/master/Classification/DecisionTrees_RandomForest_Classification.ipynb))

* Support vector machine classification ([Here is the Notebook](https://github.com/tirthajyoti/Machine-Learning-with-Python/blob/master/Classification/Support_Vector_Machine_Classification.ipynb)) (**[check the article I wrote in Towards Data Science on SVM and sorting algorithm](https://towardsdatascience.com/how-the-good-old-sorting-algorithm-helps-a-great-machine-learning-technique-9e744020254b))**

<img src="https://docs.opencv.org/2.4/_images/optimal-hyperplane.png"/>

* Naive Bayes classification ([Here is the Notebook](https://github.com/tirthajyoti/Machine-Learning-with-Python/blob/master/Classification/Naive_Bayes_Classification.ipynb))

---

### Clustering
<img src="https://i.ytimg.com/vi/IJt62uaZR-M/maxresdefault.jpg" width="450" height="300"/>

* _K_-means clustering ([Here is the Notebook](https://github.com/tirthajyoti/Machine-Learning-with-Python/blob/master/Clustering-Dimensionality-Reduction/K_Means_Clustering_Practice.ipynb))

* Affinity propagation (showing its time complexity and the effect of damping factor) ([Here is the Notebook](https://github.com/tirthajyoti/Machine-Learning-with-Python/blob/master/Clustering-Dimensionality-Reduction/Affinity_Propagation.ipynb))

* Mean-shift technique (showing its time complexity and the effect of noise on cluster discovery) ([Here is the Notebook](https://github.com/tirthajyoti/Machine-Learning-with-Python/blob/master/Clustering-Dimensionality-Reduction/Mean_Shift_Clustering.ipynb))

* DBSCAN (showing how it can generically detect areas of high density irrespective of cluster shapes, which the k-means fails to do) ([Here is the Notebook](https://github.com/tirthajyoti/Machine-Learning-with-Python/blob/master/Clustering-Dimensionality-Reduction/DBScan_Clustering.ipynb))

* Hierarchical clustering with Dendograms showing how to choose optimal number of clusters ([Here is the Notebook](https://github.com/tirthajyoti/Machine-Learning-with-Python/blob/master/Clustering-Dimensionality-Reduction/Hierarchical_Clustering.ipynb))

<img src="https://www.researchgate.net/profile/Carsten_Walther/publication/273456906/figure/fig3/AS:294866065084419@1447312956501/Example-of-hierarchical-clustering-clusters-are-consecutively-merged-with-the-most.png" width="700" height="400"/>

---

### Dimensionality reduction
* Principal component analysis

<img src="https://i.ytimg.com/vi/QP43Iy-QQWY/maxresdefault.jpg" width="450" height="300"/>

---

### Synthetic data generation techniques
* [Notebooks here](https://github.com/tirthajyoti/Machine-Learning-with-Python/tree/master/Synthetic_data_generation) -->