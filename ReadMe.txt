I.Using Libraries

The following Python libraries will be used in the study and these libraries will be imported into the project. Here are the libraries and explanations we will use:

1.NumPy : This library is actually a dependency for other libraries. The main purpose of this library is to provide a variety of mathematical operations on matrices and vectors in Python. Our project will be used this library to provide support to other libraries.
2.Pandas : This library performs import and processing of dataset in Python. In our project, it will be used to include the CSV extension dataset in the project and to perform various operations on it.
3.Matplotlib : This library, which is usually used to visualize data. It will perform the same task in our project.
4.Seaborn : This library which has similar features to Matplotlib is another library used for data visualization in Python. In our project, it will be used for the implementation of various features not included in the Matplotlib library.

II.Importing the Libraries -

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

III. In addition the following libraries were for the Feature engineering and Machine Learing part:

Importing the Libraries -

from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import learning_curve, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import utils
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix 
from sklearn import tree
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

IV. To create the virtual environment -

The Anaconda prompt was used to create the virtual environment 'classroom' as follows:

conda create -n classroom python=3.7
conda activate classroom

V. Installations

The environment 'classroom' created will be isibleon the left hand side of the Anaconda prompt. Then run the commands:
'conda install sklearn' or 'pip install sklearn' and Anaconda will automatically download the packages and any dependicies respectively.

VI. Running in Jupyter Notebook

Once the above steps are carried out select your virtual environment from the drop down and install Jupyter Notebook in the environment if not already installed. 