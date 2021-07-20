# Heart_attack
Heart attack analisis
#Import
  
import pandas as pd
import plotly.express as px 
import plotly.graph_objects as go
!pip install catboost
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
  
