# Heart attack dataset
Heart attack analisis
# Import
```python
import pandas as pd
import plotly.express as px 
import plotly.graph_objects as go
!pip install catboost
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
``` 
# Data initialization
```python
data = pd.read_csv('/content/drive/MyDrive/heart_attack/heart.csv')
```
```python
data
```
```
age	sex	cp	trtbps	chol	fbs	restecg	thalachh	exng	oldpeak	slp	caa	thall	output
0	63	1	3	145	233	1	0	150	0	2.3	0	0	1	1
1	37	1	2	130	250	0	1	187	0	3.5	0	0	2	1
2	41	0	1	130	204	0	0	172	0	1.4	2	0	2	1
3	56	1	1	120	236	0	1	178	0	0.8	2	0	2	1
4	57	0	0	120	354	0	1	163	1	0.6	2	0	2	1
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
298	57	0	0	140	241	0	1	123	1	0.2	1	0	3	0
299	45	1	3	110	264	0	1	132	0	1.2	1	0	3	0
300	68	1	0	144	193	1	1	141	0	3.4	1	2	3	0
301	57	1	0	130	131	0	1	115	1	1.2	1	1	3	0
302	57	0	1	130	236	0	0	174	0	0.0	1	1	2	0
303 rows × 14 columns
```
# Vizualization
Vizualization of distribution of age in dataset
![image](https://user-images.githubusercontent.com/35808072/126371259-4f816417-54ac-4326-b0d8-f51b6d79ab87.png)

