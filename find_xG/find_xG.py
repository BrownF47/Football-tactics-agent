import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import sklearn.linear_model as lm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from plotting_functions.plot_pitch import plot_half_pitch

shots_df = pd.read_parquet('data/shots.parquet')

print(shots_df.head())
shots_df = shots_df[shots_df['X']>0.5]

# 1 = goal, 0 no goal #
encoded_result = (shots_df['result'] == 'Goal').astype(int)

# distance from goal #
shots_df['distance_from_goal_center'] = ((1-shots_df['X'])**2 + (shots_df['Y']-0.5)**2)**0.5
shots_df['angle'] = np.arccos((1-shots_df['X'])/shots_df['distance_from_goal_center'])

plt.scatter(shots_df['distance_from_goal_center'], encoded_result)
plt.show()

plt.scatter(shots_df['angle'], encoded_result)
plt.show()

my_X = shots_df[['distance_from_goal_center','angle']]

# get train and test data # 
x_train, x_test, y_train, y_test = train_test_split(my_X, encoded_result, train_size=0.8, random_state=47)

model = lm.LogisticRegression(class_weight='balanced')
model.fit(x_train, y_train)

xG = model.predict_proba(x_train)[:,1]
print(xG)

# accuracy not a great measure for xG given the imbalance in the data #
model_score = model.score(x_test, y_test)
print(model_score)

# AUC-ROC (Area Under the Receiver Operating Characteristic Curve) is a performance metric for binary classification models.
# Ranging from 0 to 1, a score of 1 represents a perfect classifier, while 0.5 denotes random guessing.
y_prob = model.predict_proba(x_test)[:,1]
auc = roc_auc_score(y_test, y_prob)
print(auc)


# Visualise the xG # 
x = np.linspace(0,1,100)   # only attacking half
y = np.linspace(0,1,100)

xx, yy = np.meshgrid(x,y)

goal_x = 1
goal_y = 0.5

distance = np.sqrt((goal_x - xx)**2 + (goal_y - yy)**2)
angle = np.arctan2(abs(goal_y - yy), goal_x - xx)

grid_features = np.column_stack([distance.ravel(), angle.ravel()])

grid_features = pd.DataFrame(
    grid_features,
    columns=['distance_from_goal_center', 'angle']
)
xg_surface = model.predict_proba(grid_features)[:,1]

xg_surface = xg_surface.reshape(xx.shape)

plt.figure(figsize=(8,6))

plt.imshow(
    xg_surface,
    extent=(0,50,0,70),
    origin='lower',
    cmap='viridis'
)

plot_half_pitch()

plt.colorbar(label="xG Probability")

plt.title("Expected Goals Surface")
plt.savefig('find_xG/xG_surface.png', dpi=150, bbox_inches='tight')

plt.show()