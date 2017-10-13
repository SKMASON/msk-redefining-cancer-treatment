# use LSTM result to stack with XGB by using gene frequency
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import *
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import log_loss
#########################################
# input data
#########################################
# Get train_y
train_variant = pd.read_csv("training_variants")
test_variant = pd.read_csv("test_variants")
train_text = pd.read_csv("training_text", sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
test_text = pd.read_csv("test_text", sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
train = pd.merge(train_variant, train_text, how='left', on='ID')
train_y = train['Class'].values
label = open("stage1_solution.csv")
lines = label.readlines()[1:]
label = []
for line in lines:
    main = line.strip().split(',')
    main = main[1:10]
    label.append(main.index('1') + 1)

label = np.array(label)
train_y = np.concatenate((train_y, label), axis=0)

# Get train_x
train_text_x = pd.read_csv("TrainForXGB.csv")
train_Gene_x = pd.read_csv("Train_Gene.csv")
train_x = pd.merge(train_text_x, train_Gene_x, how='left', on='ID')
ID_train = train_text_x['ID']
train_x = train_x.drop(['ID'], axis=1)


# Get pred_x
pred_text_x = pd.read_csv("TestForXGB.csv")
pred_Gene_x = pd.read_csv("Train_Gene.csv")
pred_x = pd.merge(pred_text_x, pred_Gene_x, how='left', on='ID')
ID_pred = pred_x['ID']
pred_x = pred_x.drop(['ID'], axis=1)

#########################################
# XGB
#########################################
xgb_model = xgb.XGBClassifier(
               objective = 'multi:softprob',
               silent    = True,
               nthread   = 4,
             )
xgb_params = {
              'learning_rate': [0.008], 
              'max_depth': [4],
              'subsample': [0.85],
              'colsample_bytree': [0.8],
              'n_estimators': [600], 
              'seed': [2017]
              }

clf = GridSearchCV(xgb_model, xgb_params, n_jobs=1, cv=10, verbose=50, refit=True, scoring='log_loss')
clf.fit(train_x, train_y)

# best params
best_parameters, score, _ = max(clf.grid_scores_, key=lambda x: x[1])
print('score:', score)
for param_name in sorted(best_parameters.keys()):
     print("%s: %r" % (param_name, best_parameters[param_name]))


#predict
class_pred = clf.predict_proba(pred_x)
submission   = pd.DataFrame(class_pred)
ID_test      = pd.DataFrame(ID_pred)
pred         = ID_test.join(submission)
pred.columns = ['ID','class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7', 'class8', 'class9']
pred.to_csv("XGBResultTemp.csv",index=False)

#########################################
# Merge with Overlapped Data
#########################################
from subprocess import check_output
stage_1_test = pd.read_csv(filepath_or_buffer = "test_variants")
stage_2_test = pd.read_csv(filepath_or_buffer = "stage2_test_variants.csv")
result = stage_1_test.merge(stage_2_test, on = ["Gene", "Variation"])
y = pd.read_csv("stage1_solution_all.csv")
id_y = list(y["ID"])
y = y.drop("ID",axis = 1)
y = np.array(y)
id_1 = list(result['ID_x'])
id_2 = list(result['ID_y'])
LSTM_result = pd.read_csv("XGBResultTemp.csv")
LSTM_result = LSTM_result.drop("ID", axis = 1)
LSTM_result = np.array(LSTM_result)
a = []
for i in range(1,987):
    if i in id_2:
        location = id_2.index(i)
        marker = id_1[location]
        if marker in id_y:
            marker_2 = id_y.index(marker)
            a.append(list(y[marker_2]))
    else:
        a.append(list(LSTM_result[i-1]))


sub = pd.DataFrame(a)
sub.columns = ['class'+str(c+1) for c in range(9)]
sub["ID"] = range(1,987)
sub.to_csv('XGBResult.csv', index=False)
