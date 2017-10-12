# adjust the label which is not necessary
import pandas as pd
import numpy as np

test = pd.read_csv("XGBResult.csv")
temp = []
for i in range(len(test)):
	freq = [test['class1'][i],test['class2'][i],test['class3'][i],test['class4'][i],test['class5'][i],test['class6'][i],test['class7'][i],test['class8'][i],test['class9'][i]]
	freq = np.array(freq)
	if freq[3] == np.max(freq):
		freq_2_temp = freq[1]
		freq_4_temp = freq[3]
		freq[1] = freq_4_temp
		freq[3] = freq_2_temp
	elif freq[6] == np.max(freq):
		freq_7_temp = freq[6]
		freq_8_temp = freq[7]
		freq[6] = freq_8_temp
		freq[7] = freq_7_temp
	freq = list(freq)
	temp.append(freq)

output = pd.DataFrame(temp)
output['ID'] = range(1,987)
output.columns = ['class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7', 'class8', 'class9', 'ID']
output.to_csv("final_submission_temp",index=False)


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
LSTM_result = pd.read_csv("final_submission_temp")
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
sub.to_csv('submission.csv', index=False)