# generate the gene frequency for each sample
import pandas as pd
import numpy as np

train_variant = pd.read_csv("training_variants")
test_variant  = pd.read_csv("test_variants")

label = open("stage1_solution.csv")
lines = label.readlines()[1:]
label = []
for line in lines:
    main = line.strip().split(',')
    main = main[1:10]
    label.append(main.index('1') + 1)

label = np.array(label)
ID = pd.DataFrame(pd.read_csv("stage1_solution.csv")["ID"])
test_variant = pd.merge(test_variant, ID, how='right', on='ID')
test_variant["Class"] = label

df    = pd.DataFrame(np.concatenate((train_variant, test_variant), axis=0))
df.columns = train_variant.columns
dfTest= df

def ComputeProbabilities(df):
    Pclass=[0,0,0,0,0,0,0,0,0,0]
    Nclass=[0,0,0,0,0,0,0,0,0,0]
    vc = df['Class'].value_counts()
    N = vc.size
    countTotal=0
    for i in range(N):
        cl = vc.index[i]
        count = vc.values[i]
        countTotal=countTotal+count
        Nclass[cl]=count
    for icl in range(10):
        Pclass[icl]=float(Nclass[icl])/float(countTotal)
    return Pclass

Pbase=ComputeProbabilities(df)

Ntest = dfTest['ID'].count()

strSubmColumns="ID,class1,class2,class3,class4,class5,class6,class7,class8,class9".split(',')
dfSubm = pd.DataFrame(columns=strSubmColumns)

Parr=Pbase
for i in range(Ntest):
    ID=int(dfTest.loc[i]['ID'])
    strGene = dfTest.loc[i]['Gene']
    if (df[df['Gene']==strGene]['ID'].count()==0): pass
    else:
        Parr=ComputeProbabilities(df[df['Gene']==strGene])
    dfSubm.loc[i] = pd.Series({'ID':ID,'class1':Parr[1],'class2':Parr[2],'class3':Parr[3],
                               'class4':Parr[4],'class5':Parr[5],'class6':Parr[6],
                               'class7':Parr[7],'class8':Parr[8],'class9':Parr[9]})	

dfSubm['ID']=dfSubm['ID'].apply(lambda n: int(n))
dfSubm.to_csv('Test_Gene.csv', index=False)


##################################################################################
import pandas as pd
import numpy as np
df = pd.read_csv("training_variants")
dfTest= df

def ComputeProbabilities(df):
    Pclass=[0,0,0,0,0,0,0,0,0,0]
    Nclass=[0,0,0,0,0,0,0,0,0,0]
    vc = df['Class'].value_counts()
    N = vc.size
    countTotal=0
    for i in range(N):
        cl = vc.index[i]
        count = vc.values[i]
        countTotal=countTotal+count
        Nclass[cl]=count
    for icl in range(10):
        Pclass[icl]=float(Nclass[icl])/float(countTotal)
    return Pclass

Pbase=ComputeProbabilities(df)

Ntest = dfTest['ID'].count()


strSubmColumns="ID,class1,class2,class3,class4,class5,class6,class7,class8,class9".split(',')
dfSubm = pd.DataFrame(columns=strSubmColumns)

Parr=Pbase
for i in range(Ntest):
    ID=int(dfTest.loc[i]['ID'])
    strGene = dfTest.loc[i]['Gene']
    if (df[df['Gene']==strGene]['ID'].count()==0): pass
    else:
        Parr=ComputeProbabilities(df[df['Gene']==strGene])
    dfSubm.loc[i] = pd.Series({'ID':ID,'class1':Parr[1],'class2':Parr[2],'class3':Parr[3],
                               'class4':Parr[4],'class5':Parr[5],'class6':Parr[6],
                               'class7':Parr[7],'class8':Parr[8],'class9':Parr[9]}) 

dfSubm['ID']=dfSubm['ID'].apply(lambda n: int(n))
dfSubm.to_csv('Train_Gene.csv', index=False)



















