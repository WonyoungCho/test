import pandas as pd
import numpy as np

#############[ BASIC ]#############
a = list(range(10))
print(a*20)

b = np.arange(10)
print(b*20)

for i in b:
    print(i)

#############[ DATASET ]#############
fname = 'iris.txt'
df = pd.read_csv(fname,sep='\t')
df = pd.read_table(fname)#index_col=0)
print(df)

X = df.iloc[:,:2]
y = df.target

df.to_csv('iris_re.txt',sep='\t',index=None)
#############[ PLOT ]#############
import matplotlib.pyplot as plt
import seaborn as sns

sns.scatterplot(X.iloc[:,0], X.iloc[:, 1],hue=y,palette="deep")
plt.tight_layout()
plt.show()

sns.barplot(x=X.index,y=X.iloc[:,1])
plt.tight_layout()
plt.show()

