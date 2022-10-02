import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#############[ BASIC ]#############
def basic():
    a= list(range(10))
    print(a*20)
    
    b = np.arange(10)
    print(b*20)

    for i in b:
        print(i)
        
#############[ DATASET ]#############
def dataset():
    fname = 'iris.txt'
    df = pd.read_csv(fname,sep='\t')
    df = pd.read_table(fname)#index_col=0)
    print(df)
    
    X = df.iloc[:,:2]
    y = df.target

    df.to_csv('iris_re.txt',sep='\t',index=None)
    return X, y

#############[ PLOT ]#############
def plot_iris(X,y):
    sns.scatterplot(X.iloc[:,0], X.iloc[:, 1],hue=y,palette="deep")
    plt.tight_layout()
    plt.show()
    
    sns.barplot(x=X.index,y=X.iloc[:,1])
    plt.tight_layout()
    plt.show()


def main():
    basic()
    X,y = dataset()
    plot_iris(X,y)

if __name__ == '__main__':
    main()
