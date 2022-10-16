import pandas as pd
import numpy as np

import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             average_precision_score, f1_score,
                             precision_score, recall_score)


df = pd.read_table('test.tsv',index_col=0)

Xtrain = df[['bMET','age', 'sex','city','bmi','smoke','income','job','drink']]

formula = 'bMET ~ sex + age + city + bmi + smoke + income + drink'
model = smf.logit(formula, data=Xtrain).fit()
print(model.summary())

odds_ratios = pd.DataFrame(
    {
        "OR": model.params,
        "Lower CI": model.conf_int()[0],
        "Upper CI": model.conf_int()[1],
    }
)

odds_ratios = np.exp(odds_ratios)
odds_ratios.index.name = '2009'

print(odds_ratios.to_string())

AME = model.get_margeff(at = "overall", method = "dydx")
print(AME.summary())
