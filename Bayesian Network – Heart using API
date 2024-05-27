import pandas as pd
import numpy as np
h=pd.read_csv('heart.csv')
print(h)
h=h.replace('?',np.nan)
from pgmpy.models import BayesianNetwork
model=BayesianNetwork([('age','target'),('sex','target'),('trestbps','target'),('exa
ng','target'),('restecg','target')])
from pgmpy.estimators import MaximumLikelihoodEstimator
model.fit(h,estimator= MaximumLikelihoodEstimator)
from pgmpy.inference import VariableElimination
infer=VariableElimination(model)
q1=infer.query(variables=['target'],evidence={'restecg':2})
print(q1)
