#! wget https://raw.githubusercontent.com/dataprofessor/data/master/delaney.csv

!pip install rdkit pandas scikit-learn

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


def generate(smiles, verbose=False):
    moldata= []
    for elem in smiles:
        mol=Chem.MolFromSmiles(elem) 
        moldata.append(mol)
       
    baseData= np.arange(1,1)
    i=0  
    for mol in moldata:        
        desc_MolLogP = Descriptors.MolLogP(mol)
        desc_MolWt = Descriptors.MolWt(mol)
        desc_NumRotatableBonds = Descriptors.NumRotatableBonds(mol)
           
        row = np.array([desc_MolLogP,
                        desc_MolWt,
                        desc_NumRotatableBonds])   
    
        if(i==0):
            baseData=row
        else:
            baseData=np.vstack([baseData, row])
        i=i+1      
    
    columnNames=["MolLogP","MolWt","NumRotatableBonds"]   
    descriptors = pd.DataFrame(data=baseData,columns=columnNames)
    
    return descriptors


def AromaticAtoms(m):
  aromatic_atoms = [m.GetAtomWithIdx(i).GetIsAromatic() for i in range(m.GetNumAtoms())]
  aa_count = []
  for i in aromatic_atoms:
    if i==True:
      aa_count.append(1)
  sum_aa_count = sum(aa_count)
  return sum_aa_count

def calculate_instances(sol):
  df = generate(sol.SMILES)


  desc_AromaticAtoms = [AromaticAtoms(element) for element in mol_list]
  desc_HeavyAtomCount = [Descriptors.HeavyAtomCount(element) for element in mol_list]
  desc_AromaticProportion = [AromaticAtoms(element)/Descriptors.HeavyAtomCount(element) for element in mol_list]

  df_desc_AromaticProportion = pd.DataFrame(desc_AromaticProportion)

  X = pd.concat([df,df_desc_AromaticProportion], axis=1)
  Y = sol.iloc[:,1]
  return X,Y

def plots(Y_train,Y_pred_train, Y_test, Y_pred_test):
  plt.figure(figsize=(5,11))

  # 2 row, 1 column, plot 1
  plt.subplot(2, 1, 1)
  plt.scatter(x=Y_train, y=Y_pred_train, c="#7CAE00", alpha=0.3)

  # Add trendline
  # https://stackoverflow.com/questions/26447191/how-to-add-trendline-in-python-matplotlib-dot-scatter-graphs
  z = np.polyfit(Y_train, Y_pred_train, 1)
  p = np.poly1d(z)
  plt.plot(Y_test,p(Y_test),"#F8766D")

  plt.ylabel('Predicted LogS')


  # 2 row, 1 column, plot 2
  plt.subplot(2, 1, 2)
  plt.scatter(x=Y_test, y=Y_pred_test, c="#619CFF", alpha=0.3)

  z = np.polyfit(Y_test, Y_pred_test, 1)
  p = np.poly1d(z)
  plt.plot(Y_test,p(Y_test),"#F8766D")

  plt.ylabel('Predicted LogS')
  plt.xlabel('Experimental LogS')

  plt.savefig('plot_vertical_logS.png')
  plt.savefig('plot_vertical_logS.pdf')
  plt.show()

def model_performances(model, X, Y):
  Y_pred = model.predict(X)
  print(3*" ")
  print('Coefficients:', model.coef_)
  print('Intercept:', model.intercept_)
  print('Mean squared error (MSE): %.2f'
        % mean_squared_error(Y, Y_pred))
  print('Coefficient of determination (R^2): %.2f'
        % r2_score(Y, Y_pred))
  return Y_pred

def print_intercept(model):
  print(3*"")
  print("equation")
  yintercept = '%.2f' % model.intercept_
  LogP = '%.2f LogP' % model.coef_[0]
  MW = '%.4f MW' % model.coef_[1]
  RB = '%.4f RB' % model.coef_[2]
  AP = '%.2f AP' % model.coef_[3]
  print('LogS = ' + 
        ' ' + 
        yintercept + 
        ' ' + 
        LogP + 
        ' ' + 
        MW + 
        ' ' + 
        RB + 
        ' ' + 
        AP)



##################
sol = pd.read_csv('delaney.csv')
sol.head()

sol.SMILES




mol_list= []
for element in sol.SMILES:
  mol = Chem.MolFromSmiles(element)
  mol_list.append(mol)



X, Y = calculate_instances(sol)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.2)



model = linear_model.LinearRegression()
model.fit(X_train, Y_train)

#4. Results
#4.1. Linear Regression Model
#4.1.1. Predict the LogS value of X_train and X_test data

Y_pred_train = model_performances(model, X_train, Y_train)
Y_pred_test = model_performances(model, X_test, Y_test)


print_intercept(model)

print(10*"")
plots(Y_train,Y_pred_train, Y_test, Y_pred_test)






