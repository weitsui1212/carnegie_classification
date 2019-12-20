'''
Created on Dec 5, 2019

@author: wei-hsuan.tsui
'''

import pandas as pd
import numpy as np
df=pd.read_csv('CCIHE2018.csv',encoding='latin-1')
print(df.columns)

df = df[df['BASIC2018'].isin([15,16,17])] #Doctoral
df = df[df['UNITID']!=185828] ##New Jersey Institute of Technology
df = df[df['UNITID']!=204857] ##Ohio University-Main Campus
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df.columns=df.columns.str.replace('&','_')
df.columns=df.columns.str.replace(' ','')
df.fillna(0,inplace=True)
print(df.columns)
##& df['UNITID']!=185828]
##df.query('BASIC2018 ==[15,16,17] & UNITID!=185828')

df=df.replace(to_replace='.*NotIn/*',value='', regex=True)
df=df.replace(to_replace='.*Notin/*',value='', regex=True)
#df.S_ER_D[df['UNITID']==185828]

df['S_ER_D']=df['S_ER_D'].astype(int)
df['NONS_ER_D']=df['NONS_ER_D'].astype(int)
df['PDNFRSTAFF']=df['PDNFRSTAFF'].astype(int)
df['FACNUM']=df['FACNUM'].astype(int)
df['SOCSC_RSD']=df['SOCSC_RSD'].astype(int)
df['STEM_RSD']=df['STEM_RSD'].astype(int)
df['HUM_RSD']=df['HUM_RSD'].astype(int)

df.fillna(0,inplace=True)
##df['HUM_RSD'].head(10)

def division(n,d):
    try:
        return n/d
    except ZeroDivisionError:
        return 0

##df['Per_Cap_SE_Exp']=df.S_ER_D/df.FACNUM
##df['Per_Cap_Non_SE']=df.NONS_ER_D/df.FACNUM
##df['Per_Cap_Prof_Staf']=df.PDNFRSTAFF/df.FACNUM

df['Per_Cap_SE_Exp']=division(df['S_ER_D'],df['FACNUM'])
df['Per_Cap_Non_SE']=division(df['NONS_ER_D'],df['FACNUM'])
df['Per_Cap_Prof_Staf']=division(df['PDNFRSTAFF'],df['FACNUM'])

df.fillna(0,inplace=True)

aggregate=['HUM_RSD','NONS_ER_D','OTHER_RSD','PDNFRSTAFF','S_ER_D','SOCSC_RSD','STEM_RSD']
perCapita=['Per_Cap_SE_Exp','Per_Cap_Non_SE','Per_Cap_Prof_Staf']

from sklearn.preprocessing import StandardScaler

#Normalization
features = df[perCapita]
scaler = StandardScaler().fit(features.values)
features = scaler.transform(features.values)
df[perCapita] = features


#Normalization
features = df[aggregate]
scaler = StandardScaler().fit(features.values)
features = scaler.transform(features.values)
df[aggregate] = features

from sklearn.decomposition import PCA
pca = PCA(n_components=1)
principalComponents = pca.fit_transform(df[['HUM_RSD','NONS_ER_D','OTHER_RSD','PDNFRSTAFF','S_ER_D','SOCSC_RSD','STEM_RSD']])
##principalDf = pd.DataFrame(data = principalComponents
##columns = ['principal component 1'])
df.insert(0,'PrincipalComponents1',principalComponents)

pca = PCA(n_components=1)
principalComponents1 = pca.fit_transform(df[['Per_Cap_SE_Exp','Per_Cap_Non_SE','Per_Cap_Prof_Staf']])
##principalDf2 = pd.DataFrame(data = principalComponents
##columns = ['principal component 2'])
df.insert(0,'PrincipalComponents2',principalComponents1)

from scipy.stats import zscore
df['PrincipalComponents1']=zscore(df['PrincipalComponents1'])
df['PrincipalComponents2']=zscore(df['PrincipalComponents2'])

import matplotlib.pyplot as plt

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 

ax.set_xlabel('Principal Component 1', fontsize = 15)
##ax.set_xlim([0, 10])

ax.set_ylabel('Principal Component 2', fontsize = 15)
##ax.set_ylim([0, 10])

ax.set_title('2 component PCA', fontsize = 20)
targets = [15, 16, 17]
colors = ['r', 'g', 'b']


for target, color in zip(targets,colors):
    indicesToKeep = df['BASIC2018'] == target
    x=df.loc[indicesToKeep, 'PrincipalComponents1']
    y=df.loc[indicesToKeep, 'PrincipalComponents2']
# t=df.loc[indicesToKeep, 'NAME']
    
    ax.scatter(x, y, c = color, s = 50)
#ax.text(x,y)#,t)


ax.annotate('Nova Southeastern University', xy=(1.4, -.5), xytext=(1.8, 0),
            arrowprops=dict(facecolor='black', shrink=0.01),
            )    
ax.annotate('California Institute of Technology', xy=(0.5, 6.4), xytext=(0.8, 6.8),
            arrowprops=dict(facecolor='black', shrink=0.01),
            ) 
ax.annotate('Rockefeller University', xy=(-0.2, 12.1), xytext=(0.3, 11.6),
            arrowprops=dict(facecolor='black', shrink=0.01),
            ) 
ax.annotate('Havard University', xy=(5.6, 6.1), xytext=(5, 6.9),
            arrowprops=dict(facecolor='black', shrink=0.01),
            )  
ax.annotate('American University', xy=(0.1, 0.9), xytext=(0.7, 1.5),
            arrowprops=dict(facecolor='black', shrink=0.01),
            )   
#theta = np.arange(0, np.pi / 2, 0.01)
#ax.plot(0.6 * np.cos(theta), 0.6 * np.sin(theta))

ax.legend(targets)
ax.grid()
plt.show()
