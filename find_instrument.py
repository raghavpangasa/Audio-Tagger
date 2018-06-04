#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 22:53:04 2018

@author: somyagoel
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 19:28:17 2018

@author: somyagoel
"""
import pandas as pd
import math

def find_instrument(arg1, para1, para2, para3, para4) :

  print(arg1)
  print(para1)
  print(para2)
  print(para3)
  print(para4)

 # dataset = pd.DataFrame(pd.read_csv('/Users/somyagoel/nsynth_add_labels/test/test_st/guitar_acoustic_030-061-127.wav_st.csv',low_memory=False))
  dataset = pd.DataFrame(pd.read_csv(arg1,low_memory=False))
  dataset = dataset.loc[:, ~dataset.columns.str.contains('^Unnamed')]
  dataset['instrument'] = 'Inst'
  #dataset.drop(['instrument'], axis = 1, inplace = True)
  #dataset.drop(['velocity'], axis = 1, inplace = True)
  #dataset.drop(['pitch'], axis = 1, inplace = True)

  X = dataset
#X.as_type(float)
#print(X)
  y = pd.DataFrame(X['instrument'])
#print(y)
  X.drop(['instrument'], axis = 1, inplace = True)
#print(X)

  w = 'ins'
  z = []
  for i in range(len(y)):
      z.append(w + str(i+1))
    
  df = pd.concat([y,X], axis = 1)             #fam1,fam2 ki rows 
#print(df)
  df['instrument'] = z
#print(df)
  centroid = pd.DataFrame(pd.read_csv(para4,low_memory=False))
  centroid = centroid.loc[:, ~centroid.columns.str.contains('^Unnamed')]
  centroid = centroid.iloc[:, 0:35]
  #centroid.columns = df.columns

  family_result = []
  maha_distances_per_row =[]

  for row in df.iterrows():
      print(row)
      x = pd.DataFrame(row[1]).transpose()
  #  x.as_type(float)
   # print(x)

      data = pd.concat([x,centroid],ignore_index=True)  #fam wali row ko baki 3 type ki row ke saath concat
  #  data.columns = df.columns    
    #data.reset_index(drop=True)
      data=pd.DataFrame(data)
      f = data['instrument']
      data.drop(['instrument'],axis = 1, inplace = True)
      data[:] = data[ : ].astype('float64')
      data['instrument'] = f
    
  #  cols = list(data.columns)
  #  cols = cols[-1: ] + cols[: -1]
  #  data.columns = df.columns
    
      fam = list(x['instrument'])[0]
      print("fam is ",fam)
 
      pairsdict = {
          'instrument1': para1,
      #  'instrument2': ['non_sustained','String','brass','woodwind']}
       # 'instrument2': ['bass_acoustic', 'brass_acoustic', 'flute_acoustic' , 'guitar_acoustic', 'keyboard_acoustic', 'mallet_acoustic', 'organ_acoustic', 'reed_acoustic', 'string_acoustic' ]}
          'instrument2': para2 }
 
    #DataFrame that contains the pairs for which we calculate the Mahalanobis distance
      pairs = pd.DataFrame(pairsdict)
    
        #Add data to the country pairs
      pairs = pairs.merge(data, how='left', left_on=['instrument1'], right_on=['instrument'])
      pairs = pairs.merge(data, how='left', left_on=['instrument2'], right_on=['instrument'])

      c1 = ['zero_crossing_rate_x', 'energy_x',
         'entropy_of_energy_x', 'spectral_centroid_x',
         'spectral_spread_x', 'spectral_entropy_x', 'spectral_flux_x',
         'spectral_rolloff_x', 'MFCC1_x', 'MFCC2_x', 'MFCC3_x',
         'MFCC4_x', 'MFCC5_x', 'MFCC6_x', 'MFCC7_x', 'MFCC8_x',
         'MFCC9_x', 'MFCC10_x', 'MFCC11_x', 'MFCC12_x',
         'MFCC13_x', 'chroma_vector1_x', 'chroma_vector2_x',
         'chroma_vector3_x', 'chroma_vector4_x', 'chroma_vector5_x',
         'chroma_vector6_x', 'chroma_vector7_x', 'chroma_vector8_x',
         'chroma_vector9_x', 'chroma_vector10_x', 'chroma_vector11_x',
         'chroma_vector12_x', 'chroma_deviation_x']
       
      c2 = ['zero_crossing_rate_y', 'energy_y',
         'entropy_of_energy_y', 'spectral_centroid_y',
         'spectral_spread_y', 'spectral_entropy_y', 'spectral_flux_y',
         'spectral_rolloff_y', 'MFCC1_y', 'MFCC2_y', 'MFCC3_y',
         'MFCC4_y', 'MFCC5_y', 'MFCC6_y', 'MFCC7_y', 'MFCC8_y',
         'MFCC9_y', 'MFCC10_y', 'MFCC11_y', 'MFCC12_y',
         'MFCC13_y', 'chroma_vector1_y', 'chroma_vector2_y',
         'chroma_vector3_y', 'chroma_vector4_y', 'chroma_vector5_y',
         'chroma_vector6_y', 'chroma_vector7_y', 'chroma_vector8_y',
         'chroma_vector9_y', 'chroma_vector10_y', 'chroma_vector11_y',
         'chroma_vector12_y', 'chroma_deviation_y']

#Convert data columns to list in a single cell
      pairs['vector1'] = pairs[c1].values.tolist()
      pairs['vector2'] = pairs[c2].values.tolist()
    
      mahala = pairs[['instrument1', 'instrument2', 'vector1', 'vector2']]
      import scipy as sp
      from scipy.spatial.distance import mahalanobis  
        #Calculate covariance matrix
 #   data.astype(float)
      covmx = data.cov()
      print(covmx)
      invcovmx = sp.linalg.inv(covmx)
    
    #Calculate Mahalanobis distance
      mahala['mahala_dist'] = mahala.apply(lambda x: (mahalanobis(x['vector1'], x['vector2'], invcovmx)), axis=1)
 
      mahala = mahala[['instrument1', 'instrument2', 'mahala_dist']]
      maha_distances_per_row.append(mahala)       #append every row mahala distances
    #print(mahala)
      res = mahala.sort_values(['mahala_dist'])       #sort mahala in increasing order per row
    #if res['mahala_dist'].iloc[0] !='nan' :
      if math.isnan(res['mahala_dist'].iloc[0]) :
          print "2"
      else :    
          r = res['instrument2'].iloc[0]                      #min. distance ki family 2 store kar raha
        
          family_result.append(r)                         # family 2 append kar raha row wise
      print(math.isnan(res['mahala_dist'].iloc[0]))

  return family_result  

      