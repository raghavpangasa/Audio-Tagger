## Somya Goel and Raghav Pangasa

from flask import Flask, render_template, redirect, url_for
from flask import request
from flask import json
from werkzeug import secure_filename
import os
import audioAnalysis
from pandas import read_csv
import csv
import pickle
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
import re
import find_family_st as ff
import find_elec_family_mt as fe
import find_syn_family_mt as fs
import find_instrument as ac_ff
from collections import Counter
from sklearn.externals import joblib
import math
from pydub import AudioSegment


app = Flask(__name__)


@app.route('/')

def upload_file():
   return render_template('index.html')


pitch = []
velocity = []
@app.route('/uploader', methods = ['GET', 'POST'])


def upload_files():
   if request.method == 'POST':
      f = request.files["file"]
      f.save(os.path.join('/Users/somyagoel/Flask/basic_app/music',secure_filename(f.filename)))
      print("......................starting freshhhhhhhhhhhh.....................")
      print("filenameeeeeeee")
      print(f.filename)
      t = re.sub(r'\[', '', f.filename)
      f.filename = re.sub(r'\]', '', t)
      print(f.filename)
      f_replace = f.filename.replace(" ", "_")
      print(f_replace)
      print(f.filename.replace(" ", "_"))
      path_to_music = '/Users/somyagoel/Flask/basic_app/music/'

      file_pass1 = f_replace
      print(file_pass1)
      var_x = file_pass1[-3:]
#      print("var_x")
#      print(var_x)
      var_y = file_pass1[:-3]
#      print("print var y")
#      print(var_y)
      if(var_x == "mp3"):	  	
      	sound = AudioSegment.from_mp3(path_to_music + file_pass1)
      	var_y = var_y + "wav"
      	sound.export(path_to_music + var_y, format="wav")
      command = "python /Users/somyagoel/Flask/basic_app/audioAnalysis.py featureExtractionFile -i /Users/somyagoel/Flask/basic_app/music/"
      command = command + file_pass1 +" -mw 2.0 -ms 2.0 -sw 0.50 -ss 0.50 -o /Users/somyagoel/Flask/basic_app/music/" + file_pass1
      print(command)
 #     print(command)
      os.system(command)
 #     os.system('python /Users/somyagoel/Flask/basic_app/audioAnalysis.py featureExtractionDir -i /Users/somyagoel/Flask/basic_app/music -mw 2.0 -ms 2.0 -sw 0.50 -ss 0.50 ')
      rf = '/Users/somyagoel/Flask/basic_app/music/st/'+file_pass1+'_st.csv'
      mt = '/Users/somyagoel/Flask/basic_app/music/mt/'+file_pass1+'.csv'

      with open(rf) as fi:
      		r = csv.reader(fi)
      		data = [line for line in r]

      with open(rf,'w') as fi:
        	w = csv.writer(fi)
        	w.writerow(['zero_crossing_rate','energy','entropy_of_energy','spectral_centroid',
            	'spectral_spread','spectral_entropy','spectral_flux','spectral_rolloff',
            	'MFCC1','MFCC2','MFCC3','MFCC4','MFCC5','MFCC6','MFCC7', 
            	'MFCC8','MFCC9','MFCC10','MFCC11','MFCC12','MFCC13',
            	'chroma_vector1','chroma_vector2','chroma_vector3','chroma_vector4',
            	'chroma_vector5','chroma_vector6','chroma_vector7','chroma_vector8',
            	'chroma_vector9','chroma_vector10','chroma_vector11','chroma_vector12',
            	'chroma_deviation'])
        	w.writerows(data)


      with open(mt) as fi:
          r = csv.reader(fi)
          data1 = [line for line in r]

      with open(mt,'w') as fi:
          w = csv.writer(fi)
          w.writerow(['zero_crossing_rate_mean','energy_mean','entropy_of_energy_mean','spectral_centroid_mean',
              'spectral_spread_mean','spectral_entropy_mean','spectral_flux_mean','spectral_rolloff_mean',
              'MFCC1_mean','MFCC2_mean','MFCC3_mean','MFCC4_mean','MFCC5_mean','MFCC6_mean','MFCC7_mean', 
              'MFCC8_mean','MFCC9_mean','MFCC10_mean','MFCC11_mean','MFCC12_mean','MFCC13_mean',
              'chroma_vector1_mean','chroma_vector2_mean','chroma_vector3_mean','chroma_vector4_mean',
              'chroma_vector5_mean','chroma_vector6_mean','chroma_vector7_mean','chroma_vector8_mean',
              'chroma_vector9_mean','chroma_vector10_mean','chroma_vector11_mean','chroma_vector12_mean',
              'chroma_deviation_mean','zero_crossing_rate_sd','energy_sd','entropy_of_energy_sd','spectral_centroid_sd',
              'spectral_spread_sd','spectral_entropy_sd','spectral_flux_sd','spectral_rolloff_sd',
              'MFCC1_sd','MFCC2_sd','MFCC3_sd','MFCC4_sd','MFCC5_sd','MFCC6_sd','MFCC7_sd', 
              'MFCC8_sd','MFCC9_sd','MFCC10_sd','MFCC11_sd','MFCC12_sd','MFCC13_sd',
              'chroma_vector1_sd','chroma_vector2_sd','chroma_vector3_sd','chroma_vector4_sd',
              'chroma_vector5_sd','chroma_vector6_sd','chroma_vector7_sd','chroma_vector8_sd',
              'chroma_vector9_sd','chroma_vector10_sd','chroma_vector11_sd','chroma_vector12_sd',
              'chroma_deviation_sd'])
          w.writerows(data1)

      X_test = pd.read_csv(rf,low_memory=False)				# st wali file uploaded audio ki
      X_test = X_test.iloc[:, 0:34]

      X_test_mt = pd.read_csv(mt,low_memory=False)			# mt wali file uploaded audio ki
      X_test_mt_68 = X_test_mt.iloc[:, 0:68]
      X_test_mt = X_test_mt.iloc[:, 0:34]
   #   print(X_test_mt_68.head())

      typef = []
      family = []


      aes_model = "/Users/somyagoel/Flask/basic_app/bin/rf_aes2.pkl"			# RF model instrument type ke liye (3 classes)
      aes_model = joblib.load(aes_model)
      y_pred = aes_model.predict(X_test_mt)										# testing on mt wali file
 #     print("y_pred = ",y_pred)
      type_res = Counter(y_pred).most_common(1)[0][0]
      print(type_res)															# RF result stored in type_res (ac, el, syn)

      if(type_res == 'acoustic'):												# AGAR ACOUSTIC HAI 
	      fam_res = []
	      fam_res =  ff.find_family_type(rf)
	      fam_res1 = Counter(fam_res).most_common(1)[0][0]
	      print("fam ressss isssss")
	      print(fam_res1)         			 # yahi final result hai jo print karna hai screen par ...its not a list ...ek string hai bas 
	    #  fam_res2 = Counter(fam_res).most_common(2)[1][0]
	      #    print(fam_res)
	  #    print(Counter(fam_res))
	      if(fam_res1 == 'bass_guitar_mallet'):
	        para_1_count = 3
	        para_2 = ["bass_acoustic","guitar_acoustic","mallet_acoustic"]
	        para_3 = '/Users/somyagoel/Flask/basic_app/centroids/acoustic_cluster_2.csv'
	      else :
	        if(fam_res1 == 'brass_string'):
	          para_1_count = 2
	          para_2 = ["brass_acoustic","string_acoustic"]
	          para_3 = '/Users/somyagoel/Flask/basic_app/centroids/acoustic_cluster_3.csv'
	        else :
	          if(fam_res1 == 'flute_organ_reed'):
	            para_1_count = 3
	            para_2 = ["flute_acoustic","organ_acoustic","reed_acoustic"]
	            para_3 = '/Users/somyagoel/Flask/basic_app/centroids/acoustic_cluster_4.csv'
	          else :
	           	family_result = 'key'
	      if(fam_res1 != 'key') : 
		      final_ins = []
		      dataset = pd.DataFrame(pd.read_csv(rf,low_memory=False))
		      dataset = dataset.loc[:, ~dataset.columns.str.contains('^Unnamed')]
		      dataset['instrument'] = 'Inst'
		      X = dataset	
		      y = pd.DataFrame(X['instrument'])
		      X.drop(['instrument'], axis = 1, inplace = True)
		      w = 'ins'
		      z = []
		      for i in range(len(y)):
		      	z.append(w + str(i+1))    
		      df = pd.concat([y,X], axis = 1)
		 #     print(df)
		      df['instrument'] = z
		  #    print(df)
		      centroid = pd.DataFrame(pd.read_csv(para_3,low_memory=False))
		      centroid = centroid.loc[:, ~centroid.columns.str.contains('^Unnamed')]
		      centroid = centroid.iloc[:, 0:35]
		  #    print(centroid)
		      centroid.columns = df.columns
		      family_result = []
		      maha_distances_per_row =[]
		      for row in df.iterrows():
		  #    	print(row)
		      	x = pd.DataFrame(row[1]).transpose()
		        data = pd.concat([x,centroid],ignore_index=True)  #fam wali row ko baki 3 type ki row ke saath concat
		      #  data.columns = df.columns    
		        #data.reset_index(drop=True)
		        data=pd.DataFrame(data)
		        f = data['instrument']
		        data.drop(['instrument'],axis = 1, inplace = True)
		        data[:] = data[ : ].astype('float64')
		        data['instrument'] = f
		        
		        fam = list(x['instrument'])[0]
		        if(para_1_count == 3):
		        	para_1 = [fam, fam, fam]
		        else:
		        	if(para_1_count == 2):
		        		para_1 = [fam, fam]	
		        pairsdict = {
		            'instrument1': para_1,
		           # 'instrument2': ['bass_acoustic', 'brass_acoustic', 'flute_acoustic' , 'guitar_acoustic', 'keyboard_acoustic', 'mallet_acoustic', 'organ_acoustic', 'reed_acoustic', 'string_acoustic' ]}
		           # 'instrument2': ['key','bass_guitar_mallet','brass_string','flute_organ_reed']}
		            'instrument2': para_2}
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
		     
		        covmx = data.cov()
		        invcovmx = sp.linalg.inv(covmx)
	#	        print(invcovmx)		        
		        #Calculate Mahalanobis distance
		        mahala['mahala_dist'] = mahala.apply(lambda x: (mahalanobis(x['vector1'], x['vector2'], invcovmx)), axis=1)
		        mahala = mahala[['instrument1', 'instrument2', 'mahala_dist']]
	#	        print(mahala)
		        maha_distances_per_row.append(mahala)       #append every row mahala distances
		        #print(mahala)
		        res = mahala.sort_values(['mahala_dist'])       #sort mahala in increasing order per row
		        if math.isnan(res['mahala_dist'].iloc[0]) :
		            print "2"
		        else :    
		            r = res['instrument2'].iloc[0]                      #min. distance ki family 2 store kar raha	  
		            family_result.append(r)   
		     #       print(family_result)	



      if(type_res == 'electronic'):												# AGAR ELECTRNOIC HAI 
	      fam_res_ELEC = []
	      fam_res_ELEC =  fe.find_family_type_elec(rf)
	      fam_res_ELEC = Counter(fam_res_ELEC).most_common(1)[0][0]
	      print(fam_res_ELEC)         		 # yahi final result hai jo print karna hai screen par ...its not a list ...ek string hai bas 
	      if(fam_res_ELEC == 'key_guitar_mallet'):
	        para_1_count = 3
	        para_2 = ["keyboard_electronic","guitar_electronic","mallet_electronic"]
	        para_3 = '/Users/somyagoel/Flask/basic_app/centroids/electronic_cluster_1.csv'
	      else :
	        if(fam_res_ELEC == 'brass_reed_string'):
	          para_1_count = 3
	          para_2 = ["brass_electronic","reed_electronic","string_electronic"]
	          para_3 = '/Users/somyagoel/Flask/basic_app/centroids/electronic_cluster_3.csv'
	        else :
	          if(fam_res_ELEC == 'flute_organ'):
	            para_1_count = 2
	            para_2 = ["flute_electronic","organ_electronic"]
	            para_3 = '/Users/somyagoel/Flask/basic_app/centroids/electronic_cluster_4.csv'
	          else :
	          	family_result = 'bass_electronic'
	      if(fam_res_ELEC != 'bass') : 
		      final_ins = []
		      dataset = pd.DataFrame(pd.read_csv(rf,low_memory=False))
		      dataset = dataset.loc[:, ~dataset.columns.str.contains('^Unnamed')]
		      dataset['instrument'] = 'Inst'
		      X = dataset	
		      y = pd.DataFrame(X['instrument'])
		      X.drop(['instrument'], axis = 1, inplace = True)
		      w = 'ins'
		      z = []
		      for i in range(len(y)):
		      	z.append(w + str(i+1))    
		      df = pd.concat([y,X], axis = 1)
		 #     print(df)
		      df['instrument'] = z
		  #    print(df)
		      centroid = pd.DataFrame(pd.read_csv(para_3,low_memory=False))
		      centroid = centroid.loc[:, ~centroid.columns.str.contains('^Unnamed')]
		      centroid = centroid.iloc[:, 0:35]
		  #    print(centroid)
		      centroid.columns = df.columns
		      family_result = []
		      maha_distances_per_row =[]
		      for row in df.iterrows():
		  #    	print(row)
		      	x = pd.DataFrame(row[1]).transpose()
		        data = pd.concat([x,centroid],ignore_index=True)  #fam wali row ko baki 3 type ki row ke saath concat
		      #  data.columns = df.columns    
		        #data.reset_index(drop=True)
		        data=pd.DataFrame(data)
		        f = data['instrument']
		        data.drop(['instrument'],axis = 1, inplace = True)
		        data[:] = data[ : ].astype('float64')
		        data['instrument'] = f
		        
		        fam = list(x['instrument'])[0]
		        if(para_1_count == 3):
		        	para_1 = [fam, fam, fam]
		        else:
		        	if(para_1_count == 2):
		        		para_1 = [fam, fam]	
		        pairsdict = {
		            'instrument1': para_1,
		           # 'instrument2': ['bass_acoustic', 'brass_acoustic', 'flute_acoustic' , 'guitar_acoustic', 'keyboard_acoustic', 'mallet_acoustic', 'organ_acoustic', 'reed_acoustic', 'string_acoustic' ]}
		           # 'instrument2': ['key','bass_guitar_mallet','brass_string','flute_organ_reed']}
		            'instrument2': para_2}
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
		     
		        covmx = data.cov()
		        invcovmx = sp.linalg.inv(covmx)
		        
		        #Calculate Mahalanobis distance
		        mahala['mahala_dist'] = mahala.apply(lambda x: (mahalanobis(x['vector1'], x['vector2'], invcovmx)), axis=1)
		     
		        mahala = mahala[['instrument1', 'instrument2', 'mahala_dist']]
		        maha_distances_per_row.append(mahala)       #append every row mahala distances
		        #print(mahala)
		        res = mahala.sort_values(['mahala_dist'])       #sort mahala in increasing order per row
		        if math.isnan(res['mahala_dist'].iloc[0]) :
		            print "2"
		        else :    
		            r = res['instrument2'].iloc[0]                      #min. distance ki family 2 store kar raha
		            
		            family_result.append(r)   
		       #     print(family_result)
	        #    print(fam_res_ELEC)
	  #    print(Counter(fam_res_ELEC))

      if(type_res == 'synthetic'):												# AGAR SYNTHETIC HAI 
	      fam_res_SYN = []
	      fam_res_SYN =  fs.find_family_type_syn(rf)
	      fam_res_SYN = Counter(fam_res_SYN).most_common(1)[0][0]
	      print(fam_res_SYN)          # yahi final result hai jo print karna hai screen par ...its not a list ...ek string hai bas 
#	      family_result = fam_res_SYN
#	      print(family_result)
	      if(fam_res_SYN == 'bass_keyboard'):
	        para_1_count = 2
	        para_2 = ["bass_synthetic","keyboard_synthetic"]
	        para_3 = '/Users/somyagoel/Flask/basic_app/centroids/synthetic_cluster_2.csv'
	      else :
	        if(fam_res_SYN == 'reed'):
	        	family_result = 'reed_synthetic'
	        else :
	          if(fam_res_SYN == 'guitar'):
	          	family_result = 'guitar_synthetic'
	          else :
	           	family_result = 'mallet_synthetic'
	      if(fam_res_SYN =='bass_keyboard') : 
		      final_ins = []
		      dataset = pd.DataFrame(pd.read_csv(rf,low_memory=False))
		      dataset = dataset.loc[:, ~dataset.columns.str.contains('^Unnamed')]
		      dataset['instrument'] = 'Inst'
		      X = dataset	
		      y = pd.DataFrame(X['instrument'])
		      X.drop(['instrument'], axis = 1, inplace = True)
		      w = 'ins'
		      z = []
		      for i in range(len(y)):
		      	z.append(w + str(i+1))    
		      df = pd.concat([y,X], axis = 1)
		 #     print(df)
		      df['instrument'] = z
		  #    print(df)
		      centroid = pd.DataFrame(pd.read_csv(para_3,low_memory=False))
		      centroid = centroid.loc[:, ~centroid.columns.str.contains('^Unnamed')]
		      centroid = centroid.iloc[:, 0:35]
		  #    print(centroid)
		      centroid.columns = df.columns
		      family_result = []
		      maha_distances_per_row =[]
		      for row in df.iterrows():
		  #    	print(row)
		      	x = pd.DataFrame(row[1]).transpose()
		        data = pd.concat([x,centroid],ignore_index=True)  #fam wali row ko baki 3 type ki row ke saath concat
		      #  data.columns = df.columns    
		        #data.reset_index(drop=True)
		        data=pd.DataFrame(data)
		        f = data['instrument']
		        data.drop(['instrument'],axis = 1, inplace = True)
		        data[:] = data[ : ].astype('float64')
		        data['instrument'] = f
		        
		        fam = list(x['instrument'])[0]
		        if(para_1_count == 3):
		        	para_1 = [fam, fam, fam]
		        else:
		        	if(para_1_count == 2):
		        		para_1 = [fam, fam]	
		        pairsdict = {
		            'instrument1': para_1,
		           # 'instrument2': ['bass_acoustic', 'brass_acoustic', 'flute_acoustic' , 'guitar_acoustic', 'keyboard_acoustic', 'mallet_acoustic', 'organ_acoustic', 'reed_acoustic', 'string_acoustic' ]}
		           # 'instrument2': ['key','bass_guitar_mallet','brass_string','flute_organ_reed']}
		            'instrument2': para_2}
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
		     
		        covmx = data.cov()
		        invcovmx = sp.linalg.inv(covmx)
	#	        print(invcovmx)		        
		        #Calculate Mahalanobis distance
		        mahala['mahala_dist'] = mahala.apply(lambda x: (mahalanobis(x['vector1'], x['vector2'], invcovmx)), axis=1)
		        mahala = mahala[['instrument1', 'instrument2', 'mahala_dist']]
	#	        print(mahala)
		        maha_distances_per_row.append(mahala)       #append every row mahala distances
		        #print(mahala)
		        res = mahala.sort_values(['mahala_dist'])       #sort mahala in increasing order per row
		        if math.isnan(res['mahala_dist'].iloc[0]) :
		            print "2"
		        else :    
		            r = res['instrument2'].iloc[0]                      #min. distance ki family 2 store kar raha	  
		            family_result.append(r)   
		            print(family_result)	
	        #    print(fam_res_SYN)
	  #    print(Counter(fam_res_SYN))
      if(type_res == 'acoustic'):
      	if(family_result == 'key'):												 
      		s1 = 'keyboard'
      		family_result =  s1+"_acoustic"
      	else :
      		if(family_result != 'key') :
      			family_result = Counter(family_result).most_common(1)[0][0]
      	#print(family_result) 

      if(type_res == 'electronic'):
      	if (fam_res_ELEC == 'bass'):
      		family_result = 'bass_electronic'
      	else :
      		family_result = Counter(family_result).most_common(1)[0][0]

      if(type_res == 'synthetic'):
    	print("family result in if synthetic")
    	print(family_result)
    	print(len(family_result))
    	if(family_result!='reed_synthetic' and family_result!='guitar_synthetic' and family_result!='mallet_synthetic'):
    		family_result = Counter(family_result).most_common(1)[0][0]
    		print(family_result)
    	else:
    		print(family_result)

      
# GENRE WALA PART 
      genre_result = []

      inst_result = ["ins1", "instr2", "instrum3", "instrument4", "ins5"]

      filename_sav3 = '/Users/somyagoel/Flask/basic_app/bin/irmas__mt_genre_remix.sav'      
      loaded_model3 = pickle.load(open(filename_sav3, 'rb'))
      y_pred3 = loaded_model3.predict(X_test)
 #     print("y_pred = ",y_pred3)
      inst3 = set(y_pred3)
      genre_list_irmas_remix = list(inst3)
      print("genres present are: ", genre_list_irmas_remix)
      gen_name_list1 = ['classical','country folk','jazz','pop']              #irmas ki list

      instr = family_result
      pitch_file  = "/Users/somyagoel/Flask/basic_app/bin/pitch/" + instr + "_mt.csv.pkl"
      velocity_file = "/Users/somyagoel/Flask/basic_app/bin/velocity/" + instr + "_mt.csv.pkl"
      pitch_model = joblib.load(pitch_file)
      velocity_model = joblib.load(velocity_file)
      p = pitch_model.predict(X_test_mt_68)
      v = velocity_model.predict(X_test_mt_68)
      print(p)
      pitch.append(p)
      velocity.append(v)
      print(v)
      print(pitch)
      print(velocity)

      genre_name_result=[]
      for a in genre_list_irmas_remix :
          genre_name_result.append(gen_name_list1[a])
  #         print(a)
  #        print(gen_name_list1[a])

      return render_template('results.html',result1=inst_result,result2=genre_name_result[0],filen = file_pass1, family_result = family_result, type_result = type_res)


@app.route('/details')
def details():
	return render_template('details.html',pitch_results=pitch,velocity_results=velocity)


@app.route('/home')
def home():
    return redirect(url_for('upload_file'))


if __name__ == "__main__":
	app.run()
