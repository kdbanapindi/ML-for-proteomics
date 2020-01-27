import os
import pandas as pd
import numpy as np
import pymzml

def read_mzml(data_dir):

	msrun=pymzml.run.Reader(data_dir, MS_precisions={1:1e-5, 2:1e-4})

	col_names=list(np.round(np.linspace(300,1500,12001),1))
	col_names.append('RT')
	df_all_spectra=pd.DataFrame(columns=col_names)

	spec_ID=[]
	spec_RT=[]
	all_data=[]

	count=0
	for spectrum in msrun: 
 	    
	    count+=1
	    if count%1000==999:
	        #print(count+1)
	    
	    if spectrum.ms_level == 2:
	        
	        #spec_ID.append(int(spectrum.ID))
	        #spec_RT.append(spectrum.scan_time[0])
	        
	        if spectrum.peaks('centroided').any():
	            
	            df_spect = pd.DataFrame(spectrum.peaks('centroided'),columns=['m/z',spectrum.ID])
	            df_spect['m/z'] = round(df_spect['m/z'],0)
	        
	            indexNames = df_spect[(df_spect['m/z']<300) | (df_spect['m/z']>1500) ].index
	            df_spect.drop(indexNames , inplace=True)
	        
	            df_spect=df_spect.groupby('m/z')[spectrum.ID].apply(lambda x: sum(x)).reset_index()
	            df_spect.set_index('m/z', inplace=True)
	            df_spect=df_spect.T
	            df_spect['RT']=spectrum.scan_time[0]
	            df_spect=df_spect.squeeze()
	            
	            all_data.append(df_spect)
	        
	            #df_all_spectra=df_all_spectra.append(df_spect)
	        
	            #print(spectrum.ID)
	               


	all_data=list(filter(lambda x : type(x) == pd.Series, all_data))

	all_data_df=pd.DataFrame(all_data)

	df_all_spectra=df_all_spectra.append(all_data_df)

	return df_all_spectra