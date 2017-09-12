def Preprocessing(input_data):

	import pandas as pd
	import numpy as np
	from sklearn import preprocessing
	from sklearn.preprocessing import Imputer
	from sklearn.ensemble import ExtraTreesClassifier
	from collections import Counter
	
	N=len(input_data.columns)-1
	N1=len(input_data.index)

	#Separate input and output
	input1=input_data.iloc[:,1:N].copy()
	output1=input_data.iloc[:,N].copy()
	s=pd.DataFrame([N1,N+1],index=['Samples','Attributes'],columns=[''])
	print(s)

	print("\nClasses and their counts:\n")
	cl_val=pd.DataFrame(output1.value_counts(normalize=False, sort=True, ascending=False, bins=None)).transpose()
	print(cl_val)

	if((output1.dtype=='O')):
		#print("\nChanging classes values on the basis of counts...")
		for i in range(len(cl_val.columns)):
			output1[output1==cl_val.columns[i]]=i
		cl_nm=cl_val.columns
		cl_val.columns=np.arange(len(cl_val.columns))

	if(0 not in cl_val.columns):
		#print("\nChanging classes values on the basis of counts...")
		output1=output1-min(output1)
		cl_nm=cl_val.columns
		cl_val.columns=cl_val.columns-1
	output1=pd.to_numeric(output1)
	col_name=pd.Series(input1.columns)
	nan_val=['?','Unknown','Invalid','Unknown/Invalid','NaN']
	print("\nChecking for Missing Values...\n")
		
	#Find the type of the attributes (Categorical attributes)
	att_type= col_name.apply(lambda x: input1.loc[:,x].dtype=='O')
	attribute=col_name[att_type]
		
	#Assignment
	for j in attribute:
		#unique values of an attribute
		att_val=np.unique(input1.loc[:,j])
			
		#Discard Nan values and do assignment only for the 
		#valid attribute values
		att_val=att_val[pd.Series(att_val).apply(lambda x: x not in nan_val)]
			
		for i in range(len(att_val)):
			input1.loc[input1.loc[:,j]==att_val[i],j]=i
				
		val=input1.loc[:,j].apply(lambda x: x in nan_val)
		input1.loc[val,j]=np.nan
			
	miss= col_name.apply(lambda x: sum(pd.isnull(input1.loc[:,x])))
	miss_val=pd.concat([col_name,miss],axis=1,ignore_index=True)
	miss_val.columns=['Att_name','Miss_Val_Count']
		
	print(miss_val)
		
	#impute missing values
	print("\nImputation for missing values...\n")
	imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
	input1=imp.fit_transform(input1)		
		
	print("Checking for Categorical Attributes...\n")
	s=pd.DataFrame([len(attribute)],index=['Categorical Attributes:'],columns=[''])
	print(s)
	print('.....................................................................')	
		
	print("\nAssigning Numerical Values for Categorical Attributes...\n")
	input2=pd.DataFrame(input1,columns=col_name)
	print(input2.iloc[0:2,0:5])
	print('.....................................................................')		

	#Normalization
	print("\nFeature Scaling...\n")
	min_max_scaler = preprocessing.MinMaxScaler()
	input1_norm = min_max_scaler.fit_transform(input1)
	input2_norm=pd.DataFrame(input1_norm,columns=col_name)
	print(input2_norm.iloc[0:2,0:5])

	print('.....................................................................')	
	#Calculate Variable Importance
	print("\nComputing Variable Importance...\n")
	model = ExtraTreesClassifier()
	model.fit(input1,output1)
	vi=model.feature_importances_
		
	#Scaling of the Variable Importance
	vi_minmax = min_max_scaler.fit_transform(vi.reshape(-1,1)).ravel()*100
	vi_idx=np.argsort(vi_minmax)[::-1]
	var_imp=pd.concat([col_name[vi_idx].reset_index(drop=True),pd.Series(vi_minmax[vi_idx])],axis=1,ignore_index=True)
	var_imp.columns=['Att_name','Var_Imp']
		
	print(var_imp)
	print('.....................................................................')		

	#Feature Selection
	print("\nFeature Selection...\n")
	ind_sel=vi_minmax>=5
	input_sel=input1[:,ind_sel]
	input_sel_norm=input1_norm[:,ind_sel]
	input2_norm_sel=pd.DataFrame(input_sel_norm,columns=col_name[ind_sel])
	print("\nSelected Features ....\n")
	print(col_name[ind_sel].reset_index(drop=True))
	print(input2_norm_sel.iloc[0:2,0:4])
		
	inp=pd.concat([input_data.iloc[:,0],pd.DataFrame(input_sel,columns=col_name[ind_sel]),output1],axis=1)	
	inp_norm=pd.concat([input_data.iloc[:,0],input2_norm_sel,output1],axis=1)
	return inp,inp_norm,cl_val
	
	