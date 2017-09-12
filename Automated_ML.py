#### Automated ML_Platform
def Platform():

	from Data_Preprocessing import Preprocessing
	import Build_Neural_Network_Classifier
	import Build_SVM_Classifier
	import Build_RF_Classifier
	import Build_Dec_Tree_Classifier
	import Build_KNN_Classifier
	import pandas as pd
	import numpy as np
	
	file_name=input("Enter the file name: ")
	print("Reading the file...\n")
	input_data = pd.read_csv(file_name)
	inp,inp_norm,cl_val=Preprocessing(input_data)
	print('.....................................................................')	

	#Asking user for Randomization times
	print('\nRandomization...')
	print('\nWe recommend 3 times Randomization.')
	ur=input("\nDo you want to change it? (y/n): ")
	if ur=='y':
		w = int(input("\nPlease enter the Randomization times: "))
	else:
		w= 3
		
	#Asking user for train test set split proportion
	print('.....................................................................')	 
	print('\nWe recommend .7 and .3 proportion for training.')
	ur=input("\nDo you want to change it? (y/n): ")

	if ur=='y':
		r = float(input("\nPlease enter the training size proportion: "))
	else:
		r= 0.7  
		
	# Building models
	result_nn,conf_matrix_nn,config_nn,model_nn=Build_Neural_Network_Classifier.Neural_Network(inp_norm,w,r,cl_val)
	result_svm,conf_matrix_svm,config_svm,model_svm=Build_SVM_Classifier.SVM(inp_norm,w,r,cl_val)
	result_rf,conf_matrix_rf,config_rf,model_rf=Build_RF_Classifier.RF(inp,w,r,cl_val)
	result_dt,conf_matrix_dt,config_dt,model_dt=Build_Dec_Tree_Classifier.Dec_Tree(inp,w,r,cl_val)
	result_knn,conf_matrix_knn,config_knn,model_knn=Build_KNN_Classifier.KNN(inp_norm,w,r,cl_val)
	
	#Display Top 3 models
	Result=pd.concat([result_nn,result_svm,result_rf,result_dt,result_knn])
	r=pd.Series(Result.index,name='r').reset_index(drop=True)
	Result=Result.reset_index(drop=True)
	Result1=pd.concat([r,Result],axis=1)
	Result1.sort_values(['Accuracy'], axis=0, ascending=False, inplace=True)
	Res=Result1.iloc[0:3,:]
	
	print('.............................................................')	 	
	print('\nTop 3 models are:\n')
	b=['st','nd','rd']
	for i in range(3):
	
		print("\n%s%s Model:\n" %(i+1,b[i]))
		m=eval('config_'+Res.model.iloc[i])[Res.r.iloc[i]]
		print(m)
		conf_mat=pd.DataFrame(eval('conf_matrix_'+Res.model.iloc[i])[Res.r.iloc[i]])
		print("\nConfusion matrix:\n\n%s" % conf_mat)
		m1=eval('result_'+Res.model.iloc[i])
		m1=pd.DataFrame(m1.iloc[Res.r.iloc[i],:])
		m1.columns=['']
		print("\nPerformance Metrices:\n%s" % m1)
		print('.............................................................')	 