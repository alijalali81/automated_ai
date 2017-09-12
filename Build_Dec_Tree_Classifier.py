def Dec_Tree(inp,w,r,cl_val):	

#Decision Tree Model
	import pandas as pd
	import numpy as np
	from sklearn.metrics import confusion_matrix
	from sklearn.tree import DecisionTreeClassifier
	import warnings
	warnings.filterwarnings("ignore")
	
	print('\nBuilding Decision Tree Model...')
	N=len(inp.columns)-1
	N1=len(inp.index)
	e=len(inp.columns)-2
	v=cl_val.columns
	split=['best','random']
	nfeat=[int(np.floor(np.sqrt(e))+1),int(np.floor(np.sqrt(e))+2)]
	l=len(split)*len(nfeat)*w
	conf_matrix=[1]*l
	config=[1]*l
	model=[1]*l
	result=dict(Accuracy=np.zeros(l,dtype=float))
	
	if(len(v)==2):
		result['Sensitivity']=np.zeros(l,dtype=float)
		result['Specificity']=np.zeros(l,dtype=float)
	else:	
		for g in np.arange(len(v)):
			result['Precision_'+str(g)]=np.zeros(l,dtype=float)
			result['Recall_'+str(g)]=np.zeros(l,dtype=float)
	u=0	
	u1=0	
	for m in range(len(split)):
		for k in range(len(nfeat)):
			result_mean=dict(Accuracy=np.zeros(1,dtype=float))
	
			for j in range(w):	
				print('\nRandomizing Data Set..')
				
				rand=pd.Series(np.arange(N1)).sample(N1)
				inp =inp.iloc[rand,:]
				
				train=pd.DataFrame()
				test=pd.DataFrame()
				print('\nSeparating data set into train and test set...')
				
				for i in range(len(v)):
					inp1=inp[inp.iloc[:,N]==v[i]]
					x=len(inp1.index)
					
					#train and test set
					train=train.append(inp1.iloc[:int(np.floor(x*r)+1),:])
					test=test.append(inp1.iloc[int(np.floor(x*r)+1):,:])
					
				train=train.sample(len(train.index))
				test=test.sample(len(test.index))
				
				i1=np.array(train.iloc[:,1:N])
				o1=train.iloc[:,N]
				
				i2=np.array(test.iloc[:,1:N])
				o2=test.iloc[:,N]
				#Building Decicion Tree Model
				e1=np.shape(i2)[0]
				config[u]=pd.DataFrame(['Decision_Tree',split[m],nfeat[k],'gini'],index=['Classifier','split','max_feat','criterion'],columns=[''])
				#print("\nDecicion Tree Building Configuration:\n%s" % config)
				print('\nBuilding Model...\n')
				class_wt = "balanced"
			
				DT = DecisionTreeClassifier(splitter=split[m],max_features=nfeat[k], 
				class_weight=class_wt)
				
				hist=DT.fit(i1,o1)
				f=DT.predict(i2)
				model[u]=DT
				#Confusion Matrix
				conf_matrix[u]=confusion_matrix(o2, f)
				#print("Confusion matrix:\n%s" % conf_matrix)
				mat=conf_matrix[u]
				result['Accuracy'][u]=(mat.trace()/mat.sum())*100
				if(len(v)==2):
					result['Sensitivity'][u]=(mat[1,1]/sum(mat[1,:]))*100
					result['Specificity'][u]=(mat[0,0]/sum(mat[0,:]))*100
				else:	
					for g in np.arange(len(v)):
						result['Precision_'+str(g)][u]=(mat[g,g]/sum(mat[:,g]))*100
						result['Recall_'+str(g)][u]=(mat[g,g]/sum(mat[g,:]))*100
				u=u+1
			mat=np.sum(conf_matrix[u1:u1+w],axis=0)	
			u1=u1+w
			result_mean['Accuracy']=np.array([(mat.trace()/mat.sum())*100])
			if(len(v)==2):
					result_mean['Sensitivity']=np.array([(mat[1,1]/sum(mat[1,:]))*100])
					result_mean['Specificity']=np.array([(mat[0,0]/sum(mat[0,:]))*100])
			else:	
				for g in np.arange(len(v)):
					result_mean['Precision_'+str(g)]=np.array([(mat[g,g]/sum(mat[:,g]))*100])
					result_mean['Recall_'+str(g)]=np.array([(mat[g,g]/sum(mat[g,:]))*100])	
			res=pd.DataFrame.from_dict(result_mean)
			res.index=['']
			print("\nPerformance Parameters:\n%s" % res)
	result['model']=np.repeat('dt',l)
	result=	pd.DataFrame.from_dict(result)
	
	return result,conf_matrix,config,model		