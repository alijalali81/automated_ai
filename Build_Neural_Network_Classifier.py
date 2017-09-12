def Neural_Network(inp,w,r,cl_val):

	import pandas as pd
	import numpy as np
	import keras
	from keras import optimizers,callbacks 
	from keras.utils import to_categorical
	from keras.regularizers import l2
	from keras.layers import Activation, Dense, Input
	from keras.models import Sequential, Model, load_model
	from keras import backend as K
	from sklearn.metrics import confusion_matrix
	import warnings
	warnings.filterwarnings("ignore")
	
	print('\nBuilding Neural Network Model...')
	N=len(inp.columns)-1
	N1=len(inp.index)
	e=len(inp.columns)-2
	v=cl_val.columns
	Act=["tanh","relu"]
	Unit=[e+10,e+5]
	l=len(Act)*len(Unit)*w
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
	for m in range(len(Act)):
		for k in range(len(Unit)):
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
					
					#Normalized train and test set
					train=train.append(inp1.iloc[:int(np.floor(x*r)+1),:])
					test=test.append(inp1.iloc[int(np.floor(x*r)+1):,:])
					
				train=train.sample(len(train.index))
				test=test.sample(len(test.index))
				
				i1=np.array(train.iloc[:,1:N])
				o1=train.iloc[:,N]
				
				i2=np.array(test.iloc[:,1:N])
				o2=test.iloc[:,N]
				#Building Neural Network Model
				
				oc1=to_categorical(o1, num_classes=len(v))
				e1=np.shape(i2)[0]
				config[u]=pd.DataFrame(['Neural_Network',Unit[k],Act[m],Unit[k]-10,Act[m],200],index=['Classifier','first_layer_units','first_layer_activation','second_layer_units','second_layer_activation','batch_size'],columns=[''])
				#print("\nNeural Network Model Building Configuration:\n%s" % config)
				
				print('\nBuilding Model...\n')
				NN = Sequential()
				NN.add(Dense(units=Unit[k], input_dim=e,activation=Act[m],use_bias=True,
				kernel_initializer=keras.initializers.RandomUniform(minval=-0.4, maxval=0.4, seed=None),
				bias_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=1, seed=None),
				activity_regularizer=l2(1e-6)))
				
				NN.add(Dense(units=Unit[k]-10, activation=Act[m],use_bias=True,
				kernel_initializer=keras.initializers.RandomUniform(minval=-0.4, maxval=0.4, seed=None),
				bias_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=1, seed=None),
				activity_regularizer=l2(1e-6)))
				
				NN.add(Dense(units=len(v),activation='softmax',use_bias=True,
				kernel_initializer=keras.initializers.RandomUniform(minval=-0.4, maxval=0.4, seed=None),
				bias_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=1, seed=None),
				activity_regularizer=l2(1e-6)))
				#Optimization
				Ada=optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-10, decay=0.0)
				NN.compile(loss='categorical_crossentropy', optimizer=Ada)
				cb=callbacks.EarlyStopping(monitor='loss', min_delta=0.0001, patience=10, verbose=0, mode='auto')
				class_wt = cl_val.apply(lambda x: cl_val[0]/x)
				class_wt=dict(zip(class_wt.columns,class_wt.iloc[0,:]))
				#Prediction
				history=NN.fit(i1,oc1, batch_size=200,epochs=500,callbacks=[cb],shuffle=True,class_weight=class_wt,verbose=0)
				f=NN.predict_classes(i2, batch_size=e1,verbose=0)
				model[u]=NN
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
	result['model']=np.repeat('nn',l)
	result=	pd.DataFrame.from_dict(result)
	
	return result,conf_matrix,config,model		