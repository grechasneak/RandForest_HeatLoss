#!/usr/bimn/env python


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import interpolate
from scipy.interpolate import UnivariateSpline
import glob
from openpyxl import Workbook
from openpyxl import load_workbook
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelBinarizer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import pickle
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# # Star CCM Data Frame
# # df = pd.read_csv('adiabatic_sides_rear_temp.csv', index_col = 'Time')
# # df['Full rad'] = pd.read_csv('rear_temp_full_radiation.csv', index_col = 'Time')
# # df['Adiabatic'] = pd.read_csv('adiabatic_rear_temp.csv', index_col = 'Time')


# Analytical Solution Data Frame
# # df1 = pd.read_csv('analytical_updated.csv', index_col = 'Time')
# # df1['Parker'] = df1['Parker'] 
# # df1['Cowen'] = df1['Cowen'] 

# Post Processing Data Frame
# # df2 = pd.DataFrame()
# # df2['Adiabatic/parker'] = (df['Adiabatic']/df1['Parker'])
# # df2['Adiabatic_sides/cowen'] = (df['Adiabatic sides']/df1['Cowen'])

def listify(gen):
    # Convert a generator to a list
    return [k for k in gen]


def load_excel_data(filename):
	"""
	Passed a full excel file path, load a list of thermophysical properties.
	Functionality depends on proper ordering of values in the correct
	column and row - see example sheet.
	Args:
		-filename: A valid .xlsx or .xls file containing data
	"""
	wb = load_workbook(filename, data_only = True)
	ws = get_ws("Sheet1", wb)
	package = [float(k.value) for k in listify(ws.columns)[3][1:]]
	return package

def get_ws(ws, wb):
	if ws not in wb:
		print("Error: Could not find worksheet %s in workbook. Aborting")
		sys.exit()
	return wb.get_sheet_by_name(ws)
	



	

#half rise time function
def calc_halfrise(dataframe):
	#max temp 
	
	# try:
		# dataframe = dataframe['Cowen']
	# except:
		# print('did not work')
		
	max_temp = dataframe.max()

	# starting temp and time
	start_temp = dataframe.iloc[0]

	#start_temp = np.argmax(dataframe > (initial_temp + 0.0001))
	#start_temp =  [i for i in dataframe if i > (initial_temp + 0.0001)][0]
	
	start_time = dataframe[dataframe == start_temp].index

	
	half_temp = (max_temp - start_temp) / 2 + start_temp

	
	x = dataframe.index
	y = dataframe
	
	#cubic spline fit
	s = UnivariateSpline(x, y, s=0, k=5)
	
	#creating x and ys
	xs = np.arange(0, x[-1] ,.000001)
	ys = s(xs)
	
	#finding half rise time by interpolating on the spline
	half_x = np.interp(half_temp, ys, xs)


	
	root = half_x #- start_time


	return root


def calc_actual_alpha():
	#for squares
	#pathMeta =r'C:\Users\grech\Documents\Python Scripts\senior\Square_meta'
	
	#for cylinder
	pathMeta =r'C:\Users\grech\Documents\Python Scripts\senior\Cylinder_meta'
	

	
	allMeta = glob.glob(pathMeta + "/*.xlsx")
	
	names = []
	alphas = []
	Qs = []
	aspects = []
	temps = []
	
	for file in allMeta:
		data = load_excel_data(file)
		
		#appends file name for square
		data.append(str(file)[-30:-10])
		
		name = data[-1]

		#defines the values in the data
		temp = data[0] #[K]
		rho = data[5] #[kg/m^3]
		cp = data[6] # [j/kg K]
		k = data[4] #[w/m K]
		l = data[2] #[m]
		r = data[3] #[m]
		Q = data[1] # [W/m^2]
		
		aspect = (2 * r) / l
		alpha = k / (rho * cp) #[m^2/s]
		
		names.append(name)
		alphas.append(alpha)
		Qs.append(Q)
		aspects.append(aspect)
		temps.append(temp)
		
	main = pd.DataFrame({'Actual Alpha' : alphas,
						'Name' : names,
						'Heat Deposited W/m^2' : Qs,
						'Aspect Ratio' : aspects,
						'Temeprature' : temps})	
	#for squares
	#path =r'C:\Users\grech\Documents\Python Scripts\senior\Square'
	
	#for analytical squares
	#path = r'C:\Users\grech\Documents\Python Scripts\senior\Analytical_square'
	
	#for cylinders
	path =r'C:\Users\grech\Documents\Python Scripts\senior\Cylinder'
	
	#for analytical cylinders
	#path = r'C:\Users\grech\Documents\Python Scripts\senior\Analytical_cylinder'
	
	allFiles = glob.glob(path + "/*.csv")
	
	_names = []
	_alphas = []
	
	for file in allFiles:
		try:
			df_all = pd.read_csv(file, index_col = 'Physical Time: Physical Time (s)', nrows = 600)
		except:
			df_all = pd.read_csv(file, index_col = 'Time', nrows = 600)
			
		#use this name for square
		name = str(file)[-24:-4]
		
		#call half rise function
		half_time = calc_halfrise(df_all)
		
		l = .001
		
		alpha = (.1388 * l ** 2 )/ half_time
		
		_names.append(name)
		_alphas.append(alpha)
		
	_main = pd.DataFrame({'Calculated Alpha' : _alphas,
						 'Name' : _names})	
						 
	result = _main.merge(main, on = 'Name', how = 'left') 

	result['calculated/actual'] = (result['Calculated Alpha'] / result['Actual Alpha']).astype(float)
	

	
	#result.to_csv('Square_results.csv')
	result.to_csv('Cylinder_results.csv')
	#result.to_csv('Analytical_square.csv')
	#result.to_csv('Analytical_cylinder.csv')
	return result
		

df = calc_actual_alpha()





# df = pd.concat([df1,df2])
# df.to_csv('Analytical_results.csv')



def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
        val_errors.append(mean_squared_error(y_val_predict, y_val))

    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="Training")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="Validating")
    plt.legend(loc="upper right", fontsize=14)   # not shown in the book
    plt.xlabel("Training set size (as a % of total data)", fontsize=14) # not shown
    plt.title('Learning Curve')
    plt.ylabel("RMSE", fontsize=14)              # not shown	
	
	


df = pd.read_csv('star_results.csv')
data_cat = df['shape']
encoder = LabelBinarizer()
data_cat_1hot = encoder.fit_transform(data_cat)
df['shape'] = data_cat_1hot


split = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
for train_index, test_index in split.split(df, df["Aspect Ratio"]):
    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index]

X = np.array(strat_train_set.drop(['calculated/actual'], 1))
#X_train = preprocessing.scale(X)

y_train = strat_train_set['calculated/actual']

X_test = np.array(strat_test_set.drop(['calculated/actual'], 1))
#X_test = preprocessing.scale(X_test)

y_test = strat_test_set['calculated/actual']

clf = RandomForestRegressor()


clf.fit(X, y_train)
#clf.fit(X_t, t_y)

with open('r_forest_regressor.pickle','wb') as f:
    pickle.dump(clf, f)

# pickle_in = open('r_forest_regressor.pickle','rb')
# clf = pickle.load(pickle_in)
	
print(clf.score(X_test, y_test))
	

	
	
	
#This will do the mean squared error from 10 cross validations
# from sklearn.model_selection import cross_val_score
# forest_scores = cross_val_score(clf, X_train, y_train,
                                # scoring="neg_mean_squared_error", cv=10)
# forest_rmse_scores = np.sqrt(-forest_scores)
# print(forest_rmse_scores.mean())


#This will plot the learning curve error
#plot_learning_curves(clf, X_train, y_train)





# This code makes the 3D plots

# df1 = pd.read_csv('Analytical_cylinder.csv')
# df1 = df1[df1.Temeprature != 1573]
# df2 = pd.read_csv('Analytical_square.csv')


# df1 = pd.read_csv('Cylinder_results.csv')
# df1 = df1[df1.Temeprature != 1573]
# df2 = pd.read_csv('Square_results.csv')

# x1 = df1['Aspect Ratio'].values
# y1 = df1['Temeprature'].values
# z1 = df1['calculated/actual'].values

# x2 = df2['Aspect Ratio'].values
# y2 = df2['Temeprature'].values
# z2 = df2['calculated/actual'].values

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.scatter(x1, y1, z1, s=50, c = 'r', alpha = 1.0, marker = 'o')
# ax.scatter(x2, y2, z2, s=50, c = 'b', alpha = 0.5, marker = 's')
# ax.set_xlabel(r'Aspect Ratio$(\frac{diameter}{thckness})$')
# ax.set_ylabel('Temperature [K]')
# ax.set_zlabel(r' $(\frac{Calculated}{Actual}) \alpha$')
# ax.set_title(r'Deviation of $\alpha$ vs. Temeprature & Aspect Ratio')
# ax.tick_params(axis='y', width=10, labelsize=10, pad=0)
# plt.savefig('Analytical.jpg', dpi=1000)	
# plt.show()


	





