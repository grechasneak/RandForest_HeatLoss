from __future__ import division

import numpy as np
np.set_printoptions(precision=8)
import matplotlib.pyplot as plt
#from preferences import *
# Bessel functions for solving Cape-Lehman equations
from scipy.special import jv
from scipy.special import jn_zeros

#root finding
import scipy.optimize

from sklearn import preprocessing

#Reading in excel data
READ_FILE = True
try:
	from openpyxl import Workbook
	from openpyxl import load_workbook
except:
	READ_FILE = False
#Outputting dataframes
import pandas


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Input 'decks' with material properties
FILENAME = "material_properties_sheet.xlsx"
FILENAME_RECTANGLE = "pavel_rect_deck.xlsx"

# STAR-CCM results in csv format
DATAFILE = "sq_030_rall_u1473_t4_x4.csv"

def main():

	# full path to the file containing the excel thermophysical properties
	# Currently has my path in it....

	# This is where you would use the time values from the simulations:
	#time = df["time"]
	#time = np.linspace(0,2,1000)

	# Loading the file contents, creating temperature arrays
	global READ_FILE, FILENAME
	if READ_FILE:
		# "Reading cylinder material data from %s" % FILENAME
		params_package = load_excel_data(FILENAME)
		# "Reading rectangle material data from %s" % FILENAME_RECTANGLE
		params_package_rectangle = load_excel_data(FILENAME_RECTANGLE)
	else:
		# If you don't have the excel module, just hard code values.
		params_package = generate_package()

	dump_params_package(params_package)
	dump_params_package(params_package_rectangle, rect=True)

	# Read in the simulated data
	global DATAFILE
	sim_u = pandas.read_csv(DATAFILE,  index_col = 'Physical Time: Physical Time (s)')
	time = sim_u.index.values
	
	
	# Analytical solutions
	N_terms = 20  # number of terms in summation (outer)
	M = 10 # number of terms in inner summation(s)

	Parker_T = Parker_u(time, N_terms, params_package)
	Cowan_T = Cowan_u(time, N_terms, params_package)
	Rectangle_T = Rectangle_u(time, M, M, N_terms, params_package_rectangle)

	# Stitching

	sim_u["cowan"] = Cowan_u(time, N_terms, params_package)
	sim_u["parker"] = Parker_u(time, N_terms, params_package)
	sim_u["rectangle"] = Rectangle_u(time, M, M, N_terms, params_package_rectangle)
	
	#sim_u["rectangle"] = preprocessing.scale(sim_u["rectangle"])
	#sim_u['star'] = preprocessing.scale(sim_u['star'])
	sim_u['star/ana']= sim_u['star'] / sim_u["rectangle"]
	print(sim_u['star/ana'].mean())
	# Here you would do something like df["analytical"] = Parker_T etc.
	sim_u['star/ana'].plot()
	plt.show()
	# Plotting
#	overlay_plots(sim_u, ["simulated", "cowan"])
	return sim_u

	

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# generating dummy data

def generate_package(rect=False):

	# dummy list of paramaters in case excel is down
	u0 = 1000  # Initial sample temperature, K
	Q =	 400 # pulse energy per area [J/m^2]
	L = .001  # sample thickness [m]
	k = 1  # conductivity [W/(m*K)]
	rho = 8000	# sample density [kg/m^3]
	c_p = .4  # sample specific heat [J/(m*K)]

	# CYLINDRICAL
	R = .02	 # sample radius [m]
	eps_z1 = 1	# emissivity, axial
	eps_z2 = 1
	eps_r = .8	# emissivity, radial

	# RECTANGULAR
	# X-, Y- dimensions are one half total lengths...
	a = .01	 # sample x-dimension [m]
	b = .01	 # sample y-dimension [m]
	eps_x = 1  # emissivity, x-direction
	eps_y = 1 # emissivity y-direction

	if rect:
		params_package = [u0, Q, L, a, b, k, rho, c_p, eps_z1, eps_z2, eps_x, eps_y]
	else:
		params_package = [u0, Q, L, R, k, rho, c_p, eps_z1, eps_z2, eps_r]

	return params_package

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Plotting routines
def normalize_col(arr, baseval=0):
	return (arr-baseval)/max( arr-baseval)

def overlay_plots(df, colnames):
	# For the given dataframe, overlay the columns in the list [colname0, colname1]

	fig = plt.figure(facecolor="white")
	ax = fig.add_subplot(111)
	ax.set_title('Comparison of %s vs %s' % (colnames[0], colnames[1]) )

	x = df.index.values

	y1 =  df[colnames[0]]
	y2 = df[colnames[1]]
	ax.plot(df.index, y1, color="red")
	ax.plot(df.index, y2, color="blue")

	ymax = max(max(y1), max(y2))*(1.1)
	ax.set_ylim([0, ymax])
	ax.set_xlim([0,8])


	plt.xlabel('Time [s]')
	plt.ylabel('Normalized Temperature Change')
	plt.legend(colnames)
	plt.show()



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

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
	wb = open_xlsx(filename)
	ws = get_ws("Sheet1", wb)
	package = [float(k.value) for k in listify(ws.columns)[3][1:]]
	return package

#Try to open a worksheet handle of ws from wb handle
def get_ws(ws, wb):
	if ws not in wb:
		# "Error: Could not find worksheet %s in workbook. Aborting"
		sys.exit()
	return wb.get_sheet_by_name(ws)

#Open an excel file, returning a handle to the workbook
def open_xlsx(fname):
	return load_workbook(fname, data_only = True)

def params_dct(package):
	# Hard-coded dictionary of params package properties/values
	u0, Q, L, R, k, rho, c_p, eps_z1, eps_z2, eps_r = package
	dct = {
			"u0": u0,
			"Q": Q,
			"L": L,
			"R": R,
			"k": k,
			"rho": rho,
			"c_p": c_p,
			"eps_z1": eps_z1,
			"eps_z2": eps_z2,
			"eps_r": eps_r
	}
	return dct

def params_dct_rect(package):
	u0, Q, L, a, b, k, rho, c_p, eps_z1, eps_z2, eps_x, eps_y = package
	dct = {
			"u0": u0,
			"Q": Q,
			"L": L,
			"a": a,
			"b": b,
			"k": k,
			"rho": rho,
			"c_p": c_p,
			"eps_z1": eps_z1,
			"eps_z2": eps_z2,
			"eps_x": eps_x,
			"eps_y": eps_y
	}
	return dct

def dump_params_package(package, rect=False):
	# # the thermophysical properties to the conole
	# "Thermophysical data:"
	if rect:
		vals = params_dct_rect(package)
	else:
		vals = params_dct(package)
	#for k, v in vals.iteritems():
		# "\t%s: %6.3f" % (k, v)
	# Diffusivity
	alpha = vals["k"]/(vals["rho"]*vals["c_p"])
	# "\talpha: %6.8f" % alpha

	# Final temp
	T_inf = vals["Q"]/(vals["rho"]*vals["c_p"]*vals["L"])
	# "\tT_inf: %6.8f" % T_inf


	# characteristic time
	t_c = (vals["L"]/np.pi)**2 / alpha
	# "\tt_c: %6.8f" % t_c



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# General functions for use in many models
def N_root_values(N, func, int_func, symmetric=False):
	# Given a function func, find its roots on the intervals
	# guessed by int_func. Return a list of N-many roots
	# Kwarg 'symmetric' - if True, return roots in range (-N, N)
	out = []
	if symmetric:
		lower = -N
	else:
		lower = 0
	for m in range(lower, N):
		a, b = int_func(m)
		# DEGUG
		## "(%5.2f, %5.2f): f(a)=%7.3f, f(b) = %7.3f" %(a,b,func(a),func(b))

		root = scipy.optimize.brentq(func,a,b)
		out.append(root)
	return np.array(out)

def loaded_func(tupl, func):
	# Sketchy sugar tool for loading params into root function
	# tupl is a tuple of arguments to be passed to the inner root-finding function
	def inner(x):
		return func(x,tupl)
	return inner

def sigma():
	return 5.67E-8 # Stefan-Boltzman constant [W/(m^2*K^4)]

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#Parker (1961)
def Parker_u(t, N, package, buff=4):
	"""
	Compute the analytical solution for a known diffusivity
	Finds the temperature at z = L
	Params:
		t - array representing full time
		N - terms to carry in Parker's summation
	Kwargs:
		buff - number of initial temp values to set to zero. This gets rid of
				 N-dependent diverging behavior at t ~ 0
	"""
	u0, Q, L, __, k, rho, c_p, __, __, __ = package
	alpha = k/(rho*c_p) # sample diffusivity [units??]

	summ = 0
	for i in range(N):

		n = i + 1 # Account for index from zero
		term = (-1)**n*np.exp(-n**2*np.pi**2*alpha*t/L**2)
		summ += term

	out = (1 + 2*summ)*Q/(rho*c_p*L)
	out[:buff] = 0	   #FIXME: zeroth term buggy based on N...
	return out + u0

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#Cowan (Cowan, 1962 )

def Cowan_u(t, N, package, z=None):
	# Compute the full Cowan temperature for a given set of params
	# Compute up to N values over an array of times 't'
	u0, Q, L, R, k, rho, c_p, eps_z1, eps_z2, __ = package
	T_inf = Q/(rho*c_p*L)
	alpha = k/(rho*c_p)
	if z == None:
		z = L

	# (1) Physical/BC constants
	a, b, r = a_b_r(u0, L, eps_z1, eps_z2, k)

	# (2) Rootfinding for y_n
	yvals = N_y_root_values(N, a, b)

	summ = 0
	for n in range(N):
		yn = yvals[n]
		# Non-radiative limiting case
		if a == 0:
			# Avoid Zero Division errors by applying proper limits
			if n == 0:
				summ += .5*np.cos(yn*(L-z)/L) #cosine term, apply y0->sqrt(a)
												#this kills the sine term
			else:
				# This is a pidgin of limiting values, since roots of yn
				# will pick up proper limiting behavior but ratios of
				# yn to Dn will not.
				summ += np.exp(-alpha*yn**2*t/L**2)*np.cos(yn*(L-z)/L)*(-1)**n

		elif a > 0:
			#No diverging denominators for T0 > 0; hard-code original soltn
			A = np.cos(yn*(L-z)/L) # cosine term
			B = a*np.sin(yn*(L-z)/L)/(yn*(1+r))
			top = yn**2*( A + B )*np.exp(-alpha*yn**2*t/L**2)
			bot = D_n(n,a,b,yvals)
			# # "term", n
			# # "bot", bot
			# # "top", top
			# # n, top[:5]/bot
			summ += top/bot

	return u0 + 2*T_inf*summ

# (1) Boundary condition parameters
def a_b_r(u0, L, eps_z1, eps_z2, k):
	# FIXME: rad. BC's aren't dependent on separate face temps values yet
	c0 = 4*sigma()*eps_z1*u0**3/k
	cL = 4*sigma()*eps_z2*u0**3/k
	# Note: cL refers to the _front_ face, c0 is the _rear_ face
	a = L*(c0+cL)
	b = c0*cL*L**2
	try:
		r = cL/c0
	except ZeroDivisionError:
		#FIXME should converge to ratio of eps_z (front/back)?
		r = 1

	return a, b, r

# (2) root finding
def N_y_root_values(N, a, b):

	if a == 0:
		return np.array([n*np.pi for n in range(N)])
	else:
		## loaded_y_func(a,b,y_root)()
		return N_root_values(N, loaded_func((a,b), y_root), y_root_guesses)

def y_root(x, ab):
	#Pass over yn = 0, as it should not be a root
	# ab is a tuple of (a,b)
	try:
		k = 1/x
	except ZeroDivisionError:
		return 0
	else:
		return cot(x) + ab[1]/(ab[0]*x) - x/ab[0]

def y_root_guesses(m):
	#FIXME: Very high sensitivity to guessed range at large m...
	# Currently works up to 20
	# A function that guesses the interval where the mth root of y_root lies
	return (m*np.pi+.00000001, (m+1)*np.pi-.000000001)

def cot(x):
	try:
		out = 1/np.tan(x)
	except ZeroDivisionError:
		out = np.nan
	return out

# (3) Series coefficients
def D_n(n, a, b, y_lst):
	"""
	Caluclate D_n from Cowan's paper. Supports limit for a->0
	"""
	# non-radiative case
	if a == 0:
		if n == 0:
			return 0
		else:
			return (n**2)*(np.pi**2)*(-1)**n
	elif a > 0:
		yn = y_lst[n]
		BB = (1 + a - (2*b)/a + yn**2/a +b/yn**2 + b**2/(a*yn**2))
		## "in Dn", yn, BB
		return yn*np.sin(yn)*BB

def print_Cowan_params(package):
	u0, Q, L, R, k, rho, c_p, eps_z1, eps_z2, __ = package

	a,b,r = a_b_r(u0, L, eps_z1, eps_z2, k)

	N = 10
	yvals = N_y_root_values(N, a, b)
	plt.figure()
	plt.plot(range(10), yvals)
	plt.show()
	return

########################################################################
########################################################################
# Rectangle solution

def Rectangle_u(t, Mx, My, Nz, rect_package, x=None, y=None, z=None, buff=8):
	"""
	Generate a solution to radiative boundary conditions for LFM on a
	parallelpiped object.
	Args:
		t - array of times
		Mx, My, Nz - max iters on X, Y, Z
		rect_package - set of experiment characteristics
		x, y - coordinates on rear face for temperature. Default x=0, y=0
		z - depth at which to assess temperature. Default z=0

	"""

	"""
	buffer characteristics:
		- smaller buffer required for larger Nz
		- buffer is index-based; use good judgement compared to expected
			position of half rise time = t_c
	"""
	u0, Q, L, a, b, k, rho, c_p, eps_z1, eps_z2, eps_x, eps_y = rect_package

	alpha = k/(rho*c_p)
	#_rect_params(rect_package)

	# k, eps_y
	# default to center of the rear face
	if z == None:
		z = L
	if x == None:
		x = 0
	if y == None:
		y = 0

	# (1) Establish constants
	# Sey Y = h*a
	Y_y = make_Y(u0, b, eps_y, k)
	Y_x = make_Y(u0, a, eps_x, k)
	Y_z1 = make_Y(u0, L, eps_z1, k)
	Y_z2 = make_Y(u0, L, eps_z2, k)

	# (2) Rootfinding for gamma_l
	x_gammas = N_gamma_root_values(Mx, Y_x)
	y_gammas = N_gamma_root_values(My, Y_y)
	z_betas = N_X_values(Nz, Y_z1, Y_z2)

	# (3) Spatial temperature distribution
	# For backwards compatibility, ignore x-, y-, and z0 convolutions
	# (added integral terms if h== 0):
	x_convo = 1
	y_convo = 1
	z_convo = 1

	Z = L - z # special z coordinates ugh

	# (4) iterate on Mx, My, N

	summ = 0
	for mx in range(0, Mx):

		# X-DEPENDENCE
		g_mx = x_gammas[mx]
		exp_x = np.exp(-g_mx**2*alpha*t/a**2)

		if mx == 0 and Y_x == 0:
			# non-radiative case only has the integrated term added
			xterm = x_convo*np.ones(len(t))
		else:
			# otherwise, sum grabs index-0 == first non-zero root
			# radiative case: '0th' term is actually m=1
			xterm = C_m_rect(Y_x, g_mx, a)*np.cos(g_mx*x/a)*exp_x
			## "xterm %i" % mx, xterm
		## "exp_x", exp_x[:6]
		for my in range(0, My):

			# Y-DEPENDENCE
			g_my = y_gammas[my]
			exp_y = np.exp(-g_my**2*alpha*t/b**2)
			## "exp_y", exp_y[:6]

			if my == 0 and Y_y == 0:
				yterm = y_convo*np.ones(len(t))
			else:
				yterm = C_m_rect(Y_y, g_my, b)*np.cos(g_my*y/b)*exp_y

			for nz in range(0, Nz):
				# Z-DEPENDENCE
				b_n = z_betas[nz]
				exp_z = np.exp(-b_n**2*alpha*t/L**2)
				## "exp_z", exp_z[:6]

				# Non-radiative: the '0th' term is a constant
				if nz == 0 and Y_z1 == 0 and Y_z2 == 0:
						zterm = z_convo*np.ones(len(t))
				# radiative: The index=0 root in beta should be the first
				#	POSTIVE root solution i.e. the n=1 term in the summation
				else:
					zterm1 = D_n_rect(Y_z1, Y_z2, b_n)
					zterm2 = ( b_n*np.cos(b_n*Z/L) + Y_z1*np.sin(b_n*Z/L) )
					zterm = exp_z*zterm1*zterm2*2

				# Each component of the sum is calculated individually
				xyz_term = xterm*yterm*zterm
				summ += xyz_term
				## "term %s%s%s" % (mx,my,nz), xyz_term[:6], "\n"

	total = u0 + summ*Q/(rho*c_p*L)
	total[:buff] = u0	  #FIXME: zeroth term buggy based on N...

	# Buffer terms are wiped to account for divergin behavior at t=0. This is
	# the case in the Parker solutoin as well.

	# "total", total
	return total
			# NOW: do the Cape-Lehman solutions......

	return

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# x- and y- dependence terms
def make_Y(T0, x, eps, k):
	# Defining Y = h*a
	h = 4*sigma()*eps*T0**3/k
	return x*h

def N_gamma_root_values(N, Y):

	if Y == 0:
		# positive roots according to Carslaw..
		# INCLUDE A BUFFER N=0 TERM - it will be skipped for non-rad..
		return np.array([n*np.pi for n in range(N)])
	else:
		## loaded_y_func(a,b,y_root)()
		return N_root_values(N, loaded_func((Y), gamma_root), gamma_root_guesses)

def gamma_root(x, Y):
	# eps_a is a tupl of (eps, h)
	return x*np.tan(x) - Y

def gamma_root_guesses(m):
	#FIXME: Very high sensitivity to guessed range at large m...
	# A function that guesses the interval in which gamma roots lies
	# DOES NOT INCLUDE 0! - 0 is a root IFF h = 0; this case is skipped
	# in the definition of N_gamma_root_values

	#These have been verified with Carsloaw appendix IV
	if m == 0:
		return (0, np.pi/2)
	else:
		n = 2*m - 1
		return (n*np.pi/2+.00000001, (n+2)*np.pi/2-.000000001)

# Main coefficient for x, y- radiation
def C_m_rect(Y, gamma, x):
	top = 2*(Y**2 + gamma**2)*np.sin(gamma)
	bot = gamma*(Y**2 + gamma**2 + Y)
	return top / bot

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# z-dependence terms
def X_root(x, Yz):
	# ! Yz is a tuple of (Yz)
	return np.tan(x)*(x**2-Yz[0]*Yz[1]) - x*(Yz[0] + Yz[1])

def X_root_guesses(m):
	#Guess at the location (Xm is about m*pi) (warning: m*pi/2 is a false root.)
	# avoid the zero-root
	if m == 0:
		return (0.0000001, np.pi/2)
	else:
		n = 2*m - 1
		return (n*np.pi/2+.00000001, (n+2)*np.pi/2-.000000001)
	return (a,b)

def N_X_values(N, Yz1, Yz2, approx=False):
	# Solving for the same roots as Cape-Lehman

	# Use only the positive roots in Y= 0 case. There's an additional 0th
	# term tacked on to the sum if Y = 0, so skip X_0
	if Yz1 == 0 and Yz2 == 0:
		# Same buffer term as in gamma
		return np.array([n*np.pi for n in range(N)])
	else:
		## loaded_y_func(a,b,y_root)()
		return N_root_values(N, loaded_func([Yz1, Yz2], X_root), X_root_guesses)

def D_n_rect(Yz1, Yz2, beta):
	top = (beta**2 + Yz2**2)*(beta*np.cos(beta) + Yz1*np.sin(beta))
	bot = (beta**2 + Yz1**2)*(beta**2 + Yz2**2 + Yz2) + Yz1*(beta**2 + Yz2**2)
	return top / bot


def omega_lmn(gamma_l, gamma_m, beta_n, a, b, L):
	return gamma_l**2/a**2 + gamma_m**2/b**2 + beta_n**2/L**2


def print_rect_params(package, approx=False):
	u0, Q, L, a, b, k, rho, c_p, eps_z1, eps_z2, eps_x, eps_y = package
	alpha = k/(rho*c_p)
	dump_params_package(package, rect=True)

	# (1) Establish constants
	# Sey Y = h*a
	Y_y = make_Y(u0, b, eps_y, k)
	Y_x = make_Y(u0, a, eps_x, k)
	Y_z1 = make_Y(u0, L, eps_z1, k)
	Y_z2 = make_Y(u0, L, eps_z2, k)

	# "Y_x = %2.6f" % Y_x
	# "Y_y = %2.6f" % Y_y
	# "Y_z1 = %2.6f" % Y_z1
	# "Y_z2 = %2.6f" % Y_z2

	# (2) Rootfinding for gamma_l
	x_gammas = N_gamma_root_values(7, Y_x)
	y_gammas = N_gamma_root_values(7, Y_y)
	z_betas = N_X_values(7, Y_z1, Y_z2)

	# "gamma_x"
	# x_gammas

	# "gamma_y"
	# y_gammas

	# "beta_z"
	# z_betas


	D = [D_n_rect(Y_z1, Y_z2, b)/b for b in z_betas]
	Cx = [C_m_rect(Y_x, g_mx, a) for g_mx in x_gammas ]
	Cy = [C_m_rect(Y_y, g_my, b) for g_my in y_gammas ]
	omega = []
	# "D_n/beta"
	# D

	# "Cx"
	# Cx

	# "Cy"
	# Cy

	return










# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Cape and Lehman model
def Cape_Lehman_u(t, M, I, package, r=None, approx=False):
	"""
	Compute the full Cowan temperature for a given set of params
	Args:
		M - Number of terms to include in outer summation (over CmXm)
		I - Number of terms in the inner summation (over D_i)
		t - array of time values to compute temp for
		package - parameters package of thermophysical data
	Kwargs:
		r - radial location of temperature calculation
	"""

	u0, Q, L, R, k, rho, c_p, eps_z1, eps_z2, eps_r = package
	T_inf = Q/(rho*c_p*L)
	alpha = k/(rho*c_p)
	if r == None:
		r = 0

	#_CL_params(package, approx=approx)

	# (1) Physical/BC constants
	Yz = get_Y(u0, eps_z1, k, L)
	Yr = get_Y(u0, eps_r, k, R)

	# (2) Rootfinding for C_m, X_m, Z_i
	# Note: Use the positive roots of Xm, CmXm!
	X = N_X_values(M, Yz, approx=approx)
	CmXm = M_CmXm_root_values(M+1, alpha, L, Yz, approx=approx)
	Zeta = N_Zeta_values(I, Yr)
	D = [D_i(i, r, R, Yr, Zeta) for i in range(I) ]

	# (3) Put it all together in the summation
	outer_sum = 0
	for m in range(M):
		inner_sum = 0
		for i in range(1):
			# "omega_%i%i" % (i, m), omega(Zeta[i], X[m], L, R)
			inner_sum += D[i]*np.exp(-omega(Zeta[i], X[m], L, R)*alpha*t)
		outer_sum += CmXm[m]*inner_sum
		# "final inner sum: ", inner_sum[:10]*CmXm[m]
		# "final outer sum: ", outer_sum[:10]

	# TODO: Find out why this temp is spiking for ODD M summateions
		# should be near the final m-term; suggests CmXm does not converge to +/-const
	# Idea: Pull C0X0 out
	# Idea: implement the approximation version..
	#outer_sum +=
	return Q*outer_sum/k + u0



	return
	for n in range(N):
		yn = yvals[n]
		# Non-radiative limiting case
		if a == 0:
			# Avoid Zero Division errors by applying proper limits
			if n == 0:
				summ += .5*np.cos(yn*(L-z)/L) #cosine term, apply y0->sqrt(a)
												#this kills the sine term
			else:
				# This is a pidgin of limiting values, since roots of yn
				# will pick up proper limiting behavior but ratios of
				# yn to Dn will not.
				summ += np.exp(-alpha*yn**2*t/L**2)*np.cos(yn*(L-z)/L)*(-1)**n

		elif a > 0:
			#No diverging denominators for T0 > 0; hard-code original soltn
			A = np.cos(yn*(L-z)/L) # cosine term
			B = a*np.sin(yn*(L-z)/L)/(yn*(1+r))
			top = yn**2*( A + B )*np.exp(-alpha*yn**2*t/L**2)
			bot = D_n(n,a,b,yvals)
			# # "term", n
			# # "bot", bot
			# # "top", top
			# # n, top[:5]/bot
			summ += top/bot

	return u0 + 2*T_inf*summ

def get_Y(T, eps, k, D):
	# Calculate Biot(?) number BC for first-order rad. boundary condition
	return 4*sigma()*eps*T**3*D/k

def X_root(x, Yz):
	# ! Yz is a tuple of (Yz)
	return np.tan(x)*(x**2-Yz[0]**2)-2*x*Yz[0]
def X_root_guesses(m):
	#Guess at the location (Xm is about m*pi) (warning: m*pi/2 is a false root.)
	a = m*np.pi - .3
	b = m*np.pi + .3
	return (a,b)
def N_X_values(N, Y_z, approx=False, symmetric=False):
	out = N_root_values(N, loaded_func( (Y_z,), X_root), X_root_guesses, symmetric=symmetric)
	if approx:
		X0 = (1 - Y_z/12 + 7*Y_z**2/288)*(2*Y_z)**.5
		out[0] = X0
	return out

def M_CmXm_root_values(N, alpha, L, Y_z, approx=False):
	# Rootfinding is packaged under here because of funny limits
	if Y_z == 0:
		# Cape and Lehman's limits for roots of Cm*Xm (footnote 7)
		CmXm = [2*alpha*(-1)**m/L for m in range(N)]
		CmXm[0] = CmXm[0]/2
	else:
		X = N_X_values(N, Y_z)
		CmXm = []
		for m in range(N):
			# Try some approximation from C-L eqn (23)
			if m == 0 and approx==True:
				X0 = (1-Y_z/12+7*Y_z**2/288)*(2*Y_z)**.5
				C0X0 = X0**2 / (X0**2 + 2*Y_z + Y_z**2)
				CmXm.append(C0X0)
			else:
				a = (2*alpha*(-1)**m)/L
				b = X[m]**2/(X[m]**2 + 2*Y_z + Y_z**2)
				CmXm.append(a*b)

	return np.array(CmXm)


def Zeta_root(x, Yr):
	return Yr[0]*jv(0,x)-x*jv(1,x)
def Zeta_root_guesses(m):
	# mth root is about equal to mth root of J_1(x) for Yr->0
	J1_roots = jn_zeros(1,m+1)
	if m == 0:
		a = -.001
		b = 2
	else:
		# Python does not call the origin a root of J_1(x)...
		a = J1_roots[m-1] *.8
		b = J1_roots[m-1] *1.2
	return (a,b)

def N_Zeta_values(N, Y_r):
	# Solve for the zeta coefficients
	# Return dummy solution for Y_r = 0 -> D_i is agnostic to Z_i for Yr=0
	if Y_r == 0:
		return np.array([0 for i in range(N)])
	else:
		Z = N_root_values(N, loaded_func( (Y_r,), Zeta_root), Zeta_root_guesses)
		return Z

def D_i(i, r, R, Y_r, Zeta):
	# D_i coefficient for Cape and Lehman
	# Zeta is an array of Zeta values
	if Y_r == 0:
		# The proper limits rely on J1(x)/x -> 1/2 for x-> 0
		# See supporting documentation for explanation
		if i == 0:
			return 1
		else:
			return 0
	elif Y_r > 0:
		a = 2*Y_r**2/(Y_r**2 + Zeta[i]**2)
		b = jv(0, Zeta[i]*r/R)/jv(0,Zeta[i])**2
		# b->1 at the center, etc.
		## "i=%i, Z_i=%6.5f, a=%4.2f, b=%4.2f" %(i, Zeta[i], a, b)

		return a*b

def omega(Zeta_i, X_m, L, R):
	# MODIFIED FROM CAPE-LEHMAN DEFINITION
	return ( (X_m/L)**2 + (Zeta_i/R)**2 )

def print_CL_params(package, approx=False):
	u0, Q, L, R, k, rho, c_p, eps_z1, eps_z2, eps_r = package
	alpha = k/(rho*c_p)
	dump_params_package(package)

	r = 0
	str_bool = "True" if approx else False
	# "Using approximate values: %s" % str_bool


	Yz = get_Y(u0, eps_z1, k, L)
	# "Y_z = %2.6f" % Yz
	Yr = get_Y(u0, eps_r, k, R)
	# "Y_r = %2.6f" % Yr

	X = N_X_values(10, Yz, approx=approx)
	# "X"
	# X

	CmXm = M_CmXm_root_values(10, alpha, L, Yz, approx=approx)
	# "Cm*Xm"
	# CmXm

	Zeta = N_Zeta_values(10, Yr)
	# "Zeta_i"
	# Zeta

	D = [D_i(i, r, R, Yr, Zeta) for i in range(8) ]
	# "D_i"
	# D
	# N = 10
	# yvals = N_y_root_values(N, a, b)
	# plt.figure()
	# plt.plot(range(10), yvals)
	# plt.show()
	return

def Bart_approx(t, package):
	u0, Q, L, R, k, rho, c_p, __, __, __ = package
	alpha = k/(rho*c_p) # sample diffusivity [units??]

	# Compute the shape factor
	f_shape = ((1/L**2) + jn_zeros(0,1)[0]**2/(np.pi*R)**2)/(2/L + 2/R)**2

	return (1 - np.exp(-
					   (np.pi**2)*alpha*t*f_shape/L**2) )*Q/(rho*c_p*L)

def bart_parker(t,package):
	bart = Bart_approx(t, package)
	parker = Parker_u(t, 10, package)
	fig = plt.figure(facecolor="white")
	ax = fig.add_subplot(111)
	ax.plot(t, bart, lw=2.5, ls="--" )
	ax.plot(t, parker, lw=2.5)
	ax.xaxis.grid(True,'minor')
	ax.yaxis.grid(True,'minor')
	ax.xaxis.grid(True,'major',linewidth=2)
	ax.yaxis.grid(True,'major',linewidth=2)
	plt.legend(["Bart", "Parker"], loc=4)
	plt.show()

	
df = main()	
df.to_csv('rectangle_results.csv')