from numpy import *

def recalc(w):
	mw = array([60.09,79.90,151.99,101.94,159.7,71.85,70.94,40.32,56.08,61.98,94.20])
	nm = array([1,1,2,2,2,1,1,1,1,2,2])
	no = array([2,2,3,3,3,1,1,1,1,1,1])
	p = w / mw
	fac = n / sum(p*no)
	c = fac * p * nm
	csum = sum(c)

	# minimum Fe3 constraints:
	f1 = 16/csum  # >16 cations
	f2 = 8/c[0]   # > 8 Si
	f3 = 15/(csum-c[9]-c[10])   # no Na in M4
	f4 = 2/c[8]   # > 2 Ca
	f5 = 1.0

	# maximum Fe3 constraints:
	f6 = 8/(c[0]+c[3])   # Si+Al<8
	f7 = 15/(csum-c[10]) # No Na in A site
	f8 = 12.9/(csum-c[8]-c[9]-c[10])  # Sum of M1, M2, M3 < 5
	f9 = 36/(46-c[0]-c[3]-c[1]) # Fe3 + Al + Ti in M2 > 2, corrected from paper
	f10 = 46/(c[4]+c[5]+46)  # max fe3, corrected from paper

	fmin = min(f1,f2,f3,f4,f5)
	fmax = max(f6,f7,f8,f9,f10)
	if fmin >1: fmin = 1
	if fmax >1: fmax = 1
	f = (fmin+fmax)/2
	# f = 2*n / (R * c[5] + 2*n)  # ferric ratio can be used instead
	c = f * c

	c[4] = 2 * n * (1 - f)
	c[5] = c[5] - c[4]
	f = c[5] / (c[5] + c[4])
	w[4] = w[5] * (1 - f) * 1.11134
	w[5] = w[5] * f
	return c

# set up terms
ox = array(["SiO2","TiO2","Cr2O3","Al2O3","Fe2O3","FeO","MnO","MgO","CaO","Na2O","K2O"])

# sample analysis below:
w = array([44.75,1.77,0.00,9.76,0.00,18.80,0.32,9.42,11.16,1.44,0.42])

# amphibole: oxygen basis n=23, ferric ratio R=0.2
n = 23
R = 0.2

c = recalc(w)
print("Amphibole recalculation:")
print("{:10} {:9} {:8}".format("oxide","wt %","cat"))
for i in range(11):
	print("{:6} {:8.2f} {:8.3f}".format(ox[i], w[i], c[i]))
print("{:3} {:11.2f} {:8.3f}".format("sum",sum(w),sum(c)))
