from sqlalchemy.dialects.mssql.information_schema import columns

import my_math as mm
import numpy as np


#################################  Section2 ####################################
def total_cost(n,retail_price):
    if n > 10:
        tcost = n * retail_price *(2.0/3.0)
    elif n > 4 and n < 10:
        tcost = n * retail_price * (13.0/15.0)
    else:
        tcost = n * retail_price
    return tcost

def sum_x_squared(x,y):
    sum = 0
    for i in range(x,y+1):
        sum+=i**2
    return sum

def sum_x_squared_while(lim=500):
    sum = 0
    i = 0
    while(sum<lim):
        summ1=sum
        sum += i**2
        if sum > lim:
            return summ1, i-1
        i += 1
    return sum, i

def factorial(x):
    if x == 0:
        return 1
    else:
        return x * factorial(x-1)


if __name__ == '__main__':
    """ # %% md
    #### Exercise 2.1
    # A bakery sells cupcakes for €1.50 each.
    However, they offer discounts such that the more you buy the cheaper they are.
    If you buy more than 10 cupcakes they cost €1.00 each, 
    or if you buy more than 4 but less than 10 they cost €1.30 each.
    Write a set of ` if / elif / else ` statements to calculate the total cost of the
     order for $n$ cupcakes.Set $n = 3$, $n = 7$ and $n = 12$ to confirm that you 
     get the correct answer.
"""
    retail_price = 1.50
    print( f"The cost of 3 cupcakes is €{total_cost(3,retail_price):.2f}")
    print( f'The cost of 7 cupcakes is €{total_cost(7,retail_price):.2f}')
    print( f'The cost of 12 cupcakes is €{total_cost(12,retail_price):.2f}')
    """#%% md
    #### Exercise 2.2 

    Write a `for` loop to calculate the sum of $x^2$ for $x$ from 0 to 9, $\sum_{x=0}^9 x^2$.
"""
    print(f"The sum of x^2 between 0 and 9 is {sum_x_squared(0,9)}")

    """#%% md
     Exercise 2.3 Alter your for loop from Exercise2 to a whileloop,which again calculates ∑
    x=0x2,
    but terminates  instead when    the su m    is  greater then 500."""
    while_sum, index = sum_x_squared_while(500)
    print(f"The sum of x^2 between 0 and {index} is {while_sum}")


    """Exercise 2.4 Write a function factorial to compute the factorial of x. The input argument should
 be x and the function should return x!. For example, the factorial of 5, 5! = 5×4×3×2×1 = 120"""

    #x=input("Please enter the number whose factorial you require:\n")
    x=5
    res = factorial(int(x))
    print(f"The factorial of {x} is {res}")


    """Exercise 2.4a - create a file called my_math.py containing function to print the odd numbers
    between 0 and n and the even numbers between 0 and n
    and import it such that its functions can be used in main"""
    n = 20
    print(f"the odd numbers between 0 and {n} are:\n")
    mm.odd_numbers(n)
    print("\n")
    print(f"the even numbers between 0 and {n} are:\n")
    print("\n")
    mm.even_numbers(n)

    """Exercise 2.5 Write a function *multiples_three* that creates and returns a 1D numpy 
    array with the first *n* multiples of 3, where *n* is the function input."""

    def multiples_three(n):
        return 3*np.array(range(1,n+1))
        #or return np.array[3*i for in in range(1,n+1)]

    print(f"The first 10 multiples of 3 are:\n{multiples_three(10)}")

    my_array_2d = np.array([range(0, 5), [10] * 5])
    print(my_array_2d)
    print(np.zeros((5, 4)))
    print(np.ones((4, 6)))

    """Exercise 6
    Write a function *matrix_index* that returns the entry in the *n*th row and 
    *m*th column of a matrix *X*. The function inputs should be *X*, *n* and *m* (in that order).
    For example, if $$X = \begin{pmatrix} 7 & 3 \\ 1 & 9 \end{pmatrix},
    $$ `matrix_index(X,1,2)` should return 3."""
    def matrix_index(X,n,m):
        return X[n-1,m-1]
    my_matrix = np.array([[7,3],[1,9]])
    print(matrix_index(my_matrix,2,2))
    print(matrix_index(my_matrix, 1, 2))


#################################  Section3 ####################################
    #Numpy and Pandas
    #simple matrix generation
    A = np.array((range(5),[10]*5))
    B = np.random.rand(5,5)
    C = np.ones((5,2))
    D = np.eye(5)

    print(A)
    print(B)
    print(C)
    print(D)

    """Exercise1 Create the followin gmatrices in Python,
 X=
 1 2 3 4
 2 4 6 8
 3 6 9 12

Y=
 2 2 2
 2 2 2
 2 2 2
 2 2 2
 and compute the matrix products XY andYX."""
    X = np.array([[1,2,3,4],[2,4,6,8],[3,6,9,12]])
    #Y = np.array([[2,2,2],[2,2,2],[2,2,2],[2,2,2]])
    Y = np.array([[2]*3,[2]*3,[2]*3,[2]*3])
    print(X)
    print(Y)
    print(f"XY = {X.dot(Y)}")
    print(f"XY = {np.dot(X,Y)}")
    print(f"YX = {Y.dot(X)}")
    print(f"YX = {np.dot(Y,X)}")

    #other matrix operations
    print(D.diagonal())
    print(D.trace())
    print(f"A transpose {A.transpose()}")
    print(f"A transpose alt {A.T}")

    #Need linear algebra package for other operations
    import numpy.linalg as npl
    print(npl.inv(B))
    print(f"The B.invB{B.dot(npl.inv(B))}")
    B_invB_rounded = np.round(B.dot(npl.inv(B)),2)
    B_invB_rounded[B_invB_rounded <1e-6] = 0
    print(f"B_invB_rounded{B_invB_rounded}")
    print(npl.inv(D))
    print(f"The D.invD{D.dot(npl.inv(D))}")
    #determinant
    #- **Determinant**: Indicates invertibility, scaling,
    # #and orientation-changing properties of the transformation represented by the matrix.

    print(f"Determinant of B is {npl.det(B)}")

    # eigenvalues
    #- **Eigenvalues**: Indicate scaling factors and stability
    # along specific directions (eigenvectors).
    # They have applications in understanding dynamics, transformations, and data compression.
    #Applications also in Physics, e.g. calculating energy levels in quantum mechanics

    print(f"Eigenvalues of B are:{npl.eigvals(B)}")


    #matrix decomposition
    M = np.array([[3, 6, 1], [2, 3, 7], [9, 2, 5]])
    my_qr = npl.qr(M)  # Extract matrices
    print(f"Q matrix\n{my_qr[0]}")   # Q matrix
    print(f"R matrix\n{my_qr[1]}")   # R matrix
    print(f"Check decomposition\n{my_qr[0].dot(my_qr[1])}")   # Check decomposition

    """Exercise 2 Write the system of equations
    2x +y−7z =10 6x−2y−z =5 x−5y+2z=8  as a matrix system and 
    use the solve function to solve for x, y and z."""

    A = np.array([[2,1,-7],[10,-2,-1],[1,-5,2]])
    B = np.array([10,5,8])
    print("solutions")
    X = npl.solve(A,B)
    print(X)
    print(A.dot(X))
    print(B)
    x=X[0]
    y=X[1]
    z=X[2]
    print(f"check {2*x+y-7*z}")


    #A = np.random.rand(5,5)
    #print(A)
    #B = np.random.rand(5,2)
    #print(B)
    #print(A.dot(npl.solve(A,B)))

    #random sampling
    import numpy.random as npr
    #random matrix of dim 4,5 from standard Normal dist N(0,1)
    print(npr.randn(4,5))
    print(np.mean(npr.randn(4,5)))
    # random matrix of dim 2,2 from uniform dist U(0,1)
    print(npr.rand(2,2))

    print(npr.normal(10, 2, size=(3, 2)))
    print(npr.randint(0, 100, size=(2, 6)))

    #random samples
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    print(npr.permutation(x))
    print(x)
    npr.shuffle(x)
    print(x)
    print(npr.choice(x, 2))

    #random seed
    npr.seed(1234)
    print(npr.rand())


    #Pandas - series and dataframes, data analysis

    import pandas as pd
    from pandas import Series, DataFrame

    #series is a 1d version  of a dataframe

    my_series = Series([1, 2, 3, 4, 5])
    print(my_series)
    print(my_series[0])
    print(my_series[1:3])
    print(my_series[[1, 3]])
    print(my_series[my_series > 3])
    print(my_series.index)
    print(my_series.values)
    print(my_series.describe())

    my_series_2 = Series([6, 9, 2, 5], index=['a', 'd', 'b', 'c'])
    print(my_series_2)

    print(my_series_2['c'])

    #dataframe is a 2d series, to create simply pass a nested list or a dict to
    #the DataFrame function

    dict1 = {'PhoneType':['iPhone','iPhone 3G', 'iPhone 3GS', 'iPhone 4','iPhone 4S','iPhone X'],'memory_MB':[128,128, 256, 512,512,1024], 'weight_g':[135, 133,135, 137,140,112],'camera_MP':[2, 2, 3,5,8, 8],'year':[2007,2008, 2009,2010,2011, 2012]}

    iphone_df = DataFrame(dict1)
    print(iphone_df)

    #print(iphone_df['memory_MB'].values)
    #print(iphone_df['memory_MB'])
    #print(iphone_df.memory_MB)
    #print(iphone_df.PhoneType)

    print(iphone_df.iloc[2:6,:3])
    print(iphone_df.iloc[0:,2])

    #Passing by reference
    #If part of a dataframe is extracted to another and that extracted dataframe
    #is changed, the original dataframe is changed also, unless the copy method is used

    df_2 = DataFrame({'x': [0, 1, 2, 3], 'y': [4, 5, 6, 7], 'z':[8, 9, 10, 11]}, index = ('a', 'b', 'c', 'd'))
    print(df_2)
    sample = df_2.x
    sample[0:2] = 10
    print(df_2)

    df_3 = DataFrame({'x': [0, 1, 2, 3], 'y': [4, 5, 6, 7], 'z':[8, 9, 10, 11]}, index = ('a', 'b', 'c', 'd'))
    sample2 = df_3.x.copy()
    sample2[0:2] = 10
    print(df_3)


 #Exercise 5 Take the data from the table below and store it as a pandas DataFrame. Use the
 #Name column as the row index, and the remaining column headings as the column indices. After
 #creating the DataFrame, print out Darren’s height.

 #name,age,height,weight
 #Aaron,23,1.7,64
 #Barry,36,2.1,99
 #Catherine,27,1.8,68
 #Darren,19,1.9,85
 #Edel,41,1.7,102
 #Francis,38,2.0,84
 #George,57,1.8,87
 #Helen,17,1.6,90
 #Ian,28,1.7,78


df_4 = DataFrame({'age':[23,36,27,19,41,38,57,17,28],'height':[1.7,2.1,1.8,1.9,1.7,2.0,1.8,1.6,1.7],'weight':[64,99,68,85,102,84,87,90,78]},index=('Aaron','Barry','Catherine','Darren','Edel','Francis','George','Helen','Ian'))

print(df_4)

print(df_4['height'].loc['Darren'])

print(df_4.iloc[3,1])

print(df_4.loc['Darren','height'])


#################################  Section4 ####################################
###############################Loading and saving data#######################
print("######  Section4 ########")
path = 'C:\\Users\\brian\\OneDrive\\Documents\\MSC Data Analytics\\Data Prog with Python\\Full course\\'

f = open(path + 'sample1.txt')
print(f.read())
f.close()

f= open(path + 'sample1.txt','r')
lines = [i for i in f]
print(lines)
f.close()


f= open(path + 'sample1.txt','r')
lines = [int(i) for i in f]
print(lines)
f.close()


# The file sample2.txt contains a slightly more complicated data set where one of the lines contains
# two elements and we want to read it in as [[abc],[de,f],[g]]. We could use a comprehension, but it
# would be a little harder to follow, so we will use a for loop instead


f = open(path + 'sample2.txt')
data = []
for line in f:
    lines = line.strip('\n').split(" ")
    data.append(lines)

print(data)

f.close()

###Exercise 1 Adapt the above code above to import integers instead of characters and use it to
# import the sample3.txt data set.

data2 = []
f= open(path+'sample3.txt')

for i in f:
    lines = i.strip('\n').split(" ")
    data2.append([int(j) for j in lines])

print(data2)
f.close()

#file sample 5 contains a mixture of characters and numbers which need to be
#assessed when reading
data = []
for line in open(path+'sample4.txt'):
    items = line.rstrip('\n').split(' ')
    curr_items = []
    print(items)
    for j in items:
        if j.isalpha():
            curr_items.append(j)
        else:
            curr_items.append(int(j))
    data.append(curr_items)
print(data)

#Create a dict from sample5.txt which has the following format
#John 25
#Mary 28
#Jim 19

data = {}

for line in open(path+'sample5.txt'):
    data[line.split(' ')[0]] = int(line.split(' ')[1])
print(data)
## Exercise 2 Adapt the above code above to allow for more than 2 entries per line. The first entry
# should be used as the key and the remaining entries used as the values. Now use your code to
# import the file sample6.txt

data2 = {}
for line in open(path+'sample6.txt'):
    items = line.rstrip('\n').split(' ')
    print(items)
    data2[items[0]] = items[1:]
print(data2)

#load data as an array
import numpy as np
data = np.loadtxt(path+'array.txt',delimiter=',')
print(data)

#import numpy as np
#data = np.loadtxt(path+'array_test.txt',delimiter=',')
#print(data)

#data3 = []
#f = open(path+'array_test.txt')
#for i in f:
#    lines = i.strip('\n').split(',')
#    data3.append([int(j.split("'")) for j in lines])

#print(data3)
#f.close()

an_array = np.arange(20)
np.save(path+'my_array',an_array)

an_array_2 = np.load(path+'my_array.npy')
print(an_array_2)
print(type(an_array_2))


#import data as dataframe

df_1 = pd.read_csv(path+'sample_csv.csv')
print(df_1)

#for tab separated
df_2 = pd.read_csv(path+'sample_dat.dat',sep='\t')
print(df_2)

df_3 = pd.read_csv(path+'array.txt',sep=',')
print(df_3)

#print(df_3[1][1])

df_3= pd.read_csv(path+'array.txt',sep=',',names=['a','b','c','d','e'])
print(df_3)
print(df_3['a'][1])

#other variations and options
df_2= pd.read_csv(path+'sample_dat.dat',sep='\t',skiprows=[0],names=['x','y'],nrows=4)
print(df_2)

df_1= pd.read_csv(path+'sample_csv.csv',index_col='Name')
print(df_1)

print(df_1.loc['Brian']['Height'])
print(df_1.loc['Brian','Height'])

my_missing = ['NA','NULL','-999','-1']
diamonds = pd.read_csv(path+'diamonds.csv',na_values=my_missing)
print(diamonds.head())

#Exercise 4 Download the diamonds.csv data set and write a piece of code that imports the first
# 8 rows and the columns ‘carat’, ‘cut’, ‘depth’ and ‘table’ as a DataFrame. You should use the depth
# column as the row index
df_diamonds = pd.read_csv(path+'diamonds.csv',na_values=my_missing,nrows=8,index_col='depth',usecols=['carat','cut','depth','table'])
print(df_diamonds)


#Write dataframes to other formats

my_series = Series(np.arange(10),name='x')
my_series.to_csv(path+'my_series.csv',header=True)
my_df = DataFrame({'a':np.arange(5),'b':np.arange(5,10)})
my_df.to_csv(path+'my_df.csv')

df_covid = pd.read_excel(path+'covid-cases.xlsx',skiprows=3,index_col=0,parse_dates=True)
print(df_covid.tail(10))


#Loading data from APIs

import sys
import wbdata as wbd

country1 = ['IE']
indicator1 = {'SP.POP.TOTL': 'TotalPopulation'}
data1 = wbd.get_dataframe(indicator1, country1)
#print(data1)

country2 = ['US', 'CN']
indicator2 = {'SP.POP.TOTL': 'TotalPopulation', 'SP.POP.GROW': 'Populationgrowth (annual %)'}
data2 = wbd.get_dataframe(indicator2, country2).unstack(level=0)
#print(data2)

#print(wbd.get_countries())

#print(wbd.get_indicators()[1:2])

#################################  Section 5 ####################################
###################################Plotting######################################

print("######  Section5 ########")

import matplotlib.pyplot as plt
"""
plt.figure()
#plt.plot(diamonds.carat,diamonds.price,'r.')
#plt.plot(diamonds.carat,diamonds.price,color='g',marker='*',linestyle='none')
#plt.plot(diamonds.carat,diamonds.price,color='r',marker='.',linestyle='none')
plt.scatter(diamonds.carat,diamonds.price,color='g',marker='*',label='Price')
plt.plot(diamonds.carat,2*diamonds.price,color='r',marker='.',linestyle='none',label='Double Price' )
plt.xlabel('Carat')
plt.ylabel('Price')
plt.title('Scatter plot of price vs carat for Diamonds data',fontsize=14)
plt.xticks(np.linspace(0,5,5))
#plt.yticks([0,10000,20000,30000],['Free!','Cheap','Pricey','Expensive'])
plt.text(3.5, 12000, 'Very big diamond!')
plt.grid(b=True, which='major',color='0.6', linestyle=':')
plt.legend()

plt.show()
##Exercise 2 Write a piece of code to produce a plot of the diamond dimensions x, y and z versus
# the carat. Format your plot such that it matches the plot given here exactly. Pay attention to axis
# labels, markers, legend, etc.


plt.figure()
plt.scatter(diamonds.carat,diamonds.x,color='c',marker='d',label='x')
plt.scatter(diamonds.carat,diamonds.y,color='m',marker='o',label='y')
plt.scatter(diamonds.carat,diamonds.z,color='g',marker='s',label='z')
plt.xlim(0,3)
plt.ylim(0,10)
plt.legend(loc='lower right')
plt.xlabel('Carat')
plt.ylabel('Length')
plt.grid(b=True, which='major',color='0.6', linestyle=':')
plt.yticks(np.linspace(0,10,5))
plt.title('Scatter plot of diamond dimension versus carat',fontsize=14,  fontweight='bold')
plt.savefig(path+'diamond_dimensions.png')
plt.show()


plt.figure(figsize=(8,10))

plt.subplot(2,1,1)
plt.title('Scatter plot of price vs carat for Diamonds data',fontsize=14,fontweight='bold')
plt.plot(diamonds.carat,diamonds.price,'go')
plt.ylabel('Price')
plt.xlabel('Carat')
plt.subplot(2,1,2)
plt.plot(diamonds.depth,diamonds.price,'r+')
plt.title('Scatter plot of price vs depth for Diamonds data',fontsize=14,fontweight='bold')
plt.ylabel('Price')
plt.xlabel('Depth')
plt.show()



#Exercise 3
# Write a piece of code to create a figure of size 10x10, with 4 panels (2 rows of 2). The
# top left panel should be price vs carat, the top right x vs carat, the bottom left y vs carat and the
# bottom right z vs carat. Format your plot such that it matches the plot given here exactly. Pay
# attention to axis labels, markers,etc.

plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
#plt.title('Scatter plot of price vs carat for Diamonds data',fontsize=14,fontweight='bold')
plt.plot(diamonds.carat,diamonds.price,'ro')
plt.ylabel('Price')
plt.xlabel('Carat')
plt.subplot(2,2,2)
plt.plot(diamonds.carat,diamonds.x,color='c', marker='d',linestyle='none')
#plt.title('Scatter plot of price vs depth for Diamonds data',fontsize=14,fontweight='bold')
plt.xlabel('Carat')
plt.ylabel('X dimension')
plt.subplot(2,2,3)
plt.plot(diamonds.carat,diamonds.y,'mo')
plt.xlabel('Carat')
plt.ylabel('Y dimension')
plt.subplot(2,2,4)
plt.plot(diamonds.carat,diamonds.z,color='g', marker='s',linestyle='none')
plt.xlabel('Carat')
plt.ylabel('Z dimension')
plt.show()

#Plotting directly with pandas

diamonds.plot('carat','price',color='r',linestyle='None',marker='*')
plt.ylabel('Price',fontsize=12)
plt.xlabel('Carat',fontsize=12)
plt.title('Scatter plot of price vs carat for Diamonds data',fontsize=12)
plt.axis([0,5,0,20000])
plt.grid(b=True, which='major',color='0.6', linestyle=':')
plt.show()

#Bar charts and histgrams

"""
print(diamonds.color.value_counts())
diamonds.color.value_counts().plot(kind='bar',color='b',alpha=0.4)
plt.ylabel('Count')
plt.xlabel('Color')
#plt.figure()
#plt.bar(diamonds.color.value_counts(),color='g',alpha=0.5)
#plt.title('Bar chart of diamond colour',fontsize=14)
#plt.xticks(np.linespace(0,5,5))
plt.show()
diamonds.price.hist(bins=[0,2000,5000,8000,13000,15000,20000],color='r',alpha=0.5,label='Unequal bins')
diamonds.price.plot(kind='hist',color='b',alpha=0.5,bins=20,label='20 evenly spaced bins')
plt.legend()
plt.show()

#Exercise 4
#Create a histogram for each of the variables x, y and z, and plot them on the same
# graph. Use bins of width 0.5, starting at 2 and ending at 9.
diamonds.x.hist(bins=[2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9],color='b',alpha=0.5,density=True)
diamonds.y.hist(bins=[2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9],color='r',alpha=0.5,density=True)
diamonds.z.hist(bins=[2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9],color='g',alpha=0.5,density=True)
plt.legend(['x','y','z'])
plt.xlabel('Dimension')
plt.ylabel('Density')
plt.title('Histogram of diamond dimensions',fontsize=14)

plt.grid(b=True, which='major',color='0.6', linestyle=':')
plt.show()


"""
diamonds.price.hist(bins=20,density=True,color='b',alpha=0.2)
diamonds.price.plot(kind='kde',style='--',linewidth=2.0)
plt.axis([0,25000,0,0.0004])
plt.show()


#Other types of plots, boxplots etc

diamonds.boxplot('price',by='color')
plt.show()

pd.plotting.scatter_matrix(diamonds, diagonal='kde',color='g',alpha=0.4,figsize=(14,14))
plt.show()
"""


####Seaborn

import seaborn as sns
"""
#sns.set_style('whitegrid')
sns.set_style('darkgrid')
sns.boxplot(x='color',y='price',data=diamonds)
plt.show()

sns.relplot(x='carat',y='price',data=diamonds,kind='scatter',hue='color',style='cut')
plt.xlabel('Carat')
plt.ylabel('Price')
plt.axis([0,3,0,20000])

plt.show()

sns.jointplot(x='carat', y='price', data=diamonds)
plt.show()

#distplot with kde (kernel density estimation) overlaid
sns.distplot(diamonds["price"],hist=True,kde=True,bins=10,norm_hist=True)
plt.show()

sns.relplot(x="carat", y="price", data=diamonds, kind="line", ci='sd')
plt.axis([0,2.5,0,20000])
plt.show()


# Exercise 5.  Read the help file for the seaborn function countplot (sns.countplot?), and use
# it to reproduce the figure given here. Pay particular attention to the palette and saturation
# arguments. You will also need to change the seaborn plotting style.
"""
""" 
sns.set_style('whitegrid')
sns.countplot(x='cut',data=diamonds,palette=['b','c','lightgreen','y','orange'],saturation=0.9)
plt.xlabel('Cut')
plt.ylabel('Number of diamonds')
plt.show()
"""
penguins = sns.load_dataset("penguins")
print(penguins.head())
print(penguins.info())
print(penguins.species.value_counts())
print(penguins.island.value_counts())
print(penguins.sex.value_counts())

sns.scatterplot(data=penguins,x='bill_length_mm',y='flipper_length_mm',hue='species',style='species')
plt.show()

sns.distplot(penguins["body_mass_g"],bins=[2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000],kde=False)
plt.grid(b=True, which='major',color='0.6', linestyle=':')
plt.show()