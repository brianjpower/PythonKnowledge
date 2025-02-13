import my_math as mm
import numpy as np


#################################  Week2 ####################################
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





    def matrix_index(X,n,m):
        return X[n-1,m-1]







    #def multiples_three(n):
    #    return np.arange(3,n*3,3)
    #print(f"The first 10 multiples of 3 are:\n{multiples_three(10)}")




