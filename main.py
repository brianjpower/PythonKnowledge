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

if __name__ == '__main__':
    """ # %% md
    #### Exercise 1
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
    #### Exercise 2 

    Write a `for` loop to calculate the sum of $x^2$ for $x$ from 0 to 9, $\sum_{x=0}^9 x^2$.
"""
    print(f"The sum of x^2 between 0 and 9 is {sum_x_squared(0,9)}")

    """#%% md
     Exercise3 Alter your for loop from Exercise2 to a whileloop,which again calculates ∑
    x=0x2,
    but terminates  instead when    the su m    is  greater then 500."""
    while_sum, index = sum_x_squared_while(500)
    print(f"The sum of x^2 between 0 and {index} is {while_sum}")

