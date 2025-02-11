def total_cost(n,retail_price):
    if n > 10:
        tcost = n * retail_price *(2.0/3.0)
    elif n > 4 and n < 10:
        tcost = n * retail_price * (13.0/15.0)
    else:
        tcost = n * retail_price
    return tcost

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
