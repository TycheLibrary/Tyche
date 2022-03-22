from language import *

if __name__ == "__main__":
    x = atom("x")
    print(x)
    y = atom("y")
    z = atom('z')
    cond = conditional(x,y,z)
    print(cond)
    role = Role('r')
    marginal = marginal(role,x,y)
    print(marginal)
    print(yes.operator)
