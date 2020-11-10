from scipy.optimize import linprog
objective_function=[-1,-2]
bounds=[(0,float("inf")),(0,float("inf"))] #bounds on x,y
A_condition=[[2,1],[-4,5],[1,-2]]
B_condition=[20,10,2]
opt=linprog(c=objective_function,
            A_ub=A_condition,b_ub=B_condition,
            bounds=bounds,method="revised simplex")
print(opt.x)
