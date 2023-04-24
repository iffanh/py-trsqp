import julia
import numpy as np
jl = julia.Julia(compiled_modules=False)  

class SimulationManager():
    def __init__(self, cf, eqcs, ineqcs) -> None:
        #Initialize database
        self.db = DataBase() 
    
        self.eqcs = EqualityConstraints(eqcs=eqcs, db=self.db); 
        self.ineqcs = InequalityConstraints(ineqcs=ineqcs, db=self.db)
        self.cf = CostFunction(func=cf, db=self.db)
        
        pass
    
    def run(self, x) -> None:
        """This method runs the simulation
        """
        return
    

class DataBase():
    def __init__(self) -> None:
        """ Serves as an interface between the TR algorithm and the simulations that are run by Jutul"""
        self.database = {}
        self.database['points'] = []
        self.database['results'] = dict()
        self.database['results']['cf'] = []
        
        pass
    
    def run_simulation(self, Y):
        return
    
    def run(self, new_point:np.ndarray, tol:float=1E-5):
        
        if len(self.database['points']) == 0:
            # run new simulation
            is_run = False

            # add the point
            self.database['points'].append(new_point)
            i = 0
        
        else:
            number_of_simulated_points = len(self.database['points'])
            for i, p in enumerate(self.database['points']):
                if np.linalg.norm(p - new_point) < tol:
                    # get from the table
                    is_run = True
                    
            if i == number_of_simulated_points-1:    
                # run new simulation
                is_run = False

                # add the point
                self.database['points'].append(new_point)
                i = i+1
                
        # Ideally it should return the necessary value: cf, or something else        
        
        return is_run, i
    
class CostFunction():
    def __init__(self, func:callable, db:DataBase) -> None:
        self.number_of_function_calls = 0
        self._func = func
        self.db = db
        pass
    
    def func(self, Y):
        # Counter for number of function call
        self.number_of_function_calls += 1
        
        # We will later use the value from this function call
        is_run, i = self.db.run(new_point=Y)
        return self._func(Y)
    
    def __str__(self) -> str:
        return f"CostFunction(func = {self.func})"

class EqualityConstraints():
    def __init__(self, eqcs, db:DataBase) -> None:
        self.eqcs = [EqualityConstraint(eq, ind) for ind, eq in enumerate(eqcs)]
        self.n_eqcs = len(self.eqcs)

    def __str__(self) -> str:
        return f"EqualityConstraints(n = {self.n_eqcs})"

class EqualityConstraint():
    def __init__(self, func:callable, index:int) -> None:
        self.func = func
        self.index = index

    def __str__(self) -> str:
        return f"EqualityConstraint(index={self.index}, {self.func})"

class InequalityConstraints():
    def __init__(self, ineqcs, db:DataBase) -> None:
        self.ineqcs = [InequalityConstraint(ineq, ind) for ind, ineq in enumerate(ineqcs)]
        self.n_ineqcs = len(self.ineqcs)

    def __str__(self) -> str:
        return f"InequalityConstraints(n = {self.n_ineqcs})"

class InequalityConstraint():
    def __init__(self, func:callable, index:int) -> None:
        self.func = func
        self.index = index

    def __str__(self) -> str:
        return f"InequalityConstraint(index={self.index}, {self.func})"
    
