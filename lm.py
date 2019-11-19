import numpy as np
import pandas as pd

from family import *

class lm:

    def __init__(self, formula, data=None):
        assert isinstance(data, pd.DataFrame)
        assert isinstance(formula, str)

        '''
        formula is something like 'Y ~ X+Age+BMI'
        '''

        self.formula = "".join(formula.split(" "))
        variables = formula.split("+")

        if data is None:
            # look in global environment
            pass

        try:

            if variables[1] in ["0", "1"]:
                _intercept = bool(int(variables[1]))
                assert all(not var.isdigit() for var in [variables[0]]+variables[2:]) 
            else:
                _intercept = True
                assert all(not var.isdigit() for var in variables)

            self.y = data[variables[0]]

            if variables[1] == '.':
                variables = data.columns
                variables.remove(self.y.columns[0])

            self.x = data[variables]

        except KeyError:
            print("The names in the formula don't match the ones in the DataFrame.")
            raise KeyError

        except AssertionError:
            print("Something is wrong in your formula.")
            raise ValueError
        
        _X = self.x.values
        n, p = _X.shape
        _x = np.hstack((np.ones(shape=[n, 1]), _X))

        self.coefficients  = np.linalg.inv(_X.T@_X)@_X.T@self.y.values
        self.fitted_values = self.coefficients@_x
        self.residuals     = self.y - self.fitted_values
            



class glm(lm):
    def __init__(self, family="gaussian"):
        '''
        attributes inherited include:
        formula, y, x, coefficients, residuals

        for now, canonical link is assumed
        '''
        super().__init__(self)

        assert issubclass(family, _family)

        self.family = family