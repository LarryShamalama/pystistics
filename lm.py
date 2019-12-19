import numpy as np
import pandas as pd

from family import *

class FormulaError(Exception):
    def handle():
        return "Something is wrong with your formula..."

class lm:

    def __init__(self, formula, data=None):
        assert isinstance(data, pd.DataFrame)
        assert isinstance(formula, str)

        '''
        formula is something like 'Y ~ X+Age+BMI'
        '''

        self.formula = formula.replace(" ", "")

        try:
            _Y_var, _X_var = self.formula.split("~")
            variables = _X_var.split("+")
        except ValueError as e:
            raise FormulaError(FormulaError.handle())
    
        if data is None:
            # look in global environment
            pass

        try:

            if variables[0] in ["0", "1"]:
                _intercept = bool(int(variables[1]))
                assert all(not var.isdigit() for var in [_Y_var] + variables[1:]) 
            else:
                _intercept = True
                assert all(not var.isdigit() for var in [_Y_var] + variables)

            self.y = data[_Y_var].values # pd Series (or column)

            if variables[0] == '.':
                variables = data.columns
                variables.remove(_Y_var)

            self.x = data[variables].values

        except KeyError:
            print("The names in the formula don't match the ones in the DataFrame.")
            raise KeyError

        except AssertionError:
            raise FormulaError(FormulaError.handle())
        
        n, p = self.x.shape
        _x = np.hstack((np.ones(shape=[n, 1]), self.x))

        self.coefficients  = np.linalg.inv(_x.T@_x)@_x.T@self.y
        self.fitted_values = _x@self.coefficients
        self.residuals     = self.y - self.fitted_values
            



class glm(lm):
    def __init__(self, formula, data=None, family=gaussian):
        '''
        attributes inherited include:
        formula, y, x, coefficients, residuals

        for now, canonical link is assumed
        '''
        super().__init__(self, formula, data)

        assert issubclass(family, _family)

        self.family = family

        