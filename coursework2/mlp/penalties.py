# -*- coding: utf-8 -*-
"""Parameter penalties.

This module defines parameter penalty functions which can be used to regularise
training by adding an additional term to the objective function being
minimised which aims to restrict 'model complexity' by some measure.
"""

import numpy as np


class L1Penalty(object):
    """L1 parameter penalty.

    Term to add to the objective function penalising parameters
    based on their L1 norm.
    """

    def __init__(self, coefficient):
        """Create a new L1 penalty object.

        Args:
            coefficient: Positive constant to scale penalty term by.
        """
        assert coefficient > 0., 'Penalty coefficient must be positive.'
        self.coefficient = coefficient

    def __call__(self, parameter):
        """Calculate L1 penalty value for a parameter.

        Args:
            parameter: Array corresponding to a model parameter.

        Returns:
            Value of penalty term.
        """
        return self.coefficient * abs(parameter).sum()

    def grad(self, parameter):
        """Calculate the penalty gradient with respect to the parameter.

        Args:
            parameter: Array corresponding to a model parameter.

        Returns:
            Value of penalty gradient with respect to parameter. This
            should be an array of the same shape as the parameter.
        """
        return self.coefficient * np.sign(parameter)

    def __repr__(self):
        return 'L1Penalty({0})'.format(self.coefficient)


class L2Penalty(object):
    """L1 parameter penalty.

    Term to add to the objective function penalising parameters
    based on their L2 norm.
    """

    def __init__(self, coefficient):
        """Create a new L2 penalty object.

        Args:
            coefficient: Positive constant to scale penalty term by.
        """
        assert coefficient > 0., 'Penalty coefficient must be positive.'
        self.coefficient = coefficient

    def __call__(self, parameter):
        """Calculate L2 penalty value for a parameter.

        Args:
            parameter: Array corresponding to a model parameter.

        Returns:
            Value of penalty term.
        """
        return 0.5 * self.coefficient * (parameter**2).sum()

    def grad(self, parameter):
        """Calculate the penalty gradient with respect to the parameter.

        Args:
            parameter: Array corresponding to a model parameter.

        Returns:
            Value of penalty gradient with respect to parameter. This
            should be an array of the same shape as the parameter.
        """
        return self.coefficient * parameter

    def __repr__(self):
        return 'L2Penalty({0})'.format(self.coefficient)
    
class L1L2Penalty(object):
    """L1 parameter penalty.
    
    Term to add to the objective function penalising parameters
    based on their L1 and L2 norm.
    """

    def __init__(self, coefficientL1,coefficientL2):
        """Create a new L1L2 penalty object.
        
        Args:
            coefficient: Positive constant to scale penalty term by.
        """
        assert coefficientL1 > 0., 'Penalty coefficient must be positive.'
        self.coefficientL1 = coefficientL1
        self.coefficientL2 = coefficientL2
        
    def __call__(self, parameter):
        """Calculate L1L2 penalty value for a parameter.
        
        Args:
            parameter: Array corresponding to a model parameter.
            
        Returns:
            Value of penalty term.
        """
        return 0.5 * self.coefficientL1 * abs(parameter).sum() + 0.5 * 0.5 * self.coefficientL2 * (parameter**2).sum()
        
    def grad(self, parameter):
        """Calculate the penalty gradient with respect to the parameter.
        
        Args:
            parameter: Array corresponding to a model parameter.
            
        Returns:
            Value of penalty gradient with respect to parameter. This
            should be an array of the same shape as the parameter.
        """
        return 0.5 * self.coefficientL1 * np.sign(parameter) + 0.5 * self.coefficientL2 * parameter
    
    def __repr__(self):
        return 'L1Penalty({0}), L2Penalty({1})'.format(self.coefficientL1, self.coefficientL2)
