"""
LPEG - Linear Programming and Extended Generalization Library
A collection of machine learning utilities for regression, data generation, and preprocessing.
"""

# Dependency imports

# Import all functions from submodules to make them available at package level
from .lpeg_regressions import *
from .lpeg_data_generators import *
from .lpeg_preprocessing import *
from .lpeg_classificators import *


# Define what gets imported when someone does "from lpeg import *"
__all__ = [
    # From lpeg_regressions
    'linearRegression',
    'ridgeRegression', 
    'squaredLoss',
    'squaredLossGradient',
    'polySquaredLoss',
    'polySquaredLossGradient',
    'gradientDescent',
    'adaGraD',
    'secantMethod',
    'GDSecantMethod',
    
    # From lpeg_data_generators
    'linear_data_generator',
    'polynomial_data_generator',
    'gaussian_clouds_data_generator',
    'uniform_clouds_data_generator',
    'partialBallCreate',
    'dataset_relabler',
    'sine_sign_relabler',
    'cosine_sign_relabler',
    'linear_sign_relabler',
    'exponential_sign_relabler',
    
    # From lpeg_preprocessing
    'rescale',
    'add_bias_term',
    'remove_bias_term',
    
    # From lpeg_classificators
    'sigmoid'
    'logistic_loss'
    ]

__version__ = "1.0.0"
__author__ = "Leo Peinkhofer & Emilio Groppi"