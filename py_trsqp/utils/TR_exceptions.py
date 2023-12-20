class IncorrectConstantsException(Exception):
    """Raised when constants requirement are not met"""
    pass

class TRQPIncompatible(Exception):
    """Raised when TRQP is NOT Compatible"""
    pass

class EndOfAlgorithm(Exception):
    """
    Raised when : 
    
    1. restoration phase is impossible to compute
    """
    pass

class PoisednessIsZeroException(Exception):
    "Raised when poisedness is zero. Usually because of duplicated points"
    pass

class IllPoisedModel(Exception):
    "Raised when SVD does not converge"
    pass

class SolutionFound(Exception):
    pass

class RedundantPoint(Exception):
    pass

class IncorrectInputException(Exception):
    pass

class FailedSimulation(Exception):
    pass