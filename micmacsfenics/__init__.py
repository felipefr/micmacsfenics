# import os
# import dolfin as df

# # CRITICAL = 50 , ERROR = 40 , WARNING = 30, INFO = 20, PROGRESS = 16, TRACE = 13, DBG = 10
# df.set_log_level(40)

# from .dd.ddmetric import DDMetric
# from .dd.ddsolver import DDSolver
# from .dd.ddmaterial import DDMaterial
# from .dd.ddfunction import DDFunction
# from .dd.ddspace import DDSpace
# from .dd.ddbilinear import DDBilinear
# from .dd.ddproblem_base import DDProblemBase
# from .dd.ddproblem_generic import DDProblemGeneric as DDProblem
# from .dd.ddproblem_infinitesimalstrain import DDProblemInfinitesimalStrain
# from .dd.ddproblem_poisson import DDProblemPoisson
# from .dd.ddsearch import DDSearch
# from .dd.ddstate import DDState

# # research-oriented development
# from .ddd.generalized_metric import *
# from .ddd.ddproblem_finitestrain import DDProblemFiniteStrain
# from .ddd.ddmaterial_rve import DDMaterial_RVE
# from .ddd.ddsolver_nested import DDSolverNested
# from .ddd.dd_al_update import *
# from .ddd.ddproblem_infinitesimalstrain_omega import DDProblemInfinitesimalStrainOmega
# from .ddd.ddsolver_dropout import DDSolverDropout
# from .ddd.ddsearch_isotropy import DDSearchIsotropy
# from .ddd.ddsearch_nnls import DDSearchNNLS
# from .ddd.ddproblem_poisson_mixed import DDProblemPoissonMixed


from .multiscale_model_fetricks.multiscale_model import multiscaleModel
from .core.micro_constitutive_model_generic import MicroConstitutiveModelGeneric