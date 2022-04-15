import dolfin as df 
import numpy as np

from ddfenics.fenics.fenicsUtils import symgrad_voigt, symgrad
import ddfenics.fenics.fenicsUtils as feut
from ddfenics.fenics.enriched_mesh import EnrichedMesh 
# import ddfenics.core.fenics_tools.misc as feut
import ddfenics.mechanics.misc as mech

from functools import partial 

df.parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True, \
                "eliminate_zeros": True, \
                "precompute_basis_const": True, \
                "precompute_ip_const": True}

def solve_cook(meshfile, psi):

    mesh = EnrichedMesh(meshfile)

    Uh = df.VectorFunctionSpace(mesh, "CG", 1)
    clampedBndFlag = 2 
    LoadBndFlag = 1 
    
    ty = 5.0
    traction = df.Constant((0.0,ty ))
    bcL = df.DirichletBC(Uh, df.Constant((0.0,0.0)), mesh.boundaries, clampedBndFlag)
        
    # Chom = Chom_multiscale(tangent_dataset, mapping, degree = 0)

    # # Define variational problem
    duh = df.TrialFunction(Uh)            # Incremental displacement
    vh  = df.TestFunction(Uh)             # Test function
    uh  = df.Function(Uh)                 # Displacement from previous iteration


    Pi = psi(uh)*mesh.dx - df.inner(traction,uh)*mesh.ds(LoadBndFlag)
     
    F = df.derivative(Pi, uh, vh)
    J = df.derivative(F, uh, duh) # it will be computed even not providing it
    
    # Compute solution
    df.solve(F==0, uh, bcL, J = J)
    
    
    return uh


def getPsi(u, param):
    lamb, mu, alpha = param
    
    e = symgrad(u)
    tr_e = df.tr(e)
    e2 = df.inner(e,e)
    
    return (0.5*lamb*(tr_e**2 + 0.5*alpha*tr_e**4) +
           mu*(e2 + 0.5*alpha*e2**2))


def getSig(u, param): # it should be defined like that to be able to compute stresses
    lamb, mu, alpha = param
    
    e = symgrad(u)
    e = df.variable(e)
    
    tr_e = df.tr(e)
    e2 = df.inner(e,e)
    
    psi = 0.5*lamb*(tr_e**2 + 0.5*alpha*tr_e**4) + mu*(e2 + 0.5*alpha*e2**2)
     
    sig = df.diff(psi,e)
    
    return sig
    
if __name__ == '__main__':
    
    meshfile = './meshes/mesh_40.xdmf'
    
    metric = {'YOUNG_MODULUS': 100.0,
               'POISSON_RATIO': 0.3,
               'ALPHA': 200.0}
    
    lamb, mu = mech.youngPoisson2lame(metric['POISSON_RATIO'], metric['YOUNG_MODULUS']) 
    
    lamb = df.Constant(lamb)
    mu = df.Constant(mu)
    alpha = df.Constant(metric['ALPHA'])
    
    psi_law = partial(getPsi, param = [lamb, mu, alpha])

    uh  = solve_cook(meshfile, psi_law)
    
    
    file_results = df.XDMFFile("cook_standard.xdmf")
    file_results.parameters["flush_output"] = True
    file_results.parameters["functions_share_mesh"] = True
    file_results.write(uh, 0.0)

