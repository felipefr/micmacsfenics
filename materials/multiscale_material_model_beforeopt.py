#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 00:23:16 2022

@author: felipe
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 20:06:27 2022

@author: felipe
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 15:18:46 2022

@author: felipe
"""

import sys
import dolfin as df
import numpy as np
from micmacsfenics.materials.material_model import materialModel 
from micmacsfenics.core.fenicsUtils import LocalProjector


as_sym_tensor = lambda a: df.as_tensor( [ [ a[0], a[1], a[2]] , [a[1] , a[3], a[4]] , [a[2] , a[4], a[5]] ])
ind_sym_tensor = np.array([0, 1, 2, 4, 5, 8])

collect_stress = lambda m, e: np.array( [ m[i].getStress(e[i,:]) for i in range(len(m))] ).flatten()
collect_tangent = lambda m, e: np.array( [ m[i].getTangent(e[i,:]).flatten()[ind_sym_tensor] for i in range(len(m))] ).flatten()

class multiscaleMaterialModel(materialModel):
    
    def __init__(self, micromodels):
        
        self.micromodels = micromodels

    def createInternalVariables(self, W, Wten, dxm):
        self.sig = df.Function(W)
        self.eps = df.Function(W)
        self.tan = df.Function(Wten)
        
        self.eps.vector().set_local(np.zeros(W.dim()))
        
        self.num_cells = W.mesh().num_cells()
        
        self.projector_eps = LocalProjector(W, dxm)
        
        self.size_tan = Wten.num_sub_spaces()
        self.size_strain = W.num_sub_spaces()
        
            
            
    def tangent(self, de):
        return df.dot(as_sym_tensor(self.tan), de) 

    def update_alpha(self, epsnew):
        
        self.projector_eps(epsnew ,  self.eps) 
        
        for m in self.micromodels:
            m.setUpdateFlag(False)
    
        strains = self.eps.vector().get_local()[:].reshape( (self.num_cells, self.size_strain) )
        
        self.sig.vector().set_local( collect_stress(self.micromodels, strains) ) 
        self.tan.vector().set_local( collect_tangent(self.micromodels, strains) ) 
        