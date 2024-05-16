import openseespy.opensees as ops
import numpy as np
import sys
# import gentic library
import pygad
sys.stderr = open('error.log', 'w')

ops.wipe()
# Define a 2D model
model('basic', '-ndm', 2, '-ndf', 3)
geomTransf('Linear', 1)
Xi = 0.05
g=9.81
matTag = 1
E = 200000.0
nu = 0.3
m=200
ops.uniaxialMaterial('Elastic', matTag, E)
ops.section('Elastic', 1, E, 3225.0, 1080000.0)
ops.node(1, 0.0, 0.0)
ops.node(2, 0.0, 5.0)
ops.node(3, 2.5, 5.0)
ops.node(4, 5.0, 5.0)
ops.node(5, 5.0, 0.0)
ops.fix(1, 1, 1, 1)
ops.fix(5, 1, 1, 1)
ops.mass(3, m, 0.0, 0.0)
ops.element('elasticBeamColumn', 1, 1, 2, 1, 1, 1, 1)
ops.element('elasticBeamColumn', 2, 2, 3, 1, 1, 1, 1)
ops.element('elasticBeamColumn', 3, 3, 4, 1, 1, 1, 1)
ops.element('elasticBeamColumn', 4, 4, 5, 1, 1, 1, 1)

#load time_acceleration
times = np.linspace(0, 30, 100)
values = np.random.rand(100)
## Define your time series for random excitation
stepTime=times[1]-times[0]
timeSeries("Path", 1, "-dt", stepTime, "-values", *values, "-factor", 9.806)
pattern('UniformExcitation', 1, 1, '-accel', 1)
# Define the analysis settings
ops.system('UmfPack')
ops.numberer('Plain')
ops.constraints('Plain')
ops.integrator('LoadControl', 1.0)
ops.algorithm('Linear')
ops.analysis('Static')
ops.analyze(1)
numEigen = 10
eigenValues = ops.eigen(numEigen)
print("Eigenvalues: ", eigenValues)
