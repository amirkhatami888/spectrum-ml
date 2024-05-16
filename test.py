from openseespy.opensees import *
import numpy as np
import DataStructure as DS
# Define a 2D model
model("basic", "-ndm", 2, "-ndf", 3)

# Define your time series for random excitation
times = np.linspace(0, 30, 100)
values = np.random.rand(100)
t1 = DS.timeSeries(times, values)
time_list = t1.time_list
value_list = t1.value_list
time_step = t1.step_time
num_steps = t1.nuData

timeSeries("Path", 1, "-dt", time_step, "-values", *value_list, "-factor", 9.806)
pattern('UniformExcitation', 1, 1, '-accel', 1)


numBay = 1
numFloor =1
bayWidth = 5
bayWidth = 3

E = 29500.0

coordTransf = "Linear"  # Linear, PDelta, Corotational

# Define nodes
node(1, 0.0, 0.0)
node(2, 0.0, 5.0)
node(3, 2.5, 5.0)
node(4, 5.0, 5.0)
node(5, 5.0, 0.0)

# Apply boundary conditions

fix(1, 1, 1, 1)
fix(5, 1, 1,1)
mass(3, 1, 1, 1)

# Define the coordinate transformation
coord_transf_tag = 1
crd_transf_type = 'Linear'
geomTransf(crd_transf_type, coord_transf_tag)
# Define the elements with section tags
A = 3225.0
E = 3600.0
I = 1080000.0
section_tag = 1
section('Elastic', section_tag, E, A, I)
element('elasticBeamColumn', 1, 1, 2, A, E, I, 1)
element('elasticBeamColumn', 2, 2, 3, A, E, I, 1)
element('elasticBeamColumn', 3, 3, 4, A, E, I, 1)
element('elasticBeamColumn', 4, 4, 5, A, E, I, 1)

# Define analysis settings
constraints("Transformation")
numberer("RCM")
system("UmfPack")
test("NormUnbalance", 0.0001, 10)
algorithm("Linear")
integrator("LoadControl", 0.0)
analysis("Static")

# run the eigenvalue analysis with 7 modes
# and obtain the eigenvalues
eigs = eigen("-genBandArpack", 3)

# compute the modal properties
modalProperties("-print", "-file", "ModalReport.txt", "-unorm")

# define a recorder for the (use a higher precision otherwise the results
# won't match with those obtained from eleResponse)
filename = 'ele_1_sec_1.txt'
recorder('Element', '-file', filename, '-closeOnWrite', '-precision', 16, '-ele', 1, 'section', '1', 'force')
# some settings for the response spectrum analysis
tsTag = 1 # use the timeSeries 1 as response spectrum function
direction = 1 # excited DOF = Ux
# currently we use same damping for each mode
dmp = [0.05]*len(eigs)
scalf = [1.0]*len(eigs)
# CQC function
def CQC(mu, lambdas, dmp, scalf):
    u = 0.0
    for i in range(ne):
        u += (mu*lambdas[i]*dmp[i]*scalf[i])/(lambdas[i]**2 - mu**2)
    return u

responseSpectrum(tsTag, direction)
#print responseSpectrum(tsTag, direction)
print (CQC(0.05, eigs, dmp, scalf))
# done
wipe()


