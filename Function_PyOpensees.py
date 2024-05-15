import openseespy.opensees as OSP
import numpy as np
import sys
#showing library version
import matplotlib.pyplot as plt

# Redirect stderr to a variable
stderr = sys.stderr
sys.stderr = open('error.log', 'w')  # Open a file to write stderr

# Reset stderr
sys.stderr = stderr

# Close the file
sys.stderr.close()
OSP.wipe()



## Define material properties
m=1.0
k=1.0
xi=0.05

#define 2D model of signal DOF
OSP.model('basic', '-ndm', 2, '-ndf', 3)

# Define simplest model in one frame
OSP.node(1, 0.0, 0.0)
OSP.fix(1, 1, 1, 1)  # Fixed in both x and y directions
OSP.node(2, 0.0, 5.0)
OSP.node(3, 2.5, 5.0)
OSP.node(4, 5.0, 5.0)
OSP.node(5, 5.0, 0.0)
OSP.fix(5, 1, 1, 0)  # Fixed in x and y directions but free in rotation

# Define mass
OSP.mass(3, m, 0.0, 0.0)

# Define the coordinate transformation
coord_transf_tag = 1  # Specify a unique tag for the coordinate transformation
crd_transf_type = 'Linear'  # Choose the type of coordinate transformation, e.g., Linear
OSP.geomTransf(crd_transf_type, coord_transf_tag)  # Define the coordinate transformation


# Define the elements with section tags
A = 3225.0
E = 3600.0
I = 1080000.0
section_tag = 1  # Define a section tag
OSP.section('Elastic', section_tag, E, A, I)
OSP.element('elasticBeamColumn', 1, 1, 2, A, E, I, 1)
OSP.element('elasticBeamColumn', 2, 2, 3, A, E, I, 1)
OSP.element('elasticBeamColumn', 3, 3, 4, A, E, I, 1)
OSP.element('elasticBeamColumn', 4, 4, 5, A, E, I, 1)





# Define the loads
##Define acceleration-time history random signal
time_list = np.linspace(0,10,1000)
value_list = np.random.rand(1000)

# Define the acceleration-time history
time_series_tag = 1  # Specify a unique tag for the acceleration-time history
OSP.timeSeries('Path', time_series_tag, '-dt', 0.3, '-values', *value_list)
OSP.pattern('UniformExcitation', 1, 1, '-accel', time_series_tag)


# Specify the algorithm, constraint handler, numberer, integrator, and linear SOE
OSP.algorithm('Newton')
OSP.constraints('Plain')
OSP.numberer('RCM')
OSP.integrator('Newmark', 0.5, 0.25)
OSP.system('ProfileSPD')
OSP.analysis('Transient')

# Perform dynamic analysis
num_steps = 100  # Number of analysis steps
deltaT = 0.01  # Time step size
OSP.analyze(num_steps, deltaT)

print("Analysis completed")



#get Sa values from the analysis of time history
Sa=[]
print("Sa values:")
for i in range(1,6):
    Sa.append(OSP.nodeDisp(i,1))
    print(f"Node {i}: {OSP.nodeDisp(i,1)}")
