import openseespy.opensees as OSP
import numpy as np
import sys
import matplotlib.pyplot as plt
import DataStructure as DS

# Redirect stderr to a file
sys.stderr = open('error.log', 'w')

def PyOpenSees_Spectrum(Time_acceleration):
    OSP.wipe()
    # Define material properties
    m = 200
    k = 1.0
    xi = 0.05
    
    # Define 2D model of signal DOF
    OSP.model('basic', '-ndm', 2, '-ndf', 3)
    
    # Define nodes
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
    coord_transf_tag = 1
    crd_transf_type = 'Linear'
    OSP.geomTransf(crd_transf_type, coord_transf_tag)
    
    # Define the elements with section tags
    A = 3225.0
    E = 3600.0
    I = 1080000.0
    section_tag = 1
    OSP.section('Elastic', section_tag, E, A, I)
    OSP.element('elasticBeamColumn', 1, 1, 2, A, E, I, 1)
    OSP.element('elasticBeamColumn', 2, 2, 3, A, E, I, 1)
    OSP.element('elasticBeamColumn', 3, 3, 4, A, E, I, 1)
    OSP.element('elasticBeamColumn', 4, 4, 5, A, E, I, 1)
    
    # Define the loads
    time_list = Time_acceleration.time_list
    value_list = Time_acceleration.value_list
    time_step = Time_acceleration.step_time
    num_steps = Time_acceleration.nuData
    
    # Define the acceleration-time history
    time_series_tag = 1
    OSP.timeSeries('Path', time_series_tag, '-dt', time_step, '-values', *value_list)
    OSP.pattern('UniformExcitation', 1, 1, '-accel', time_series_tag)
    

    # Specify the algorithm, constraint handler, numberer, integrator, and linear SOE
    OSP.constraints("Transformation")
    OSP.numberer("RCM")
    OSP.system("UmfPack")
    OSP.test("NormUnbalance", 0.0001, 10)
    OSP.algorithm("Linear")
    OSP.integrator("LoadControl", 0.0)
    OSP.analysis("Static")
    
    # Perform dynamic analysis and calculate the Spectral Acceleration
    deltaT = time_step
    OSP.analyze(1, deltaT)
    eigs=OSP.eigen('-fullGenLapack', 10)
    print('Eigenvalues: ', eigs)
    # currently we use same damping for each mode
    dmp = [0.05]*len(eigs)
    scalf = [1.0]*len(eigs)
    tsTag = 1
    direction = 1
    #create the response spectrum 
    times=Time_acceleration.time_list
    Spectral_response=[]
    for i in time_list:
    

    
# Test
times = np.linspace(0, 30, 100)
values = np.random.rand(100)
t1 = DS.timeSeries(times, values)
Sa = PyOpenSees_Spectrum(t1)

