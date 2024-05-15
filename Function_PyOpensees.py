import openseespy.opensees as OSP
import numpy as np
import sys
import matplotlib.pyplot as plt
import DataStructure as DS

def PyOpenSees_Spectrum(Time_acceleration):
    OSP.wipe()
    # Define material properties
    m = 1.0
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
    OSP.algorithm('Newton')
    OSP.constraints('Plain')
    OSP.numberer('RCM')
    OSP.integrator('Newmark', 0.5, 0.25)
    OSP.system('ProfileSPD')
    OSP.analysis('Transient')
    
    # Perform dynamic analysis and calculate the Spectral Acceleration
    deltaT = time_step
    Spectral_Acceleration=[]
    fo
    
    
    # for i in time_list:
    #     OSP.loadConst('-time', i)
    #     OSP.analyze(1, deltaT)
    #     Spectral_Acceleration.a
    # print("Sa values are calculated")
    # return Sa

# Test
times = np.linspace(0, 30, 100)
values = np.random.rand(100)
t1 = DS.timeSeries(times, values)
t1.plot()
Sa = PyOpenSees_Spectrum(t1)
print(Sa)
# plot the results
plt.plot(times, Sa)
plt.show()
