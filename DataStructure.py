import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import tensorflow as tf

class timeSearies:
    def __init__(self,time_list,value_list):
        self.time_list = time_list
        self.value_list = value_list
        if len(time_list) != len(value_list):
            raise ValueError("The length of time_list and value_list should be the same")
        self.nuData = len(time_list)
        self.dataSet =[]
        for i in range(self.nuData):
            self.dataSet.append([self.time_list[i],self.value_list[i]])
        self.step_time = self.time_list[1] - self.time_list[0]
        self.Max_time = self.time_list[-1]
        self.Min_time = self.time_list[0]
        
    def __del__(self):
        del self.time_list
        del self.value_list
        del self.dataSet

    def __len__(self):
        return self.nuData          

    def __getitem__(self,idx):
        return self.dataSet[idx]
    
    def  find_directionPoint(self,idx):
        if ((self.value_list[idx-1] > self.value_list[idx]) and (self.value_list[idx] > self.value_list[idx+1]))\
            or ((self.value_list[idx-1] < self.value_list[idx]) and (self.value_list[idx] < self.value_list[idx+1])):
            return False
        else:
            return True
    
    

    
    
    def numpy(self):
        return np.array(self.dataSet,dtype=np.float32)
    
    def pandas(self):
        return pd.DataFrame(self.dataSet,columns=['time','value'],dtype=np.float32)
    
    def tensorflow(self):
        #tensorflow data format with tf.float32
        return tf.constant(self.dataSet,dtype=tf.float32)   
    
    def plot(self):
        plt.plot(self.time_list,self.value_list)
        plt.show()
    

    def linearPredictor(self,time_input):
        id_bigger = 0
        id_smaller = 0
        for i in range(0,self.nuData):
            if time_input < self.time_list[i]:
                id_bigger = i
                id_smaller = i-1
                break
        if id_bigger == 0:
            id_bigger = self.nuData-1
            id_smaller = self.nuData-2
        x1 = self.time_list[id_smaller]
        x2 = self.time_list[id_bigger]
        y1 = self.value_list[id_smaller]
        y2 = self.value_list[id_bigger]
        slope = (y2-y1)/(x2-x1)
        intercept = y1 - slope*x1
        return slope*time_input + intercept
    
    
    def changeTimestep(self,newTimeStep):
        print(f"old time step: {self.step_time} new time step: {newTimeStep}")
        if newTimeStep > self.step_time:
            raise ValueError("The new time step should be smaller than the old time step")
        if newTimeStep == self.step_time:
            return self
        else:
            new_time_list = np.arange(self.Min_time,self.Max_time,newTimeStep)
            new_value_list = []
            for i in range(len(new_time_list)):
                new_value_list.append(self.linearPredictor(new_time_list[i]))   
            return timeSearies(new_time_list,new_value_list)
            
        
    
        




# Example

##old sries
time_list = np.linspace(0,30,100)
value_list = np.random.rand(100)
t1 = timeSearies(time_list,value_list)
t1.plot()
print(t1.nuData)
t2 = t1.changeTimestep(0.1)
print(t2.nuData)