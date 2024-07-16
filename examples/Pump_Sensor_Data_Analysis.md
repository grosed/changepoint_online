---
title: "Pump Sensor Data Monitoring"
author: "Edward Austin"
format: 
  gfm
execute:
  warning: false
---

```python
import pandas as pd
#from changepoint_online import nunc
import nunc
import numpy as np
import matplotlib.pyplot as plt
```
##Kaggle Pump Sensor Data

We use a datset containing readings from sensors recording flow on a water pump network. The data is sourced from [kaggle](https://www.kaggle.com/datasets/nphantawee/pump-sensor-data)

This dataset contains labelled faults, and we focus upon the first week of data where a fault was recorded. 

##Data Preparation

We first read in the data and prepare it for analysis:

```python
sensor_df = pd.read_csv("sensor.csv")
#Work with a week of data containing a fault
sensor_df = sensor_df[(sensor_df["timestamp"] >= "2018-04-11") & (sensor_df["timestamp"] <= "2018-04-17")]
#no values in sensor 15 so remove it
sensor_df.drop(["sensor_15","Unnamed: 0"],inplace = True,axis=1)

#Index by time, store status, and then select only sensor readings
sensor_df.index = pd.to_datetime(sensor_df["timestamp"])
machine_status = sensor_df["machine_status"]
#Drop index and timestamp columns
sensor_df.drop(["timestamp", "machine_status"],inplace = True,axis=1)

#Viewing the data gives our 50 sensors and the machine status
sensor_df.head()

#Fill in missing values with last obs carried forward:
sensor_df.fillna(method = "ffill", inplace = True)

#Extract times at which pump has a fault
fault_vals = sensor_df[machine_status == "BROKEN"]
fault_times = fault_vals.index
```

##Data Visualisation
We can plot several examples of the data. The red marker indicates the time a fault is known to have occured on the network.


```python
plt.plot(sensor_df["sensor_01"])
plt.plot(fault_vals["sensor_01"],linestyle='none',marker = "o",color='red',markersize=8)
plt.gcf().autofmt_xdate()
plt.show()

plt.plot(sensor_df["sensor_41"])
plt.plot(fault_vals["sensor_41"],linestyle='none',marker = "o",color='red',markersize=8)
plt.gcf().autofmt_xdate()
plt.show()
```


    
![png](output_5_0.png)
    



    
![png](output_5_1.png)
    


Notice how not all sensors show an obvious change in the data at the time of the fault, however:


```python
plt.plot(sensor_df["sensor_26"])
plt.plot(fault_vals["sensor_26"],linestyle='none',marker = "o",color='red',markersize=8)
plt.gcf().autofmt_xdate()
plt.show()
```


    
![png](output_7_0.png)
    


A key feature of these time series is that they are nonstationary over time; they do not have a clear baseline behaviour. Some of the machine faults, however, appear to coincide with abrupt changes or spikes in the data. This therefore makes this dataset a prime candidate for analysis using NUNC. 

##Change Detection Using NUNC


```python
points_observed = 0  #this is only used for plotting the change with respect to the whole time series.
window = 180
num_quantiles = 5
Y = sensor_df["sensor_01"]
detector = nunc.nunc(window, num_quantiles)

for y in Y:
    points_observed += 1
    detector.update(y)
    if detector.statistic() > 80: #if cost exceeds threshold
        break
    
changepoint_info = detector.changepoint()
#adjust time for point in data stream and store
changepoint = sensor_df.index[(changepoint_info["changepoint"] + points_observed - window)]

#Plot the result
plt.plot(sensor_df["sensor_01"])
plt.plot(fault_vals["sensor_01"],linestyle='none',marker = "o",color='red',markersize=8)
plt.axvline(changepoint, color = "red", linestyle = "--")
plt.gcf().autofmt_xdate()
#This is excellent at finding the change
```


    
![png](output_9_0.png)
    


Although in the previous example we see a very accurate detection of the fault, we also detect some of the changes early. 


```python
points_observed = 0

Y = sensor_df["sensor_02"]
detector = nunc.nunc(window, num_quantiles)

for y in Y:
    points_observed += 1
    detector.update(y)
    if detector.statistic() > 80: #if cost exceeds threshold
        break
    
changepoint_info = detector.changepoint()
#adjust time for point in data stream and store
changepoint = sensor_df.index[(changepoint_info["changepoint"] + points_observed - window)]

#Plot the result
plt.plot(sensor_df["sensor_02"])
plt.plot(fault_vals["sensor_02"],linestyle='none',marker = "o",color='red',markersize=8)
plt.axvline(changepoint, color = "red", linestyle = "--")
plt.gcf().autofmt_xdate()
#We see in this case we have detected the drop early 
```


    
![png](output_11_0.png)
    


To an extent, however, such early detection may not be problematic.
For instance in this example we have provided an early warning of a fault in sensor 02, whilst accurately detecting a fault 
in Sensor 01.


```python

```
