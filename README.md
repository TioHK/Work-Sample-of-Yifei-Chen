# DSCI 510 - Final Project 

This is the final project of Yifei Chen for DSCI 510. The detailed report has been included in the submitted zip file. Therefore, this read.me file is ony for illustration purpose about how to run the code and what to expect if grader directly visit my github respitory. 

# How to run the code 


## Dependencies 
In order to smootly run the code, a set of dependencies needed to be installed first. The dependencies is included in the requirements.txt
> A list of dependencies is also offered here:
> requests==2.28.1
> geopandas==0.12.1
> pandas==1.5.1
> numpy==1.23.4
> matplotlib==3.6.2
> dash==2.7.0
> dash-core-components==2.0.0
> dash-html-components==2.0.0
> dash-table==5.0.0
> plotly==5.11.0
> geopy==2.3.0
> seaborn==0.12.1
> patsy==0.5.3

### The dependencies can be installed using command like this:
```
pip install -r requirements.txt
```

## Files to download 
No matter which way of running the codes you prefered, I recommend you to download the data folder on github first. And save it to the same directory as you run code using the name 'data'
Note. In the submitted files, considering the data file size limit, I include the files needed to reproduce the result for data analysis/visulisation part. However, it you want to run the full version of codes, you will still need to download the github data folder (see report for more detailed explanation)

## First way to run the code: run all of them 
If you would like to run all the codes from data collection to interactive visulisation, you can run the main.py file (this is in the code folder of the zip panckage I submitted. This way of running takes time. And it can be run by a command like this:
```
python main.py
```
Note. This exact command may change considering how you download code and where you run it 

## Second way of running the code: run separately 
If you only want to see the data collection/cleaning part or only want to see the data analysis/visulisation part, you can run the codes in the separatecodes folder. The codes can be run by using the following command line 
```
python separatecodes/visulizationandanalysis.py
python separatecodes/cleaningandcollection.py
```
Note. This exact command may change considering how you download code and where you run it 

# What you can expect 
1. If you run the full version, some data files (intermidiary ones or final ones) will be produced for further use in the analysis part. 
2. For the data vidualisation and analysis part, you will see (refer to the report in the submitted zip file for detail):
   1. One interactive map 
   2. three static map
   3. three histograms 
   4. seven regression outputs 
   5. two vif test results 
   6. one scatter plot with fitted line 

























