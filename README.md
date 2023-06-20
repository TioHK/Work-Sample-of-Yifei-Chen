# Work Sample of Yifei Chen

This is a python work sample of Yifei Chen. This read. me file is for illustration purposes about how to run the code and what to expect.

# How to run the code 

## Dependencies 
In order to smoothly run the code, a set of dependencies needed to be installed first. The dependencies are included in the requirements.txt
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

### The dependencies can be installed using a command like this:
```
pip install -r requirements.txt
```

## Files to download 
No matter which way of running the codes you prefer, I recommend you to download the data folder on Git Hub first. And save it to the same directory as you run code using the name 'data'
Note. In the submitted files, considering the data file size limit, I include the files needed to reproduce the result for the data analysis/visualisation part. However, if you want to run the full version of codes, you will still need to download the GitHub data folder.

## First way to run the code: run all of them 
If you would like to run all the codes from data collection to interactive visualisation, you can run the main.py file (this is in the code folder of the zip package I submitted. This way of running takes time. And it can be run by a command like this:
```
python main.py
```
Note. This exact command may change considering how you download code and where you run it 

## Second way of running the code: run separately 
If you only want to see the data collection/cleaning part or only want to see the data analysis/visualization part, you can run the codes in a separate code folders. The codes can be run by using the following command line 
```
python separatecodes/visulizationandanalysis.py
python separatecodes/cleaningandcollection.py
```
Note. This exact command may change considering how you download code and where you run it 

# What you can expect 
1. If you run the full version, some data files (intermediary ones or final ones) will be produced for further use in the analysis part. 
2. For the data visualization and analysis part, you will see (refer to the report in the submitted zip file for detail):
   1. One interactive map 
   2. three static map
   3. three histograms 
   4. seven regression outputs 
   5. two vif test results 
   6. one scatter plot with a fitted line 

























