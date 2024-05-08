# Time Series Analysis Gas Demand

The aim of this project it to create an algorithmn that can predict the natural gas demand for a region of the over the course of a year. The UK has publicaly available information from national grid on how much natural gas is being supplied to the system each day.

My initial hypothesis is that this is a highly predictable system seeing as natural gas demand is driven by ambient conditions. Weather data was gathered from the publicly available NOAA (National Oceanic and Atmospheric Administration).

The data required for this include:

1) __Gas Demand Data from National Grid__: Data Representing a whole region 
2) __Weather Data from NOAA__: Data representing a whole region through a grid system. Approx ~ 25 grids per region

You can find the links to this data here: 

https://www.noaa.gov/weather

https://data.nationalgas.com/reports/find-gas-reports

By using both of these data sources, it is possible to create a model that predicts what the gas demand in the UK could be based on weather conditions, time of year etc.

Logically one can see that there is a relationship between weather conditions (air temperature) and natural gas demand for a given region (East Anglia): 

![alt text](./images/image.png)

Through research I found that there is a linear relationship between airTemperature values and the demand for a given day in the year. The figure below shows the minimum air temperature vs. gas demand for the 310th day of the year between 2018-2023:

![alt text](./images/image-1.png)

This is a facinating trend, and I believe it exisits for a few reasons:

- There is an individual in a control room releasing gas into the system
- This release of gas is driven by consumer demand
- The system is very "leaky", has seen it described as being like a "siv"
- This leakiness makes it hard for operators to know exactly how much to release into the system
- Pressure is monitored throughout the day
- However, an intial estimate is calculated based on the airTemperature
- Operators look at the previous years data to get an understanding of the historic demand under certain conditions
- Operators would use a linear regression model to map this relationship

This provided a good baseline for building a model. The final draft of the model two fold: 

1) __Initial Linear Regression__: Do a RANSAC linear regression calculation for the demand vs. airTemperature_min for each grid for a given day in the year. Use this to have a proxy-prediction value for demand.

2) __Machine Learning Model__: Use information from step (1) and all of the weather information to create a time series ML model

The machine learning models that were explored where: 

- XGBoost Random Forest
- Gradient Boosted Machines
- LSTM Neural Network
- GRU Neural Network
- KNN




