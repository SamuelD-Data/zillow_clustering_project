### Title 
The Drivers of Errors in Single Unit Zestimates at Zillow

### Background

From Zillow:

"Zillow is the leading real estate and rental marketplace dedicated to empowering consumers with data, inspiration and knowledge around the place they call home, and connecting them with the best local professionals who can help.

Zillow serves the full lifecycle of owning and living in a home: buying, selling, renting, financing, remodeling and more. It starts with Zillow's living database of more than 110 million U.S. homes - including homes for sale, homes for rent and homes not currently on the market, as well as Zestimate home values, Rent Zestimates and other home-related information. Zillow operates the most popular suite of mobile real estate apps, with more than two dozen apps across all major platforms."

### Goals

- Identify the drivers of log error values in Zillow property value estimates ("Zestimates") to fluctuate. 

- We will also create a machine learning model that will predict the resulting logerror values of our zestimates.

I will also deliver the following:

- zillow_clustering_project.ipynb
    - A Jupyter Notebook that includes an introduction, agenda, project planning steps, a breakdown of our project at each phase of the data science pipeline, and a conclusion.

- README.md
    - A markdown file that includes a background summary of Zillow, project goals, data dictionary, reason for selected columns, initial thoughts, initial hypothesis, project plan, steps for how to reproduce the project, and key findings / takeaways.

- acquire.py
    - A python file that contains all functions needed to acquire the data 
        - Note that user will need credentials to access database source 

- prep.py
    - A python file containing all of the functions needed to prepare the data for exploration

- model.py
    - A python file that contains all of the functions needed to create the models used in the modeling phase

- A presentation that walks through each step of our project and the notebook as a whole.


### Data Dictionary

Defining all columns that were used in exploration and beyond in addition to heating_system columns since they were a major part of preparation.

bathroom_cnt: Number of bathrooms in property (renamed to bathroom_count)

bathroom_count: Number of bathrooms in property

bedroom_cnt: Number of bedrooms in property

bedroom_count: Number of bedrooms in property (renamed to bedroom_count)

calculatedfinishedsquarefeet: Total living square feet within property (renamed to property_sq_ft)

property_sq_ft: Total living square feet within property

taxdvalueollarcount: Total tax value of property (renamed to tax_dollar_value)

tax_dollar_value: Total tax value of property

heatingorsystemtypeid: Code for type of heating system in property (encoded and split into heating_system_type_x)

heating_system_type_2: Central heating system in property

heating_system_type_20: Floor/Wall heating system in property

heating_system_type_7: Solar heating system in property

### Reasons for Selected Columns

bathroom_count: Found this column was a near duplicate of two other columns (fullbathcnt and calculatedbathnbr). Only 46 rows differed between them so I didn't perceive any significant impact of their differences. Chose this one as the name sounded the closest to what I needed, a count of the bathrooms in the property. 

property_sq_ft: Found this column was a near duplicate of finishedsquarefeet12. Only 48 rows differed between them so I didn't perceive any significant impact of their differences. Decided to use this column since the name sounded the closest to what I wanted, the square footage within the property.

tax_dollar_value: Represents the sum of landtaxvaluedollarcnt and structuretaxvaluedollarcnt. I intuitively felt the sum of the tax value from both of the originating values would be more effective in my exploration and modeling. If this feature was found to be ineffective I would have considered using its source values instead.

All other columns have unique values that were not represented directly or indirectly in other columns. Thus they were chosen as they were the only sources for their data. 

### Initial Thoughts

- How will I handle exploring clusters?
    - Use visualizations to see cluster relationship with logerror, the perform hypothesis test to evaluate observation

- How can I perform a hypothesis test on a cluster variable with 3 or cluster types?
    - Use ANOVA test since it allows for more than 2 variable means to be tested simeltaneously

- How will I know how many clusters to make for each feature set?
    - Use subplots or elbow test to identify viable cluster amount


### Initial Hypothesis

- Log errors will push farther away from 0 in cases where the rarity of variables value increases.
    - For example, if only a few properties we've ever evaluated have more than 20 bathrooms, we're going to have trouble evaluating it's value accurately because we haven't encountered a lot of properties with that rare of a variable that relates to value.
        - Update: After exploring and modeling, this appears to be untrue, it seemed that as bathroom count and other variables decreased, log errors became more numerous and extreme, despite there being an abundance of properties with low bathroom counts.

### Project Plan

1) Acquire
- Use function with SQL query to acquire data from data science database

2) Prepare
- Prepare data as needed for exploration including but not limited to
    - Addressing null values
        - Impute if 
            - reasonable given turnaround timeframe
            - too much data will be lost by dropping
        - Drop otherwise
    - Addressing outliers
        - focus on extreme outliers (k=6) to preserve data
        - drop to conserve time
        - otherwise transform to upper/lower boundaries if too much data will be lost by dropping
    - Data types
        - make sure all columns have an appropriate datatype
    - Dropping columns
        - remove columns as needed and for reasons such as
            - being duplicate of another column
            - majority of column values are missing
            - only 1 unique value in data
    - Scale non-target variable numerical columns
    - Encode categorical columns via get_dummies
    - Use RFE on columns remaining after prep
        - For simplicity, only take top 3 columns to exploration
            - Can retrieve more if deemed necessary later

3) Explore
- Plot each feature relation to logerror 
    - Identify the relationship between them (example, as x increases, log_error increases past 0)
    - Perform hypothesis test to confirm or deny if this relationship statistically present
- Create clusters using each unique pair of features
    - Use subplots with varying number of clusters per pair of features to identify a cluster amount that produces strong separation between clusters
    - Create clusters with amount prescribed by subplots
    - Perform hypothesis on each set of clusters to see if log error varies between them
- Take all non-clustered and clustered variables that were statistically significant to modeling phase to use as features in model

4) Model
- Create baseline that predicts mean of logerror and calculate RMSE against actual logerror values in train data
- Create 3 alternate models with varying features
    - use 3 models on train set
    - top 2 models that outperform baseline will go to validate
- Use top 2 models on validate set, model with best RMSE goes to test
- Use best model on test set and evaluate results

5) Conclude
- Document the following
    - identifed drivers of log error
    - best model's details
    - evaluate effectiveness of clusters as drivers and model features
    - recommendations
    - expectations
    - what I would like to do in the future with regard to this project


### How to Reproduce

Download data from Kaggle into your working directory. (You must be logged in to your Kaggle account.)

Install acquire.py, prepare.py and model.py into your working directory.

Run the jupyter notebook.

### Key Findings and Takeaways

Install acquire.py, prepare.py and model.py into your working directory. (You must have access to Codeup data science database)

Run the jupyter notebook.

# Summary of Findings
Through visualizations, hypothesis tests and modeling, we discovered evidence that drivers of log_error may include 
- bedroom_count
- property_sq_ft
- tax_dollar_value
- clusters created from a combination of bedroom_count and property_sq_ft

We created several models including a baseline that always predicted logerror to be the sample average

- Each model's performance was evaluated based on the RMSE value produced by comparing its prediction of logerror values vs. actual log error values from the data it was predicting with

- Model 2 was the best performer (specs listed below)

    - Type: Linear Regression
    - Features: Uses all features listed above, except for clusters
    
- Although this model did not use clusters as features and outperformed models that did, our clustering algorithm was very new and with time could be improved and incorporated into this model to possibly improve its effectiveness

- It should be noted that our second best model used clusters on the validate (out of sample) data to outperform our baseline model which was using in-sample data. This is further evidence that clusters may still be useful as tool for predicting log errors.
    
# Recommendations
- We should focus on the features identified as drivers of log error when attempting to refine our zillow estimate software. For example, let's explore how we are generating zestimates for properties with lower square footage to identify why this variable relates to higher log errors.


# Expectations
- By focusing our efforts on understanding what drives our logerrors, we can improve the accuracy of our zestimates  which will increase user satisfaction and make our zestimates more viable.

# In the future 
- I'd like to revisit this project and use the heating_system categorical variable as a feature since it was ranked very highly during our RFE phase. We only elected to not use it in order to simplify our exploration and save time. We may be able to find evidence of it as a driver of logerror as well as incorporate it into our model.




