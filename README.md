# Predicting Log Errors in Single Unit Zestimates at Zillow


### Background

From Zillow:

"Zillow is the leading real estate and rental marketplace dedicated to empowering consumers with data, inspiration and knowledge around the place they call home, and connecting them with the best local professionals who can help.

Zillow serves the full lifecycle of owning and living in a home: buying, selling, renting, financing, remodeling and more. It starts with Zillow's living database of more than 110 million U.S. homes - including homes for sale, homes for rent and homes not currently on the market, as well as Zestimate home values, Rent Zestimates and other home-related information. Zillow operates the most popular suite of mobile real estate apps, with more than two dozen apps across all major platforms."

### Goals

- Improve original estimate of the log error by using clustering methodologies.

- Create a model that predicts log error 

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

bathroom_count: Number of bathrooms in property

bedroom_count: Number of bedrooms in property 

building_quality_type_id: Code for quality of building type, no description given beyond this (encoded and split into buildingqualitytypeid_x)

buildingqualitytypeid_x: Boolean columns representing value from building_quality_type_id

lotsize_sqft: Total square feet in lot property is located on

property_sq_ft: Total living square feet within property

tax_dollar_value: Total tax value of property

heating_system_type_id: Code for type of heating system in property (encoded and split into heatingsystemtype_x)

heating_system_type_x: Boolean columns representing value from heating_system_type_id

latitude: latitude of property

longitude: longitude of property

property_land_use_type_id: Code for type of heating system in property (encoded and split into propertylandusetypeid_x)

propertylandusetypeid_x: Boolean columns representing value from property_land_use_type_id

year_built: Year property was built

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
            - Update: Keeping outliers in data as they appear to be legitimately extreme values and not erroneous.
    - Data types
        - make sure all columns have an appropriate datatype
    - Dropping columns
        - remove columns as needed and for reasons such as
            - being duplicate of another column
            - majority of column values are missing
            - with identical values in each row of column
    - Scale non-target variable numerical columns
    - Encode categorical columns via get_dummies
    - Rename columns as appropriate


3) Explore
- Plot each feature relation to logerror 
    - Identify the relationship between them (example, as x increases, log_error moves away/closer from/to  0)
- Perform hypothesis tests
- Create clusters using pairs of features
    - Use subplots with varying number of clusters per pair of features to identify a cluster amount that produces strong separation between clusters
    - Create clusters with amount prescribed by subplots
    - Perform hypothesis on each set of clusters to see if log error varies between them
- Take all non-clustered and clustered variables that were statistically significant to modeling phase to use as features in model

4) Model
- Create baseline that predicts mean of logerror and calculate RMSE against actual logerror values in train data
- Create 4 alternate models with varying features
- Use 3 models on train set
- Top 2 models that outperform baseline will go to validate
- Use top 2 models on validate set, model with lowest RMSE goes to test
- Use best model on test set and evaluate results

5) Conclude
- Document the following
    - Features that were found to have connection with log error
    - Best model's details
    - Recommendations
    - Expectations
    - What I would like to do in the future with regard to this project


### How to Reproduce

Download data from Codeup's data science database into your working directory. (Must have access to database)

Install acquire.py, prepare.py and model.py into your working directory.

Run the jupyter notebook.

### Key Findings and Takeaways

__Summary of Findings__

- Explored many variables via plots and hypothesis tests and found the following to be viable for predicting log error
    - bathroom_count
    - bedroom_count
    - property_sq_ft
    - tax_dollar_value


- Created clusters from various combinations of bedroom_count, tax_dollar_value and  property_sq_ft and found statistical signifgance between the average log errors between each combination respective set of clusters 


- Created several models including a baseline that always predicted logerror to be the sample average

- Each model's performance was evaluated based on RMSE produced by comparing its prediction of logerror values vs. actual log error values

- Model 2 was the best performer 
    - Linear Regression
    - Features
        - property_sq_ft
        - bathroom count
        - cluster set 2
            - cluster variables: tax_dollar_value and bedroom_count
            - number of clusters: 2


__Recommendation__
- Begin a project to improve the accuracy of our zillow estimate software using the insights and model generated from this project


__Expectations__
- By improving the accuracy of our zestimates we will increase satisfaction among our current users and make our services more attractive to potential users. 


__In the future__
- Create clusters that include bathroom count as this variable is not included in any feature combinations currently
- Cluster latitude and longitude to see how their clusters relate to log error

