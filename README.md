# Disaster Response Message Classification Pipelines

### Table of Contents

1. [Libraries](#Libraries)
2. [Project Description](#ProjectDescription)
3. [File Description](#FileDescription)
4. [Analysis](#Analysis)
5. [Results](#Results)
6. [Future Improvements](#FutureImprovements)
7. [Licensing, Authors, and Acknowledgements](#Licensing)
8. [Instructions](#Instructions)


## Libraries <a name="Libraries"></a>
* pandas
* numpy
* sqlalchemy
* matplotlib
* plotly
* NLTK
* NLTK [punkt, wordnet, stopwords]
* sklearn
* joblib
* flask

## Project Description <a name="ProjectDescription"></a>
Figure Eight Data Set:  [Disaster Response Messages](https://www.figure-eight.com/dataset/combined-disaster-response-data/) provides thousands of messages that have been sorted into 36 categories. These messages are sorted into specific categories such as Water, Hospitals, Aid-Related, that are specifically aimed at helping emergency personnel in their aid efforts.

The main goal of this project is to build an app that can help emergency workers analyze incoming messages and sort them into specific categories to speed up aid and contribute to more efficient distribution of people and other resources. 

## File Description <a name="FileDescription"></a>
There are three main folders:

1. data
    - disaster_categories.csv: dataset including all the categories
    - disaster_messages.csv: dataset including all the messages
    - process_data.py: ETL pipeline scripts to read, clean, and save data into a database
    - DisasterResponse.db: output of the ETL pipeline, i.e. SQLite database containing messages and categories data
2. models
    - train_classifier.py: machine learning pipeline scripts to train and export a classifier
    - classifier.pkl: output of the machine learning pipeline, i.e. a trained classifier
3. app
    - run.py: Flask file to run the web application
    - templates contains html file for the web application
 
## Analysis <a name="Analysis"></a>
*Data Preparation*
- Modify the Category csv; split each category into a separate column
- Merge Data from the two csv files (messages.csv & categories.csv)
- remove duplicates and any non-categorized valued
- create SQL database DisasterResponse.db for the merged data sets

*Text Preprocessing*
- Tokenize text
- remove special characters
- lemmatize text
- remove stop words

*Build Machine Learning Pipeline*
- Build Pipeline with countevectorizer and tfidtransformer
- Seal pipeline with multioutput classifier with random forest 
- Train Pipeline (with Train/Test Split)
- Print classification reports and accuracy scores

*Improve Model*
- Preform GirdSearchCV
- Find best parameters

*Export Model as .pkl File*
- You could also use Joblib as it can be faster. read more [here](https://stackoverflow.com/questions/12615525/what-are-the-different-use-cases-of-joblib-versus-pickle)

## Results <a name="Results"></a>
1. Created an ETL pipeline to read data from two csv files, clean data, and save data into a SQLite database.
2. Created a machine learning pipeline to train a multi-output classifier on the various categories in the dataset.
3. Created a Flask app to show data visualization and classify any message that users would enter on the web page.

## Future Improvements <a name="FutureImprovements"></a>
Much of this data was imbalanced, some labels had only a handful inputs. Here are some approaches to improve our model in the future. read more [here](https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/)
- Change the performance metrics (focus more on Recall and F1-score: weighted average of precision and recall)
- Generate Synthetic Data
- Use Different Algorithms such as multilabel algorithms that take into account that labels may be connected and may not be mutually exclusive.
- Use Penalized Classification Algorithms

## Licensing, Authors, and Acknowledgements <a name="Licensing"></a>
Thanks to Udacity for the starter code and FigureEight for providing the data set to be used by this project.


## Instructions <a name="Instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves model
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

![alt text](https://github.com/Katba-Caroline/Disaster_Response_Message_Classification_Pipelines/blob/master/images/Screenshot%20(98).png)
![alt text](https://github.com/Katba-Caroline/Disaster_Response_Message_Classification_Pipelines/blob/master/images/Screenshot%20(96).png)
![alt text](https://github.com/Katba-Caroline/Disaster_Response_Message_Classification_Pipelines/blob/master/images/Screenshot%20(97).png)

