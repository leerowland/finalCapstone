import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
nlp = spacy.load('en_core_web_sm')
import pandas as pd
nlp.add_pipe('spacytextblob')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Defining function for sentiment analysis on inputted text, as requested in brief
def sentiment_an(review):
    return (nlp(review))._.blob.sentiment

# Pulling our data from the file 
# Dropping rows with no data in the review column
# Creating a sample data variable so we can choose what rows are processed
dataframe = pd.read_csv('1429_1.csv')
clean_data = dataframe.dropna(subset='reviews.text')
sample_data = clean_data[0:20]

# Defining lists for our data to be cleaned and placed into below
# List containing full review for user review later
with_stop_list = []
# List without stop words for processing
no_stop_list = []

'''
The below two loops could have been merged into one, but 
I've decided to seperate them into data cleaning and language 
processing for ease of understanding from a coding perspective
'''

# Looping through reviews, cleaning data and placing into above lists
# Using a nest for loop and if statement to confirm if words are stop words
for review in sample_data['reviews.text']:
    sentence = nlp(str(review))
    with_stop_list.append(sentence)
    output_sentence = ""
    for word in sentence:
        if word.is_stop == False:
            output_sentence += (str(word) + " ")
    no_stop_list.append(output_sentence.lower().strip())

# Creating lists to store our sentiment values 
polarity_list = []
subjectivity_list = []

# Iterating through no_stop_list, extracting sentiment values and printing results
counter = 0
for review in no_stop_list:
    nlp_review = nlp(review)
    polarity_list.append(nlp_review._.blob.polarity)
    subjectivity_list.append(nlp_review._.blob.subjectivity)
    sentiment = sentiment_an(review)
    assessment = nlp_review._.blob.sentiment_assessments.assessments
    print(f"\nReview {str(counter+1)}\n")
    print(f"{with_stop_list[counter]},\n")
    print(f"{sentiment} \n")
    print(f"Words processed:\n{assessment}")   
    print(f"\n------------------------------")
    counter += 1

"""
While reviewing a sample of the above I wondered if there was any relationship between polarity and 
subjectivity. Which is why I've recorded and plotted subjectivy against polarity values and ran the
regression model. This is discussed more in the PDF summary document.
"""

# Applying simple linear regression model to explore the relationship between polarity and subjectivity
x = subjectivity_list 
y = polarity_list
X = pd.DataFrame(subjectivity_list)
y = polarity_list
X = X.values.reshape(-1, 1)
reg_model = LinearRegression()
reg_model.fit(X, y)
y_pred = reg_model.predict(X)

# Plotting polarity/subjectivity and regression model
plt.scatter(x, y, color="b")
plt.plot(x, y_pred, color="r")
plt.xlabel("Subjectivity (Objective to Subjective)")
plt.ylabel("Sentiment of Review")
#plt.show