#==============================================================================#
#   Title: Base Recommender Engine
#   Author: Kui
#   Date: 2020-2-14
#------------------------------------------------------------------------------#
#   Version 1.0
#------------------------------------------------------------------------------#
#   2020-2-14 - Version 1.0
#------------------------------------------------------------------------------#

#==============================================================================#
#   Introduction
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#


#------------------------------------------------------------------------------#
#   SETTINGS
#------------------------------------------------------------------------------#

REGION_MODEL = "data/models/SVC_clf.mdl"
COURSE_MODEL = "data/models/LSTM.mdl"

REGION_PIPE = "data/models/Vectorizer-PCA.mdl"
LE_VECT = "data/models/le_vect.mdl"
LE_REGION = "data/models/le-region.mdl"
LE_CUISINE = "data/models/le-cuisine.mdl"
LE_CONTINENT = "data/models/le-continent.mdl"
LE_COURSE = "data/models/le-courses-2.mdl"

CORPUS_REG = "data/models/corpus-region.mdl"
CORPUS_COR = "data/models/corpus-courses.mdl"

DF = "data/by_id-11_labeled.tmp"


#------------------------------------------------------------------------------#
#   Import packages
#------------------------------------------------------------------------------#
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import joblib as j
from os.path import exists

import re, sys

from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances, cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# from models.ravel import Ravel
from ravel import Ravel
# from models.encode import Encoder

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical




#==============================================================================
# ** Recommender_Base
#------------------------------------------------------------------------------
#   This is the base class of the recommender engine.
#------------------------------------------------------------------------------
class Recommender_Base:
    #-------------------------------------------------------------
    # * Create Variables
    #-------------------------------------------------------------
    region_model        = None
    region_pipe         = None
    course_model        = None
    course_pipe         = None
    corpus_courses      = None
    corpus_region       = None

    le_region           = None
    le_cuisine          = None
    le_continent        = None

    df                  = None
    recommendations     = None
    recommendation_ids  = None


    #-------------------------------------------------------------
    # * Initialize
    #-------------------------------------------------------------
    def __init__(self):
        pd.set_option("max_columns", 999)
        pd.set_option("max_rows", 999)
        self.load_models()

    #-------------------------------------------------------------
    # * Clean Text
    #-------------------------------------------------------------
    def clean_text(self,txt):
        txt = txt.split(",")
        txt = sorted([re.sub("^(\s+)|(\s+)$","",i).lower().replace(" ","_") for i in txt])

        return txt

    #-------------------------------------------------------------
    # * Course Input Vectorizer
    #-------------------------------------------------------------
    def course_input_vectorizer(self,keys):
        get = set(keys).intersection(set(self.le_vect.classes_))
        get = sorted(list(get))
        get=pd.DataFrame({"n":[get]})

        vect = get.n.apply(lambda x: self.le_vect.transform(x))
        #vect = self.le_vect.transform(get)
        # pad sequences
        vect = pad_sequences(vect,130+1)

        return vect

    #-------------------------------------------------------------
    # * Region Input Vectorizer
    #-------------------------------------------------------------
    def region_input_vectorizer(self, keys):
        # text_string = [" ".join(text_arr)]
        # print(text_string)
        get = set(keys).intersection(self.corpus_region)
        get = sorted(list(get))
        get = " ".join(get)
        get = [get]

        # Tf-IDF Vectorizer upto PCA
        vect = self.region_pipe.transform(get)
        return vect

    #-------------------------------------------------------------
    # * Load Models
    #-------------------------------------------------------------
    def load_models(self):
        self.region_model = j.load(REGION_MODEL)
        self.region_pipe = j.load(REGION_PIPE)

        # self.df = pd.DataFrame()
        self.le_vect = j.load(LE_VECT)
        self.course_model = j.load(COURSE_MODEL)
        self.df = j.load(DF)

        # Label Encoders
        self.le_region=j.load(LE_REGION)
        self.le_course=j.load(LE_COURSE)
        
        self.le_continent=j.load(LE_CONTINENT)
        self.le_cuisine=j.load(LE_CUISINE)

        # Load Corpus
        self.corpus_courses = set(j.load(CORPUS_COR).key.tolist())
        self.corpus_region = set(j.load(CORPUS_REG).key.tolist())

        self.sc = MinMaxScaler()


    #-------------------------------------------------------------
    # * Classify
    #-------------------------------------------------------------
    def classify(self, text="", type="region"):
        text_arr = self.clean_text(text)
        if type == "region":
            vect = self.region_input_vectorizer(text_arr)
            clas = self.region_model.predict(vect)
            return clas
        elif type == "course":
            vect = self.course_input_vectorizer(text_arr)
            pred = self.course_model.predict(vect)
            clas = np.argmax(pred,axis=1)[:].ravel()[0]
            return clas
            # for i,v in enumerate(clas):
            #     if v > 0: return i

    #-------------------------------------------------------------
    # * Main
    #-------------------------------------------------------------
    def main(self, text="tomato lettuce cheese", items=10):
        print(text)

        region_label = self.classify(text.replace(" ",","),type="region")
        course_label = self.classify(text.replace(" ",","),type="course")

        region_label = [region_label[0]]
        course_label = [course_label]

        self.region = self.le_region.inverse_transform(region_label)[0]
        self.course = self.le_course.inverse_transform(course_label)[0]

        # Seed
        df = {
        "region_label": region_label,
        "course_label": course_label,
        }
        
        df["sugar_class"] = 1
        df['sodium_class'] = 1
        df['fat_class'] = 1
        df['fiber_class'] = 4
        df["cuisine_label"] = 22 # Filipino

        df = pd.DataFrame(df)

        self.predict(df,items=items)

        #df['protein_class'] = 62
        #df['energy_class'] = 1930
        
        # print(text)
        # print(region,course)
        # print(self.recommendations)


    # Test command
    def xkc__(self, text=None):
        #self.df = j.load(DF)
        self.df = self.df[self.df.energy_class >= 3]
        #df = self.df.head(100)

        x = self.df.bow_str.head(1).apply(lambda x: self.classify(x.replace(" ",","),type="region"))
        print(x)


        # self.df["course_label"] = x
        # x=self.le_course.inverse_transform(x)
        # self.df["course"] = x
        # j.dump(self.df,"by_id-9_labeled.tmp")
        # j.dump(self.df,"data/by_id-9_labeled.tmp")
        # print("Finished")

        #print(df[["title","course"]])
        #region_label = self.classify(text,"region")
        #course_label = self.classify(text,"course")
        #print(region_label, course_label)

    #-------------------------------------------------------------
    # * Predict
    #-------------------------------------------------------------
    def predict(self, df, items=20, filter=True):
        cols = df.columns.tolist()
        df_seed = df[cols].values.reshape(1, -1)
        metric = "cosine_dist"

        get_df = self.df
        if filter == True:
            #get_df = get_df[get_df["cuisine_label"] == df["cuisine_label"].squeeze()]
            #get_df = get_df[get_df["course_label"] == df["course_label"].squeeze()]
            get_df = get_df[get_df["sodium_class"] <= 1]
            get_df = get_df[get_df["sugar_class"] <= 1]
            get_df = get_df[get_df["fat_class"] <= 1]

        # MinMaxScaler
        for col in cols:
            get_df[col] = self.sc.fit_transform(get_df[[col]])

        get_df['cosine_dist'] = get_df.apply(lambda x: 1-cosine_similarity(x[cols].values.reshape(1, -1), \
            df_seed)\
            .flatten()[0], axis=1)

        # Exclude if Seed is recipe
        #recommendation_df = self.df[self.df.recipe_id != df_seed.recipe_id.squeeze()]
        

            
    
        # Rank
        recommendation_df = get_df.sort_values(by=metric).head(items)
        recommendation_df = recommendation_df[['title','recipe_id','region','cuisine','continent']+cols+[metric]]
        self.recommendations = recommendation_df
        self.recommendation_ids = recommendation_df['recipe_id'].values#.tolist()

# from tensorflow.random import set_seed
# set_seed(41)

def main():
    m = Recommender_Base()
    m.main("chicken, soy sauce, garlic, tofu")


# main()
