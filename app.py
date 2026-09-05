from fastapi import FastAPI
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import os


app = FastAPI()

# Load data paths from environment variables
# FOOD_DB_PATH = "db\food.csv"
# ORDERS_DB_PATH = "db\orders.csv"
# USERS_DB_PATH = "db\users.csv"
# NEW_SPECIALS_DB_PATH = "db\new_and_specials.csv"

FOOD_DB_PATH = r"db\food.csv"
ORDERS_DB_PATH = r"db\orders.csv"
USERS_DB_PATH = r"db\users.csv"
NEW_SPECIALS_DB_PATH = r"db\new_and_specials.csv"


# Load the data
df1 = pd.read_csv(FOOD_DB_PATH)
df1.columns = ['food_id', 'title', 'canteen_id', 'price', 'num_orders', 'category', 'avg_rating', 'num_rating', 'tags']

# Create soup for content-based filtering
def create_soup(x):
    tags = x['tags'].lower().split(', ')
    tags.extend(x['title'].lower().split())
    tags.extend(x['category'].lower().split())
    return " ".join(sorted(set(tags), key=tags.index))

df1['soup'] = df1.apply(create_soup, axis=1)

# CountVectorizer and cosine similarity
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df1['soup'])
cosine_sim = cosine_similarity(count_matrix, count_matrix)

indices_from_title = pd.Series(df1.index, index=df1['title'])
indices_from_food_id = pd.Series(df1.index, index=df1['food_id'])

# Function to get recommendations
def get_recommendations(title="", cosine_sim=cosine_sim, idx=-1):
    if idx == -1 and title != "":
        idx = indices_from_title[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:3]
    food_indices = [i[0] for i in sim_scores]
    return food_indices

# Function to get latest user orders
def get_latest_user_orders(user_id, orders, num_orders=3):
    counter = num_orders
    order_indices = []
    
    for index, row in orders[['user_id']].iterrows():
        if row.user_id == user_id:
            counter = counter - 1
            order_indices.append(index)
        if counter == 0:
            break
            
    return order_indices

# Utility function to get recommendations DataFrame
def get_recomms_df(food_indices, df1, columns, comment):
    row = 0
    df = pd.DataFrame(columns=columns)
    
    for i in food_indices:
        df.loc[row] = df1[['title', 'canteen_id', 'price']].loc[i]
        df.loc[row].comment = comment
        row = row + 1
    return df

# Function to get personalized recommendations
def personalised_recomms(orders, df1, user_id, columns, comment="based on your past orders"):
    order_indices = get_latest_user_orders(user_id, orders)
    food_ids = []
    food_indices = []
    recomm_indices = []
    
    for i in order_indices:
        food_ids.append(orders.loc[i].food_id)
    for i in food_ids:
        food_indices.append(indices_from_food_id[i])
    for i in food_indices:
        recomm_indices.extend(get_recommendations(idx=i))
        
    return get_recomms_df(set(recomm_indices), df1, columns, comment)

# Function to get new and special items
def get_new_and_specials_recomms(new_and_specials, users, df1, canteen_id, columns, comment="new/today's special item in your home canteen"):
    food_indices = []
    
    for index, row in new_and_specials[['canteen_id']].iterrows():
        if row.canteen_id == canteen_id:
            food_indices.append(indices_from_food_id[new_and_specials.loc[index].food_id])
            
    return get_recomms_df(set(food_indices), df1, columns, comment)

# Function to get top-rated items
def get_top_rated_items(df1, columns, comment="top rated items across canteens"):
    # Calculate top-rated items (you can customize this logic)
    top_rated_items = df1.sort_values('avg_rating', ascending=False).head(3)
    food_indices = top_rated_items.index.tolist()
    return get_recomms_df(food_indices, df1, columns, comment)

# Function to get popular items
def get_popular_items(df1, columns, comment="most popular items across canteens"):
    # Calculate popular items (you can customize this logic)
    pop_items = df1.sort_values('num_orders', ascending=False).head(3)
    food_indices = pop_items.index.tolist()
    return get_recomms_df(food_indices, df1, columns, comment)

# Utility function to get user's home canteen
def get_user_home_canteen(users, user_id):
    for index, row in users[['user_id']].iterrows():
        if row.user_id == user_id:
            return users.loc[index].home_canteen
    return -1

# API endpoint to get recommendations
@app.get("/recommendations/{user_id}")
def get_recommendations_for_user(user_id: int):
    orders = pd.read_csv(ORDERS_DB_PATH)
    new_and_specials = pd.read_csv(NEW_SPECIALS_DB_PATH)
    users = pd.read_csv(USERS_DB_PATH)
    
    columns = ['title', 'canteen_id', 'price', 'comment']
    current_canteen = get_user_home_canteen(users, user_id)
    
    personalised = personalised_recomms(orders, df1, user_id, columns)
    new_and_specials_recomms = get_new_and_specials_recomms(new_and_specials, users, df1, current_canteen, columns)
    top_rated = get_top_rated_items(df1, columns)
    popular = get_popular_items(df1, columns)
    
    return {
        "personalised_recommendations": personalised.to_dict(orient='records'),
        "new_and_specials": new_and_specials_recomms.to_dict(orient='records'),
        "top_rated_items": top_rated.to_dict(orient='records'),
        "popular_items": popular.to_dict(orient='records')
    }

# Run the API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)