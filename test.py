import urllib.request
from bs4 import BeautifulSoup
import re
import csv
import os
import json
import pandas as pd
from sklearn.externals import joblib
from underthesea import word_tokenize
import numpy as np
import transformers as ppb
import torch
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import time

def load_url_selenium (url):
    driver=webdriver.Chrome(executable_path='/usr/bin/chromedriver')
    print("Loading url=", url)
    driver.get(url)
    review_csv=[]
    while True:
        #Get the review details here
        WebDriverWait(driver,10).until(EC.visibility_of_all_elements_located((By.CSS_SELECTOR,"div.item")))
        product_reviews = driver.find_elements_by_css_selector("[class='item']")
        # Get product review
        for product in product_reviews:
            review = product.find_element_by_css_selector("[class='content']").text
            if (review != "" or review.strip()):
                print(review, "\n")
                review_csv.append(review)
            # else:
            #     print(review)
            #     review_csv.append("No comments/review is an image")
        #Check for button next-pagination-item have disable attribute then jump from loop else click on the next button
        if len(driver.find_elements_by_css_selector("button.next-pagination-item.next[disabled]"))>0:
            break;
        else:
            button_next=WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.CSS_SELECTOR, "button.next-pagination-item.next")))
            driver.execute_script("arguments[0].click();", button_next)
            print("next page")
            time.sleep(2)
    driver.close()
    print(review_csv)
    return review_csv

def load_url(url):
    print("Loading url=", url)
    page = urllib.request.urlopen(url)
    soup = BeautifulSoup(page,"html.parser")
    script = soup.find_all("script", attrs={"type": "application/ld+json"})[0]
    script = str(script)
    script = script.replace("</script>","").replace("<script type=\"application/ld+json\">","")
    csvdata = []

    for element in json.loads(script)["review"]:
        if "reviewBody" in element:
            csvdata.append([element["reviewBody"]])

    return csvdata

def standardize_data(row):
    # remove stopword

    # Xóa dấu chấm, phẩy, hỏi ở cuối câu
    row = re.sub(r"[\.,\?]+$-", "", row)
    # Xóa tất cả dấu chấm, phẩy, chấm phẩy, chấm thang, ... trong câu
    row = row.replace(",", " ").replace(".", " ") \
        .replace(";", " ").replace("“", " ") \
        .replace(":", " ").replace("”", " ") \
        .replace('"', " ").replace("'", " ") \
        .replace("!", " ").replace("?", " ") \
        .replace("-", " ").replace("?", " ")

    row = row.strip()
    return row

# Tokenizer
def tokenizer(row):
    return word_tokenize(row, format="text")

def analyze(result):
    bad = np.count_nonzero(result)
    good = len(result) - bad
    print("No of bad and neutral comments = ", bad)
    print("No of good comments = ", good)

    if good>bad:
        return "Good! You can buy it!"
    else:
        return "Bad! Please check it carefully!"

# 1. Load URL and print comments
url = input('Nhập url trang:')
if url== "":
    url = "https://www.lazada.vn/products/quan-boi-nam-hot-trend-i244541570-s313421582.html?spm=a2o4n.searchlist.list.11.515c365foL7kyZ&search=1"

data = load_url_selenium(url)

# 2. Standardize data
data_frame = pd.DataFrame(data)
data_frame[0] = data_frame[0].apply(standardize_data)

# 3. Tokenizer
data_frame[0] = data_frame[0].apply(tokenizer)

# 4. Embedding
X_val = data_frame[0]
print(X_val)

model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)
# joblib.dump(model, 'tfidf.pkl')
tokenized = X_val.apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
max_len = 0
for i in tokenized.values:
    if len(i) > max_len:
        max_len = len(i)

padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])

np.array(padded).shape

attention_mask = np.where(padded != 0, 1, 0)
attention_mask.shape

input_ids = torch.tensor(padded)  
attention_mask = torch.tensor(attention_mask)

with torch.no_grad():
    last_hidden_states = model(input_ids, attention_mask=attention_mask)

features = last_hidden_states[0][:,0,:].numpy()

train_features= features


#_val = X_val
# emb = joblib.load('tfidf.pkl')
# X_val = emb.transform(X_val)

# 5. Predict
model = joblib.load('saved_model.pkl')
result = model.predict(train_features)
print(result)
#print(x_val,' ',result)
print(analyze(result))
#print("Done")



