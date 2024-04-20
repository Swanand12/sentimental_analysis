from flask import Flask, jsonify,request
from flask_cors import CORS
from ndtv import scrape_ndtv_data
from et import scrape_et_data
from toi import scrape_toi_data
import threading
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import csv


app = Flask(__name__)
CORS(app)


@app.route('/search',methods=['POST'])
def search():
    global news_title
    global results
    search_query = request.json.get('query')
    print(search_query)

    results = {}

    def scrape_and_store(scrape_function, key):
        results[key] = scrape_function(search_query)

    ndtv_process = threading.Thread(target=scrape_and_store, args=(scrape_ndtv_data, 'ndtv'))
    toi_process = threading.Thread(target=scrape_and_store, args=(scrape_toi_data, 'toi'))
    et_process = threading.Thread(target=scrape_and_store, args=(scrape_et_data, 'et'))
    
    ndtv_process.start()
    toi_process.start()
    et_process.start()

    ndtv_process.join()
    toi_process.join()
    et_process.join()

    news_title = []

    tokenizer_en_to_lang = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
    model_en_to_lang = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-es")

    tokenizer_lang_to_en = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-es-en")
    model_lang_to_en = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-es-en")

    with open('headlines_backtranslated.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Original Headline', 'Back-translated Headline'])
    
        for headlines in results.values():
            for headline in headlines:
                
                input_ids = tokenizer_en_to_lang.encode(headline, return_tensors="pt")
                translated_output = model_en_to_lang.generate(input_ids)
                translated_text = tokenizer_en_to_lang.decode(translated_output[0], skip_special_tokens=True)

                input_ids_back = tokenizer_lang_to_en.encode(translated_text, return_tensors="pt")
                back_translated_output = model_lang_to_en.generate(input_ids_back)
                back_translated_text = tokenizer_lang_to_en.decode(back_translated_output[0], skip_special_tokens=True)

                print(back_translated_text)      
                news_title.append(headline)      
                news_title.append(back_translated_text)
            
                writer.writerow([headline, back_translated_text])
        
    return jsonify({'news_title':news_title})


@app.route('/api/data')
def get_data():

    return jsonify({'results':results})



if __name__ == '__main__':
    app.run(debug=True)