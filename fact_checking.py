import requests
import torch
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSequenceClassification, AutoTokenizer
 

def fact_checking(question, entities_names, entities_links, extracted_answer, question_type):
    entity_keywords_contents = ""
    for entity_link in entities_links:
        wikipedia_content = wikipedia_content_scrapper(entity_link)
        keywords_contents = keyword_contents_extraction(wikipedia_content, entities_names)
        entity_keywords_contents += keywords_contents
 
    if question_type == 0:
        bool_answer = bool_answer_extraction(question, entity_keywords_contents)
        if bool_answer == extracted_answer:
            return "Correct"
        else:
            return "Incorrect"
    else:
        wikipedia_content = wikipedia_content_scrapper(extracted_answer)
        extracted_keywords_contents = keyword_contents_extraction(wikipedia_content, entities_names)
        similarity_score = similarity(extracted_keywords_contents, entity_keywords_contents)
        if similarity_score > 0.3:
            return "Correct"
        else:
            return "Incorrect"
        
def wikipedia_content_scrapper(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        content_div = soup.find('div', {'class': 'mw-body-content'})
        paragraphs = content_div.find_all('p')
        paragraph_texts = ""
 
        for paragraph in paragraphs:
            paragraph_texts += paragraph.get_text() + ' '
        return paragraph_texts
    else:
        print(f"failed request: {response.status_code}")
        return None
 
 
def keyword_contents_extraction(content, entities_names):
    sentences = sent_tokenize(content)
    ent_names_lower = [ent.lower() for ent in entities_names]
    entContent = ""
    for sentence in sentences:
        sentence_lower = sentence.lower() 
        ent_found = True 

        for ent in ent_names_lower:
            if ent in sentence_lower:
                ent_found = True 
                break 
            elif ent not in sentence_lower:
                ent_found = False 
                break 

        if ent_found:
            entContent += sentence + " "

    return entContent
 
 
def bool_answer_extraction(question, content):
    tokenizer = AutoTokenizer.from_pretrained("nfliu/roberta-large_boolq")
    model_bool = AutoModelForSequenceClassification.from_pretrained("nfliu/roberta-large_boolq")
 
    sequence = tokenizer.encode_plus(question, content, return_tensors="pt", max_length=512, truncation=True)['input_ids']
    logits = model_bool(sequence)[0]
    probabilities = torch.softmax(logits, dim=1).detach().cpu().tolist()[0]
    prob_yes = round(probabilities[1], 2)
    prob_no = round(probabilities[0], 2)
  
    if prob_yes >= prob_no:
        return "yes"
    else:
        return "no"
 
 
def encode_text(text):
    sim_model = SentenceTransformer('bert-base-nli-stsb-mean-tokens')
    encoded_text = sim_model.encode(text, convert_to_tensor=True)
    return encoded_text
 
 
def similarity_cal(encodingExtracted, encodingEntity):
    cosine_scores = util.pytorch_cos_sim(encodingExtracted, encodingEntity)
    return cosine_scores.item()
 
 
def similarity(contentExtracted, contentEntity):
    encodingExtracted = encode_text(contentExtracted)
    encodingEntity = encode_text(contentEntity)
    similarity_score = similarity_cal(encodingExtracted, encodingEntity)
    return similarity_score
