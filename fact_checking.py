import requests
import torch
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def fact_checking(question: str, entities_names: list, entities_links: list, extracted_answer) -> str:
    """Compare extracted answers with entity information
    @param question: input question
    @param entities_names: list of relevant entity names
    @param entities_links: corresponding Wikipedia links
    @returns: validated "Correct" | "Incorrect"
    """
    keywords = entities_names
    all_keywords_contents = ""
    for entity_link in entities_links:
        wikipedia_content = wikipedia_content_scrapper(entity_link)
        keywords_contents = keyword_contents_extraction(wikipedia_content, keywords)
        all_keywords_contents += keywords_contents
 
    if extracted_answer == "yes" or extracted_answer == "no":
        bool_answer = bool_answer_extraction(question, all_keywords_contents)
        if bool_answer == extracted_answer:
            return "Correct"
        else:
            return "Incorrect"
 
    else:
        wikipedia_content = wikipedia_content_scrapper(extracted_answer)
        extracted_keywords_contents = keyword_contents_extraction(wikipedia_content, keywords)
        similarity_score = similarity(extracted_keywords_contents, all_keywords_contents)
        if similarity_score > 0.3:
            return "Correct"
        else:
            return "Incorrect"
        
def wikipedia_content_scrapper(url: str) -> str:
    """Scrapes content from wikipedia
    @param url: wiki page link
    @returns parsed body content of the page
    """
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
 
 
def keyword_contents_extraction(content: str, keywords: list) -> str:
    """Extracts sentences including given keywords
    @param content: text to search through
    @param keywords: list of keywords to look for
    @returns: relevant sentences
    """

    sentences = sent_tokenize(content)
    keywords_lower = [keyword.lower() for keyword in keywords]
    selected_sentences = ""
    for sentence in sentences:
        sentence_lower = sentence.lower() 
        all_keywords_found = True 

        for keyword in keywords_lower:
            if keyword in sentence_lower:
                all_keywords_found = True 
                break 
            elif keyword not in sentence_lower:
                all_keywords_found = False 
                break 

        if all_keywords_found:
            selected_sentences += sentence + " "

    return selected_sentences
 
 
def bool_answer_extraction(question: str, content: str) -> str:
    """Using roberta-large model, trained on standard question answering dataset and extracted context, estimates the "yes" or "no" probabilities
    @param question: input question
    @param content: parsed text context
    @returns "yes" | "no" based on calculated prediction 
    """
    tokenizer = AutoTokenizer.from_pretrained("nfliu/roberta-large_boolq")
    model_boolQ = AutoModelForSequenceClassification.from_pretrained("nfliu/roberta-large_boolq")
 
    sequence = tokenizer.encode_plus(question, content, return_tensors="pt", max_length=512, truncation=True)[
        'input_ids']
    logits = model_boolQ(sequence)[0]
    probabilities = torch.softmax(logits, dim=1).detach().cpu().tolist()[0]
    proba_yes = round(probabilities[1], 2)
    proba_no = round(probabilities[0], 2)
  
    if proba_yes > proba_no:
        return "yes"
    else:
        return "no"
 
 
def encode_text(text: str) -> str:
    """Create word embeddings using bert model
    @returns vector representation
    """
    sim_model = SentenceTransformer('bert-base-nli-stsb-mean-tokens')
    encoded_text = sim_model.encode(text, convert_to_tensor=True)
    return encoded_text
 
 
def similarity(text1: str, text2: str) -> float:
    """Calculates cosing similarity between the embeddings
    @returns floating point cosine similarity of the inputs
    """
    embedding1 = encode_text(text1)
    embedding2 = encode_text(text2)
    similarity_score = util.pytorch_cos_sim(embedding1, embedding2).item()
    return similarity_score

 
 