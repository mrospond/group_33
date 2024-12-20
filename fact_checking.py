import requests
import torch
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def fact_checking(question: str, entities_names: list, entities_links: list, extracted_answer: str, question_type: int) -> str:
    """Compare extracted answers with entity information
    @param question: input question
    @param entities_names: list of relevant entity names
    @param entities_links: corresponding Wikipedia links
    @param extracted_answer: answer received from llm
    @param question_type: type of question - yes/no or entity
    @returns: validated "Correct" | "Incorrect"
    """
    all_keywords_contents = ""
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
 
 
def keyword_contents_extraction(content: str, entities_names: list) -> str:
    """Extracts sentences including given keywords
    @param content: text to search through
    @param entities_names: list of keywords to look for
    @returns: relevant sentences
    """

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
 
 
def bool_answer_extraction(question: str, content: str) -> str:
    """Using roberta-large model, trained on standard question answering dataset and extracted context, estimates the "yes" or "no" probabilities
    @param question: input question
    @param content: parsed text context
    @returns "yes" | "no" based on calculated prediction 
    """
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
 
 
def encode_text(text: str) -> str:
    """Calculate word embeddings using bert model
    @returns vector representation
    """
    sim_model = SentenceTransformer('bert-base-nli-stsb-mean-tokens')
    encoded_text = sim_model.encode(text, convert_to_tensor=True)
    return encoded_text
 
 
def similarity(contentExtracted: str, encodingEntity: str) -> float:
    """Calculates cosing similarity between the embeddings
    @returns floating point cosine similarity of the inputs
    """
    encodingExtracted = encode_text(contentExtracted)
    encodingEntity = encode_text(encodingEntity)
    similarity_score = util.pytorch_cos_sim(encodingExtracted, encodingEntity).item()
    return similarity_score
