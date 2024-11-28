from llama_cpp import Llama
import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
import wikipedia
import wikipediaapi
from wikidata.client import Client
import requests
import re

model_path = "/home/user/models/llama-2-7b.Q4_K_M.gguf"

# If you want to use larger models...
#model_path = "models/llama-2-13b.Q4_K_M.gguf"
# All models are available at https://huggingface.co/TheBloke. Make sure you download the ones in the GGUF format

question = "What is the capital of Nicaragua? "
llm = Llama(model_path=model_path, verbose=False)
print("Asking the question \"%s\" to %s (wait, it can take some time...)" % (question, model_path))
output = llm(
      question, # Prompt
      max_tokens=32, # Generate up to 32 tokens
      stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
      echo=True # Echo the prompt back in the output
)
print("Here is the output")
print(output['choices'])

llmAnswer = output['choices'][0]['text']

#------------------------------------------------------------------------
# #NER - NLTK 
# nltk.download('words')
nltk.download('punkt_tab')
# nltk.download('averaged_perceptron_tagger_eng')
# nltk.download('maxent_ne_chunker_tab')

# tokens = word_tokenize(llmAnswer)
# pos_tags = pos_tag(tokens)

# entities = ne_chunk(pos_tags)
# print("\nEntities using NLTK:")
# for entitiy in entities:
#     if hasattr(entitiy, 'label'):
#         print(f"{' '.join(c[0] for c in entitiy)} ({entitiy.label()})")

#------------------------------------------------------------------------

#NER - SpaCy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

text = nlp(llmAnswer)
entities = [(ent.text, ent.label_) for ent in text.ents]
# print("\nEntity Recognition:\n")
# for entity in entities:
#     print(f"{entity[0]} ({entity[1]})")
 
wikidata_client = Client()
  
def entity_disambiguation(entity_name):
    entity_name = entity_name.replace(" ", "_")
    query_url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={entity_name}&language=en&format=json"
    response = requests.get(query_url).json()
 
    if 'search' not in response:
        return []
 
    links = []
    for item in response['search']:
        links.append({
            "id": item["id"],
            "label": item.get("label", ""),
            "description": item.get("description", ""),
            "url": f"https://www.wikidata.org/wiki/{item['id']}"
        })
    return links
 
def disambiguation_scoring(entity, context, links):
    context_tokens = word_tokenize(context.lower())
    entity_tokens = word_tokenize(entity.lower())
    scores = {}
    for link in links:
        label = link["label"].lower()
        description = link["description"].lower()
        score = 0
        for token in entity_tokens:
            if token in label:
                score += 1
        for token in context_tokens:
            if token in description:
                score += 1
        scores[link["url"]] = score
 
    sorted_links = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_links
 
print("\nEntity extracted:\n")
for entity in entities:
    entity_name = entity[0]
    links = entity_disambiguation(entity_name)
 
    if not links:
        print(entity_name+"\tNo links found")
        continue
 
    ranked_links = disambiguation_scoring(entity_name, llmAnswer, links)
    if ranked_links:
        best_link = ranked_links[0][0]
    else:
        best_link = "No Match"    
    match = re.search(r'(?<=\/)([^\/]+)(?=$)', best_link)
    if match:
        try:
            client = Client()
            entity = client.get(match.group(0), load=True)
            url = entity.data['sitelinks']['enwiki']['url']
            print(entity_name+"\t"+url)
        except:
            print(entity_name+"\t"+best_link)
    print("\n")