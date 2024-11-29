from llama_cpp import Llama
import spacy
import nltk
from nltk.tokenize import word_tokenize
# from nltk.chunk import ne_chunk
# from nltk.tag import pos_tag
# import wikipedia
# import wikipediaapi
from wikidata.client import Client
import requests
import re
import contextlib
import sys
import os

# If you want to use larger models...
model_path = "/home/user/models/llama-2-7b.Q4_K_M.gguf"
# All models are available at https://huggingface.co/TheBloke. Make sure you download the ones in the GGUF format
 
def entity_disambiguation(entity_name: str) -> list:
    """Returns a list of wikidata links matching the input entity
    @param entity_name: entity
    @returns: a list of wikidata links matching given entity
    """
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
 
def disambiguation_scoring(entity: str, context: str, links: list) -> list:
    """Returns a list of wikidata links sorted by score (descending)
    @param entity: entity
    @param context: context
    @param links: list of wikidata links
    @returns: a list of wikidata links sorted by score in descending order
    """
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

def main(id: str, question: str, output_file: str) -> None:
    """Prints (and saves to output.txt) out the disambiguated entity and corresponding Wikipedia link (assures correct output format)
    @param id: question id
    @param question: llm input string
    @param output_file: path to output text file
    @returns: None
    """

    # question = "What is the capital of Nicaragua? "
    # with contextlib.redirect_stdout(None):    
    llm = Llama(model_path=model_path, verbose=False, n_ctx=4096)
    # print("Asking the question \"%s\" to %s (wait, it can take some time...)" % (question, model_path))

    output = llm(
        question, # Prompt
        max_tokens=32, # Generate up to 32 tokens
        # stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
        stop=None,
        echo=True # Echo the prompt back in the output
    )

    # print(output['choices'])
    llmAnswer = output['choices'][0]['text']

    #------------------------------------------------------------------------
    # #NER - NLTK 
    # nltk.download('words')
    # nltk.download('punkt_tab')
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
    with contextlib.redirect_stdout(None):    
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            from spacy.cli import download
            download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")

    text = nlp(llmAnswer)
    entities = set((ent.text, ent.label_) for ent in text.ents)

    # print("\nEntity Recognition:\n")
    # for entity in entities:
    #     print(f"{entity[0]} ({entity[1]})")
    
    # wikidata_client = Client()
    with open(output_file, 'a') as file:

        print(id+'\tR"'+llmAnswer+'"')
        file.write(id+'\tR"'+llmAnswer+'"\n')

        # print("Entity extracted:")
        for entity in entities:
            entity_name = entity[0]
            links = entity_disambiguation(entity_name)
        
            if not links:
                # file.write(id+"\tE"+entity_name+"\tNo links found"+"\n")
                # print(id+"\tE"+entity_name+"\tNo links found")
                continue
        
            ranked_links = disambiguation_scoring(entity_name, llmAnswer, links)
            if ranked_links:
                best_link = ranked_links[0][0]
            else:
                best_link = "" 
                continue   
            match = re.search(r'(?<=\/)([^\/]+)(?=$)', best_link)
            if match:
                try:
                    client = Client()
                    entity = client.get(match.group(0), load=True)
                    url = entity.data['sitelinks']['enwiki']['url']
                    file.write(id+'\tE"'+entity_name+'"\t'+url+'\n')
                    print(id+'\tE"'+entity_name+'"\t'+url)
                except:
                    # no wikipedia link for given wikidata entity
                    pass
                    # file.write(id+'\tE"'+entity_name+'"\t'+best_link+'\n')
                    # print(id+'\tE"'+entity_name+'"\t'+best_link+'"')


def read_input_file(file: str) -> dict:
    """Returns a dictionary of input question/completions
    @param file: path to input text file
    @returns: a dict where k: id, v: str(text)
    """
    questions = dict()

    with open(file, 'r') as file:
        lines = file.readlines()
    
    split_lines = [line.strip().split('\t') for line in lines]

    for line in split_lines:
        if(len(line) != 2):
            print(line)
            raise Exception("Parsing error")
        
        id, question = line[0], line[1]
        questions[id] = question
    

    # print(lines)
    # print(split_lines)
    # print(questions)
    return questions


if __name__ == "__main__":
    
    if len(sys.argv) != 2:
        raise SystemExit("Sorry, incorrect number of arguments")

    if not os.path.exists(sys.argv[1]):
        raise SystemExit("Sorry, incorrect file path")

    input_file = read_input_file(sys.argv[1])
    output_file = "output.txt"

    if os.path.exists(output_file):
        os.remove(output_file)
        # print("output file removed")

    for id, question in input_file.items():
        # print(id +": "+question)
        main(id, question, output_file)
        # raise SystemExit("")
        # break

        
    