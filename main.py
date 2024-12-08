from llama_cpp import Llama
import spacy
from nltk.tokenize import word_tokenize
from wikidata.client import Client
import requests
import re
import contextlib
import sys
import os

# All models are available at https://huggingface.co/TheBloke. Make sure you download the ones in the GGUF format
model_path = "/home/user/models/llama-2-7b.Q4_K_M.gguf"
 
def entity_disambiguation(entity_name: str) -> list:
    """Returns a list of Wikidata links matching the input entity
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

def extract_answer(llm_answer: str, question: str, entities: set) -> tuple:
    """Extracts the answer and its type (yes/no or entity)
    @param llm_answer: llm output
    @param question: llm input question
    @param entities: 
    @return type: tuple 
    @returns: answer, correctness
    """
    lower_answer = llm_answer.lower()
    lower_question = question.lower()
 
    # 1. is yes/no in the question
    if any(k in lower_answer for k in ["yes", "no"]):
        answer = "yes" if "yes" in lower_answer else "no"
        correctness = "correct" if answer in lower_answer else "incorrect" # this wont work e.g.: "is not correct"
        return answer, correctness
 
    # 2. Implicit yes/no extraction
    if any(k in lower_question for k in ["is", "are", "does", "yes or no"]):
        # Look for patterns like "Rome is the capital of Italy"
        if any(verb in lower_answer for verb in [" is ", " does "]): # "everybody knows that Italy is a beautiful country in Europe, but not everyone knows that the capital of Italy is Rome"
            if "not" in lower_answer: # 
                return "no", "correct"
            else:
                return "yes", "correct"
 
    # 3. Entity extraction
    if entities: # meh
        # Filter and rank entities
        filtered_entities = [ent for ent in entities if ent[0].isalpha()]
        if filtered_entities:
            best_entity = filtered_entities[0][0]  # Take the first relevant entity
            links = entity_disambiguation(best_entity)
            if links:
                ranked_links = disambiguation_scoring(best_entity, llm_answer, links)
                best_link = ranked_links[0][0] if ranked_links else None
                return best_link, "correct" if best_link else "incorrect"
 
    # Default case: No valid answer found
    return None, "incorrect" #TODO: maybe add "NIL entity" instead

def main(id: str, question: str, output_file: str) -> None:
    """Prints (and saves to output file) out the disambiguated entity and corresponding Wikipedia link (assures correct output format)
    @param id: question id
    @param question: llm input string
    @param output_file: path to output text file
    @returns: None
    """
    llm = Llama(model_path=model_path, verbose=False, n_ctx=4096)

    output = llm(
        question, # Prompt
        max_tokens=32, # Generate up to 32 tokens
        # stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
        stop=None,
        echo=True # Echo the prompt back in the output
    )

    llm_answer = output['choices'][0]['text']

    #NER - SpaCy
    with contextlib.redirect_stdout(None):    
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            from spacy.cli import download
            download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")

    text = nlp(llm_answer)
    entities = set((ent.text, ent.label_) for ent in text.ents)


    with open(output_file, 'a') as file:

        print(id+'\tR"'+llm_answer+'"')
        file.write(id+'\tR"'+llm_answer+'"\n')

        # print("Entity extracted:")
        for entity in entities:
            entity_name = entity[0]
            links = entity_disambiguation(entity_name)
        
            if not links: # perhaps NIL entity?
                continue
        
            ranked_links = disambiguation_scoring(entity_name, llm_answer, links)
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

            # Extract the answer and its correctness
        extracted_answer, correctness = extract_answer(llm_answer, question, entities)
 
        # Write extracted answer and correctness
        if extracted_answer:
            file.write(f"{id}\tA\"{extracted_answer}\"\t{correctness}\n")
            print(f"{id}\tA\"{extracted_answer}\"\t{correctness}")


def read_input_file(file: str) -> dict:
    """Returns a dictionary of input question/completions
    @param file: path to input text file
    @returns: a dict where k: id, v: str(text)
    """
    questions = dict()

    with open(file, 'r') as file:
        lines = file.readlines()
    
    split_lines = [line.strip().split('\t') for line in lines]

    for i, line in enumerate(split_lines):
        if(len(line) != 2):
            raise Exception(f"Input file parsing error in line {i+1}: {line}")
        
        id, question = line[0], line[1]
        questions[id] = question
    
    return questions


if __name__ == "__main__":
    
    if len(sys.argv) != 2:
        raise SystemExit(f"Sorry, incorrect number of arguments\nUsage: python3 {sys.argv[0]} input_file.txt")

    if not os.path.exists(sys.argv[1]):
        raise SystemExit(f"Sorry, incorrect file path\n{sys.argv[1]} not found :(")

    input_file = read_input_file(sys.argv[1])
    output_file = "output.txt"

    if False:
        import nltk
        nltk.download('punkt_tab')


    if os.path.exists(output_file):
        os.remove(output_file)
        # print("output file removed")

    for id, question in input_file.items():
        # print(id +": "+question)
        main(id, question, output_file)
        # raise SystemExit("")
        # break

        
    