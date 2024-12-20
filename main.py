from llama_cpp import Llama
import spacy
import nltk
from nltk.tokenize import word_tokenize
from wikidata.client import Client
import requests
import re
import contextlib
import sys
import os

from answer_extraction import *
from entity_disambiguation import *

# If you want to use larger models...
model_path = "/home/user/models/llama-2-7b.Q4_K_M.gguf"
# All models are available at https://huggingface.co/TheBloke. Make sure you download the ones in the GGUF format

def main(id: str, question: str, output_file: str) -> None:
    """Prints (and saves to output file) out the disambiguated entity and corresponding Wikipedia link (assures correct output format)
    @param id: question id
    @param question: llm input string
    @param output_file: path to output text file
    @returns: None
    """

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

    answer = llmAnswer.lower()
    question = question.lower()

    if answer.startswith(question):
        answer = llmAnswer[len(question):].strip()

    textAnswer = nlp(answer)
    entitiesAnswer = set((ent.text, ent.label_) for ent in textAnswer.ents)

    with open(output_file, 'a') as file:

        print(id+'\tR"'+llmAnswer+'"')
        file.write(id+'\tR"'+llmAnswer+'"\n')

        for entity in entities:
            entity_name = entity[0]
            links = entity_disambiguation(entity_name)
        
            if not links:
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
                    pass

    # Write extracted answer and correctness
    extracted_answer = extract_answer(answer, llmAnswer, question, entities, entitiesAnswer)
    if extracted_answer:
        print(f"{id}\tA\"{extracted_answer}")

    correctness = "Correct"
    if True:
        print(f"{id}\tA\"{correctness}")

def read_input_file(file: str) -> dict:
    """Returns a dictionary of input question/completions
    @param file: path to input text file
    @returns: a dict where k: id, v: str(text)
    """
    questions = dict()

    with open(file, 'r') as file:
        lines = file.readlines()
    
    split_lines = [line.strip().split('\t') for line in lines if line.strip()]
    
    for i, line in enumerate(split_lines):
        if(len(line) != 2):
            print(fr"Parsing error in line {i}: {line}, len: {len(line)} != 2, sep: \t")
            continue

        id, question = line[0], line[1]
        questions[id] = question

    return questions


if __name__ == "__main__":
    
    if len(sys.argv) != 2:
        raise SystemExit("Sorry, incorrect number of arguments")

    if not os.path.exists(sys.argv[1]):
        raise SystemExit("Sorry, incorrect file path")

    input_file = read_input_file(sys.argv[1])
    output_file = "output.txt"

    nltk.download('punkt_tab')


    if os.path.exists(output_file):
        os.remove(output_file)

    for id, question in input_file.items():
        main(id, question, output_file)

        
    