from llama_cpp import Llama
import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
import wikipediaapi


model_path = "/home/user/models/llama-2-7b.Q4_K_M.gguf"

# If you want to use larger models...
#model_path = "models/llama-2-13b.Q4_K_M.gguf"
# All models are available at https://huggingface.co/TheBloke. Make sure you download the ones in the GGUF format

question = "What is the capital of Italy? "
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
# #nltk - Entity recognition
# nltk.download('words')
# nltk.download('punkt_tab')
# nltk.download('averaged_perceptron_tagger_eng')
# nltk.download('maxent_ne_chunker_tab')

# # Tokenize and tag the text
# tokens = word_tokenize(llmAnswer)
# pos_tags = pos_tag(tokens)

# # Named Entity Recognition
# entities = ne_chunk(pos_tags)
# print("\nEntities using NLTK:")
# for entitiy in entities:
#     if hasattr(entitiy, 'label'):
#         print(f"{' '.join(c[0] for c in entitiy)} ({entitiy.label()})")

#------------------------------------------------------------------------
#spacy - Entity recognition
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

text = nlp(llmAnswer)

entities = [(ent.text, ent.label_) for ent in text.ents]
print("\nEntities Extracted:\n")
for entity in entities:
    print(f"{entity[0]} ({entity[1]})")


nlp.add_pipe("entityLinker", last=True)


# Entity Disambiguation using Wikipedia API
def entity_disambiguation(entity_name):
    wiki_wiki = wikipediaapi.Wikipedia('MyProjectName (merlin@example.com)', 'en')
    page = wiki_wiki.page(entity_name)

    if not page.exists():
        return None
    return page.fullurl

# Printing Entity Disambiguation
print("\nEntities Disambiguation:\n")

for ent in text.ents:
    # print(f"Entity: {ent.text}, Label: {ent.label_}")
    disambiguated_info = entity_disambiguation(ent.text)
    if disambiguated_info:
        print(ent.text,"<TAB>",disambiguated_info)
    else:
        print(ent.text,"No disambiguation found.")
    print("\n")