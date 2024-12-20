import re
from wikidata.client import Client
from sentence_transformers import CrossEncoder
from entity_disambiguation import *

model = CrossEncoder('abbasgolestani/ag-nli-DeTS-sentence-similarity-v1')
question_type = 0

def extract_answer(answer: str, llAnswer: str, question: str, entities: set, entitiesAnswer: set) -> tuple:
    """Extracts the answer and its type (yes/no or entity)
    @param llm_answer: llm output
    @param question: llm input question
    @param entities: 
    @return type: tuple 
    @returns: answer, correctness
    """
    
    lower_question = question.lower()
    lower_answer = answer.lower()

    if not starts_with_w_word(lower_question):
        question_type = 0
        yesTokens = ["yes", "indeed", "certainly", "absolutely", "definitely", "of course", "sure", "right", "true", "affirmative", "agreed", "correct"]
        noTokens = ["no", "not", "none", "never", "cannot", "isn't", "aren't", "don't", "won't", "haven't"]

        if starts_with_token(lower_answer, yesTokens):
                return "yes", question_type
        elif starts_with_token(lower_answer, noTokens):
                return "no", question_type
        else:
            yesScore = 0
            noScore = 0
            for token in yesTokens:
                token_count = lower_answer.count(token)
                yesScore += token_count

            for token in noTokens:
                token_count = lower_answer.count(token)
                noScore += token_count

            if yesScore > noScore:
                return "yes", question_type
            elif noScore > yesScore:
                return "no", question_type
            else:
                return "yes", question_type
    else:
        question_type = 1
        question_ents = [(question, ent[0]) for ent in entitiesAnswer]
        similarities = cosine_similarity(question_ents)

        maxVal = -1000
        bestTup = 0
        for tup, val in zip(entitiesAnswer, similarities):
            if maxVal < val:
                bestTup = tup
                maxVal = val
        bestEntity = bestTup[0]
        links = entity_disambiguation(bestEntity)

        if links:
            ranked_links = disambiguation_scoring(bestEntity, llAnswer, links)
            best_link = ranked_links[0][0]
            match = re.search(r'(?<=\/)([^\/]+)(?=$)', best_link)
            if match:
                try:
                    client = Client()
                    entity = client.get(match.group(0), load=True)
                    url = entity.data['sitelinks']['enwiki']['url']
                    return url, question_type
                except:
                    pass            
    return "no", question_type

def starts_with_w_word(question: str) -> bool:
    pattern = r"^\s*(who|what|when|where|why|which)\b"
    return bool(re.match(pattern, question, re.IGNORECASE))

def starts_with_token(answer: str, tokens: str) -> bool:
    for token in tokens:
        if answer.startswith(token):
            return True
    return False

def cosine_similarity(question_ents: list) -> float:
    if not question_ents:
        return question_ents
    similarity_scores = model.predict(question_ents, show_progress_bar=False)
    return similarity_scores