from entity_disambiguation import *
import re
from wikidata.client import Client


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
    just_lower_answer = lower_answer

    if lower_answer.startswith(lower_question):
        just_lower_answer = llm_answer[len(question):].strip()
        just_lower_answer = just_lower_answer.lower()

    if not starts_with_w_word(lower_question):
        yesTokens = ["yes", "indeed", "certainly", "absolutely", "definitely", "of course", "sure", "right", "true", "affirmative", "agreed", "correct", "it is", "it does", "is the"]
        noTokens = ["no", "is not", "are not", "cannot", "do not", "have not", "will not"]

        if starts_with_token(just_lower_answer, yesTokens):
                return "yes"
        elif starts_with_token(just_lower_answer, noTokens):
                return "no"
        else:
            yesScore = 0
            noScore = 0
            for token in yesTokens:
                token_count = just_lower_answer.count(token)
                yesScore += token_count

            for token in noTokens:
                token_count = just_lower_answer.count(token)
                noScore += token_count

            print(just_lower_answer,yesScore, noScore)
            if yesScore > noScore:
                return "yes"
            elif noScore > yesScore:
                return "no"
            else:
                return "yes"
    else:
        if entities:
            filtered_entities = [ent for ent in entities if ent[0].isalpha()]
            if filtered_entities:
                best_entity = filtered_entities[0][0]
                links = entity_disambiguation(best_entity)
                if links:
                    ranked_links = disambiguation_scoring(best_entity, llm_answer, links)
                    best_link = ranked_links[0][0] if ranked_links else None
                    match = re.search(r'(?<=\/)([^\/]+)(?=$)', best_link)
                    if match:
                        try:
                            client = Client()
                            entity = client.get(match.group(0), load=True)
                            url = entity.data['sitelinks']['enwiki']['url']
                            return url
                        except:
                            pass       
                
            
    # Default case: No valid answer found
    return None, "incorrect" #TODO: maybe add "NIL entity" instead

def starts_with_w_word(question):
    pattern = r"^\s*(who|what|when|where|why|which)\b"
    return bool(re.match(pattern, question, re.IGNORECASE))

def starts_with_token(answer, tokens):
    for token in tokens:
        if answer.startswith(token):
            return True
    return False