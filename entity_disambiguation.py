import requests
import nltk
from nltk.tokenize import word_tokenize

def entity_disambiguation(entity_name: str) -> list:
    """Returns a list of wikidata links matching the input entity
    @param entity_name: entity
    @returns: a list of wikidata links matching given entity
    """
    entity_name = entity_name.replace(" ", "_")
    query_url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={entity_name}&language=en&format=json"
    try:
        response = requests.get(query_url).json()
    except:
        response = ""
 
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