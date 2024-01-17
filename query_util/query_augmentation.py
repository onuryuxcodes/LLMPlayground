

def get_augmented_prompt(source_knowledge, query):
    return f"""Using the relevant text from the thesis chapters, 
    answer the query about Onur Yuksel's thesis. 
    Be as brief as possible.
        Text from thesis:
        {source_knowledge}
        Query: {query}"""

