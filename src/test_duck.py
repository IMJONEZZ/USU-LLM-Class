from duckduckgo_search import DDGS

with DDGS() as ddgs:
    for result in ddgs.text("current weather in New York", max_results=3):
        print(result["title"], "-", result["href"])
