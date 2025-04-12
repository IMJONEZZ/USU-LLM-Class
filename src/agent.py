# agent script

from duckduckgo_search import DDGS


class WebAgent:
    def __init__(self):
        self.memory = {}

    def remember(self, session_id, user_input, model_output):
        if session_id not in self.memory:
            self.memory[session_id] = []
        self.memory[session_id].append((user_input, model_output))

    def get_context(self, session_id, max_turns=3):
        history = self.memory.get(session_id, [])
        return history[-max_turns:]

    def search_web(self, query, max_results=3):
        try:
            with DDGS() as ddgs:
                results = ddgs.text(query, max_results=max_results)
                if not results:
                    return "No relevant results found."
                return "\n".join(
                    f"- {r['title']} ({r['href']}): {r['body']}"
                    for r in results
                    if "body" in r
                )
        except Exception as e:
            return f"Search failed: {e}"

    def build_prompt_with_web(self, session_id, user_input):
        context = self.get_context(session_id)
        web_info = self.search_web(user_input)

        history = ""
        for inp, out in context:
            history += f"User: {inp}\nAssistant: {out}\n"

        prompt = f"""You are a helpful and knowledgeable assistant.
Here are some relevant web search results:
{web_info}
User: {user_input}
Assistant:"""
        return prompt.strip()
