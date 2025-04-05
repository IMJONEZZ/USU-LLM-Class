import dspy

# ========== Module: Decide what to do ==========

class DecideTask(dspy.Signature):
    """Decide whether to use the internet or optimize a prompt."""
    user_input = dspy.InputField()
    plan = dspy.OutputField(desc="Say either 'use internet' or 'optimize prompt' and explain why.")

class TaskDecider(dspy.Module):
    def forward(self, user_input):
        return dspy.Predict(DecideTask)(user_input=user_input)

# ========== Module: Internet agent ==========

class SearchQuery(dspy.Signature):
    """Generate a search engine query from user input."""
    user_input = dspy.InputField()
    search_query = dspy.OutputField(desc="A search engine query to answer the user input.")

class WebAnswer(dspy.Signature):
    """Generate a final answer using search results."""
    user_input = dspy.InputField()
    web_info = dspy.InputField()
    final_answer = dspy.OutputField(desc="A helpful answer to the user's question.")

class InternetAgent(dspy.Module):
    def __init__(self, search_tool=None):
        super().__init__()
        self.search_tool = search_tool or (lambda q: ["This is a mock search result about: " + q])
        self.query_generator = dspy.Predict(SearchQuery)
        self.answer_generator = dspy.Predict(WebAnswer)

    def forward(self, user_input):
        query = self.query_generator(user_input=user_input).search_query
        search_results = self.search_tool(query)
        web_info = "\n".join(search_results)
        final_answer = self.answer_generator(user_input=user_input, web_info=web_info).final_answer
        return final_answer

# ========== Module: Prompt optimizer ==========

class ImprovePrompt(dspy.Signature):
    """Rewrite the prompt"""
    user_prompt = dspy.InputField()
    improved_prompt = dspy.OutputField(desc = "Improved prompt with more clarity and specificity.")

class PromptOptimizer(dspy.Module):
    def forward(self, user_prompt):
        return dspy.Predict(ImprovePrompt)(user_prompt=user_prompt).improved_prompt

# ========== Agent ==========

class ConditionalAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.decider = TaskDecider()
        self.internet = InternetAgent()
        self.optimizer = PromptOptimizer()

    def forward(self, user_input):
        plan = self.decider(user_input=user_input).plan.lower()
        if "internet" in plan:
            result = self.internet(user_input)
        else:
            result = self.optimizer(user_prompt=user_input)
        return result