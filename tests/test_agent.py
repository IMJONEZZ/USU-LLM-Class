from agent import TaskDecider, InternetAgent, PromptOptimizer, ConditionalAgent


def test_mock_prompt_optimizer():
    optimizer = PromptOptimizer()
    result = optimizer.forward("make this prompt better")
    assert isinstance(result, str)


def test_mock_internet_agent():
    agent = InternetAgent()
    result = agent.forward("who is the president of the USA?")
    assert "mock search result" in result.lower()


def test_conditional_agent_routing():
    agent = ConditionalAgent()
    # Force the decider to route to the optimizer
    agent.decider.forward = lambda user_input: type("MockOutput", (), {"plan": "optimize prompt"})
    result = agent.forward("help me rewrite this")
    assert isinstance(result, str)
