from html_output import html_outlines_output


# Dummy model to use for testing.
class DummyModel:
    pass


# Fake functions to replace the outlines behavior.
def fake_transformers(model_name):
    return DummyModel()


def fake_generator(model, schema):
    # Return a generator function that ignores the prompt and returns a predictable dict.
    def generator(prompt):
        return {
            "name": "Old Joe",
            "age": 80,
            "armor": "plate",
            "weapon": "sword",
            "strength": 100,
        }

    return generator


def fake_convert(json):
    # Fake json2html conversion (just check that input is a dict).
    if isinstance(json, dict):
        return "<html><body>Converted</body></html>"
    return ""


def test_html_outlines_output(monkeypatch):
    # Patch the outlines functions.
    import outlines.models
    import outlines.generate

    monkeypatch.setattr(outlines.models, "transformers", fake_transformers)
    monkeypatch.setattr(outlines.generate, "json", fake_generator)

    # Patch json2html.convert in the module where it was imported.
    import html_output

    monkeypatch.setattr(html_output.json2html, "convert", fake_convert)

    # Run the step function.
    output = html_outlines_output()

    # Check that the output is a string containing the expected fake HTML.
    assert isinstance(output, str)
    assert "<html>" in output
