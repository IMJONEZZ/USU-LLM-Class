from zenml import step
import outlines
from json2html import *


@step
def html_outlines_output():
    schema = """{
        "title": "Character",
        "type": "object",
        "properties": {
            "name": {
                "title": "Name",
                "maxLength": 10,
                "type": "string"
            },
            "age": {
                "title": "Age",
                "type": "integer"
            },
            "armor": {"$ref": "#/definitions/Armor"},
            "weapon": {"$ref": "#/definitions/Weapon"},
            "strength": {
                "title": "Strength",
                "type": "integer"
            }
        },
        "required": ["name", "age", "armor", "weapon", "strength"],
        "definitions": {
            "Armor": {
                "title": "Armor",
                "description": "An enumeration.",
                "enum": ["leather", "chainmail", "plate"],
                "type": "string"
            },
            "Weapon": {
                "title": "Weapon",
                "description": "An enumeration.",
                "enum": ["sword", "axe", "mace", "spear", "bow", "crossbow"],
                "type": "string"
            }
        }
    }"""

    model = outlines.models.transformers("meta-llama/Llama-3.2-1B")
    generator = outlines.generate.json(model, schema)
    character = generator("Give me an old character that is incredibly strong")

    html_output = json2html.convert(json=character)

    return html_output
