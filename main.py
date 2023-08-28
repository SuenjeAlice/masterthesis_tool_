# main, run to run program
from decision_tree import Value_Assignment
from markov_chain import History_Assignment
from llm import LLM


#traits:
class Traits:
    traits = {
        "age": 82,
        "gender": 1,
        "species": 5
    }

name = "Günhilde"
pronouns = "she/her"

traits, values, states = Value_Assignment().decision_tree_values(Traits.traits)
life_history = History_Assignment().make_Life_History(values, states)
prompt = LLM().create_prompt(life_history, traits, values, states, name, pronouns)
#story = LLM().GPT_J(life_history, traits, values, states, name, pronouns)
#story = LLM().LLaMA(life_history, traits, values, states, name, pronouns)
#story = LLM().vicuna(life_history, traits, values, states, name, pronouns)

print("Traits: ", traits)
print("Values: ", values)
print("States: ", states)
print("Life History: ", life_history)
print("Prompt: ", prompt)
#print("Story: ", story)


