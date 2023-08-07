from transformers import LlamaModel, LlamaConfig, AutoTokenizer, LlamaForCausalLM
# Initializing a LLaMA llama-7b style configuration
configuration = LlamaConfig()

# Initializing a model from the llama-7b style configuration
model = LlamaModel(configuration)

# Accessing the model configuration
configuration = model.config

model = LlamaForCausalLM.from_pretrained("7B_converted")
tokenizer = AutoTokenizer.from_pretrained("7B_converted", use_fast=False)

prompt =  """
    Input: name: Lisanna, pronouns: she/her, traits: age: 21 gender: female species: human values: mood: compassionate workethic: ambitious intelligence: dumb luck: lucky magic: non-magical, states: career: bookbinder education: apprenticeship life-changing event: got knighted event: got embarrassed publicly event: received a medal, life history: apprenticeship, got knighted, received a medal, single, received a medal
    Output: Lisanna is a 21-year-old woman who is very compassionate and ambitious. She did an apprenticeship to become a bookbinder. After finishing her apprenticeship, Lisanna got knighted by the king and received a medal. She did all this while being single. Then she received another medal.
    ###
    Input: name: Drake, pronouns: he/him, traits: age: 67 gender: male species: sorcerer values: mood: cold workethic: lazy intelligence: dumb luck: lucky magic: magical, states: career: bounty hunter education: apprenticeship life-changing event: blessed by the gods event: got embarrassed publicly event: received a gift, life history: apprenticeship, received a gift, bounty hunter, got embarrassed publicly, bounty hunter
    Output: Drake is a 67-year-old sorcerer. He is cold-hearted and lazy. Drake did an apprenticeship and became a bounty hunter after he received a gift. Drake got publicly embarrassed and reluctantly returned to his duty as a bounty hunter.
    ###
    Input:  name: Alex, pronouns: they/them, traits: age: 34 gender: non-binary species: dwarf values: mood: cold workethic: lazy intelligence: smart luck: unlucky magic: non-magical, states: career: scholar education: learning on the go life-changing event: got deadly ill event: had a wish come true event: fell down a tree, life history: fell down a tree, scholar, got deadly ill, learning on the go, fell down a tree
    Output: Alex is a 34-year-old dwarf. They are cold-hearted and lazy. When Alex was young, they fell down a tree. They became deadly ill. However, Alex kept learning on the go. Unfortunately, Alex fell down another tree.
    ###
    Input: name: Günhilde, pronouns: she/her, traits: age: 82 gender: female species: dwarf values: mood: cold workethic: lazy intelligence: smart luck: unlucky magic: non-magical, states: career: soldier education: university life-changing event: found a treasure event: had a wish come true event: made a new friend, life history: dating, had a wish come true, dating, had a wish come true, found a treasure
    Output:
    """
inputs = tokenizer(prompt, return_tensors="pt")

# Generate
generate_ids = model.generate(inputs.input_ids, max_length=900)
gen_text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(gen_text)