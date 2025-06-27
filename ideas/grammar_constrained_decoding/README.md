# Grammar Constrained Decoding

# TODO
- [x] get and run llama cpp
- [x] input some non-CFG restrictions, using `guidance`
- [x] input some CFG restrictions, using `guidance`
- [x] regular expressions vs *context*-free grammars vs regular grammars

## General 
* way of limiting the output of a language model to a specific grammar or structure, like JSON, specific programming languages, API responses, etc.
* grammar can be input using BNF (Backus-Naur Form) or just by providing simple 'options' for the model to choose from

## Grammars
| Type      | Grammar          | Language     | Automata |
|---------------|-------------------|--------------|--------------|
| Type 0 | Unrestricted Grammar   | Recursive Enumerable Language| Turing Machine|
| Type 1 | Context Sensitive Grammar     | Context Sensitive Language| Linear Bounded Automaton|
| Type 2 | Context Free Grammar     | Context Free Language| Pushdown Automaton|
| Type 3 | Regular Grammar     | Regular Language| Finite State Automaton|

## References
* `libraries`
  * [guidance](https://github.com/guidance-ai/guidance)
  * [llguidance perf with rust](https://github.com/guidance-ai/llguidance)
  * [outlines](https://github.com/dottxt-ai/outlines)
* `papers`
  * [guiding llms the right way](https://arxiv.org/html/2403.06988v1)
  * [grammar constrained decoding for structured NLP tasks](https://arxiv.org/html/2305.13971v6)
  * [generating structured outputs from language models](https://arxiv.org/html/2501.10868v1)
* `articles`
  * [intro to context-free grammars](https://www.geeksforgeeks.org/theory-of-computation/what-is-context-free-grammar/)
  * [wiki on context-free grammars](https://en.wikipedia.org/wiki/Context-free_grammar)
  * [python libraries overview](https://medium.com/@docherty/python-libraries-for-llm-structured-outputs-beyond-langchain-621225e48399)
  * [logits intro](https://telnyx.com/learn-ai/logits-ai)
  * [llguidance blogpost](https://guidance-ai.github.io/llguidance/llg-go-brrr)
* `datasets`
  * [JSONSchemaBench](https://github.com/guidance-ai/jsonschemabench)
* `yt`
  * [cfg and chomsky normal form](https://youtu.be/q3zFKA1VcgQ)
  * [chomsky hierarchy of grammars](https://youtu.be/9idnQ2C6HfA)