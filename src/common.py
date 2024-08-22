#!/usr/bin/env python
# coding=utf-8
import random


# TEMPLATE="""Given a question and related structured data, please provide your answer.
# Only answer with "Yes," "No," or "Not sure enough."

# Data:
# {data}
# Q: {question}
# A: """

OPTION_MAP = [
    # mainly used
    {"option_a": "Yes", "option_b": "No", "option_c": "Not sure enough"},
    {"option_a": "No", "option_b": "Yes", "option_c": "Not sure enough"},
    {"option_a": "Not sure enough", "option_b": "Yes", "option_c": "No"},

    {"option_a": "Not sure enough", "option_b": "No", "option_c": "Yes"},
    {"option_a": "Yes", "option_b": "Not sure enough", "option_c": "No"},
    {"option_a": "No", "option_b": "Not sure enough", "option_c": "Yes"},
]


TEMPLATE="""Given a question and related structured data, please provide your answer.
Choose one of the following options as your answer:
A) {option_a}
B) {option_b}
C) {option_c}
You only need to output the option ("A", "B" or "C").

Data:
{data}
Q: {question}
A: """


TEMPLATE_COT="""Given a question and related structured data, please first provide an explanation for how you arrived at your answer.
Then, choose one of the following options as your final answer:
A) {option_a}
B) {option_b}
C) {option_c}
You should first output the explanation, followed by the selected option ("A", "B", or "C").

Data:
{data}
Q: {question}
Let's think step by step.

Explanation:"""


TEMPLATE_SHOT="""Given a question and related structured data, please provide your answer.
Choose one of the following options as your answer:
A) {option_a}
B) {option_b}
C) {option_c}
You only need to output the option ("A", "B" or "C").

{examples}

Now, consider the following data and question:

Data:
{data}
Q: {question}
A: """

TEMPLATE_SHOT_COT="""Given a question and related structured data, please first provide an explanation for how you arrived at your answer.
Then, choose one of the following options as your final answer:
A) {option_a}
B) {option_b}
C) {option_c}
You should first output the explanation, followed by the selected option ("A", "B", or "C").

{examples}

Data:
{data}
Q: {question}
Let's think step by step.

Explanation:"""


TEMPLATE_WO_DATA="""Use your knowledge to answer the following question.
Choose one of the following options as your answer:
A) {option_a}
B) {option_b}
C) {option_c}
You only need to output the option ("A", "B" or "C").

Q: {question}
A: """

TEMPLATE_WO_DATA_COT="""Use your knowledge to answer the following question.
Answer with "Yes" or "No" and provide supporting evidence. If you cannot determine the answer, respond with "Not sure enough."

Q: {question}
Let's think step by step.
A: """

TEMPLATE_ENHANCED="""Given structural data and its descriptive prompt, please analyze this data and answer the subsequent question.
Choose one of the following options as your answer:
A) {option_a}
B) {option_b}
C) {option_c}
You only need to output the option ("A", "B" or "C").

Data:
{data}
Q: {question}
A: 
"""

TEMPLATE_STRUCTURE="""Given a question and related structured data, please provide your answer. The given data is in markdown format. Columns in the table are separated by ' | ', rows are separate by '\n', and list element starts with '*'.
Choose one of the following options as your answer:
A) {option_a}
B) {option_b}
C) {option_c}
You only need to output the option ("A", "B" or "C").

Data:
{data}
Q: {question}
A: """

TEMPLATE_SELF_REFINE="""Please review the question based on the structural data and related answer.
Data:
{data}
Question: {question}
Choose one of the following options as your answer:
A) {option_a}
B) {option_b}
C) {option_c}
Answer: {answer}

Please provide feedback and suggest any corrections or additional information that could enhance the answer's accuracy, relevance to the data, or clarity.
Then, choose one of the following options as your final answer:
A) {option_a}
B) {option_b}
C) {option_c}
Feedback:"""

TEMPLATE_MAP = {
    'base': TEMPLATE,
    'cot': TEMPLATE_COT,
    'few_shot': TEMPLATE_SHOT,
    'few_shot_cot': TEMPLATE_SHOT_COT,
    'wo_data': TEMPLATE_WO_DATA,
    'wo_data_cot': TEMPLATE_WO_DATA_COT,
    'enhanced': TEMPLATE_ENHANCED,
    'struct': TEMPLATE_STRUCTURE,
    'refine': TEMPLATE_SELF_REFINE
}


EXAMPLES = [
    {
        "question": r'The Infection Fatality Rate (IFR) is higher for people aged 75 to 84 compared to those aged 35 to 44?',
        "data": """### IFR estimate per age group
    | Age Group | IFR (%)    |  
    |-----------|------------|  
    | 0–34       | 0.004%     |  
    | 35–44     | 0.068%     |  
    | 45–54     | 0.23%      |  
    | 55–64     | 0.75%      |  
    | 65–74     | 2.5%       |  
    | 75–84     | 8.5%       |  
    | 85+        | 28.3%      |""",
        "answer": 'Yes',
        "cot": r'The IFR for people aged 75 and above is 8.5%, while the IFR for the age group 35–44 is 0.068%.'
    },
    {
        "question": r'Did Justin Gatlin maintain his exact position in the top 3 of the 100m final from the 2012 Olympics to the 2016 Olympics?',
        "data": """### 2012 Olympics 100m Final Top 3 Results  
    | Rank | Athlete           | Time (seconds) |  
    |------|-------------------|----------------|  
    | 1    | Usain Bolt        | 9.63           |  
    | 2    | Yohan Blake       | 9.75           |  
    | 3    | Justin Gatlin     | 9.79           |

    ### 2016 Olympics 100m Final Top 3 Results 
    | Rank | Athlete           | Time (seconds) |  
    |------|-------------------|----------------|  
    | 1    | Usain Bolt        | 9.81           |  
    | 2    | Justin Gatlin     | 9.89           |  
    | 3    | Andre De Grasse   | 9.91           |""",
        "answer": 'No',
        "cot": r'In 2012, Justin Gatlin finished 3rd, but in 2016, he moved up to 2nd place.'
    },
    {
        "question": r'''In this election, was Donald Trump's vote percentage higher than that of any other candidate besides Hillary Clinton?''',
        "data": """### 2016 U.S. Presidential Election Vote Percentages
    | Candidate         | Party          | Vote Percentage |
    | ----------------- | -------------- | --------------- |
    | Donald Trump      | Republican     | 46.1%           |
    | Hillary Clinton   | Democratic     | 48.2%           |
    | Gary Johnson      | Libertarian    |                 |
    | Jill Stein        | Green          |                 |
    | Evan McMullin     | Independent    |                 |
    | Other Candidates  |                |                 |""",
        "answer": 'Yes',
        "cot": r'''With 46.1%, Donald Trump's vote share exceeds any other candidate besides Hillary Clinton, who had 48.2%, leaving only 5.7% for all others combined.'''
    },
    {
        "question": r'''In the tenth year after Constantinople was renamed Istanbul, was the population 1000,000?''',
        "data": """Up until the year 330 Istanbul was known as Byzantium, and then until 1453 Constantinople. Its current name of Istanbul only came into being on the 28th March 1930.
    | Year |    Population  | ±% p.a. |  
    |  -- | ---------------- | ----- |  
    | 1925 |       881,000      |         |  
    | 1927 |       691,000      | -11.44% |  
    | 1935 |       740,800      | +0.87%  |  
    | 1940 |       793,900      | +1.39%  |  
    | 1945 |       845,300      | +1.26%  |  
    | 1950 |       983,000      | +3.06%  |  
    | 1960 |     1,459,500      | +4.03%  |  
    | 1965 |     1,743,000      | +3.61%  |  
    | 1970 |     2,132,400      | +4.12%  |  
    | 1975 |     2,547,400      | +3.62%  |  """,
        "answer": 'No',
        "cot": r'''In the tenth year after the renaming, which was 1940, the population was 793,900.'''
    },
    {
        "question": r'Was the series named international formula master from 2007 to 2009?',
        "data": """### international formula master 
    | Season | Series Name                  | Champion                     | Team Champion            | Secondary Class Champion                   |  
    |--------|--------------------------------|--------------------------------|----------------------------------------|--------------------------------------------|  
    | 2005   | 3000 Pro Series                | Norbert Siedler / Max Busnelli  | Drako Junior Team          | Iago Rego Rosende (Master Junior Formula)  |  
    | 2006   | F3000 International Masters    | Jan Charouz                    | Charouz Racing System      | Daniel Campos - Hull (Master Junior Formula) |  
    | 2007   | International Formula Master   | Jerome Dambrosio               | Cram Competition           | Isaac Lopez Navarro (Master Junior Formula)  |  
    | 2008   | International Formula Master   | Chris van der Drift            | JD Motorsport              | Marcello Puglisi (Formula Master Italia)   |  
    | 2009   | International Formula Master   | Fabio Leimer                   | JD Motorsport              | Alexander Rossi (Rookie of the Year)         |""",
        "answer": 'Yes',
        "cot": r'''Series was named 'International Formula Master' from 2007-2009.'''
    }
]


def generate_few_shot_examples(n, cot, answer_map):
    sampled_examples = random.sample(EXAMPLES, n)
    examples = []
    data_placeholder = "[data example]"
    for example in sampled_examples:
        if cot:
            examples.append(f"""Data:\n{data_placeholder}\nQ: {example['question']}\nExplanation: {example['cot']}\nFinal Answer: {answer_map[example['answer']]}.""")
        else:
            examples.append(f"""Data:\n{data_placeholder}\nQ: {example['question']}\nA: {answer_map[example['answer']]}.""")
    if n > 1:
        out='Here are some examples.\n'
    else:
        out='Here is an example.\n'

    return out + "\n\n".join(examples)


def generate_few_shot_examples_v2(n, cot, answer_map):
    sampled_examples = random.sample(EXAMPLES, n)
    examples = []
    for example in sampled_examples:
        if cot:
            examples.append(f"""Data:\n{example['data']}\nQ: {example['question']}\nExplanation: {example['cot']}\nFinal Answer: {answer_map[example['answer']]}.""")
        else:
            examples.append(f"""Data:\n{example['data']}\nQ: {example['question']}\nA: {answer_map[example['answer']]}.""")
    if n > 1:
        out='Here are some examples.\n'
    else:
        out='Here is an example.\n'

    return out + "\n\n".join(examples)
