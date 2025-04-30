from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams,LLMTestCase
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI

llm=ChatOpenAI(model='gpt-4o-mini')
text="write a sql query to find the maximum salary of the employee whose age is greater than 30"
test_case1=LLMTestCase(
    input=text,
    actual_output=llm.invoke(text).content,
)
answer_relevancy=GEval(
    name="correctness",
    criteria="check whather the actual output is correct according to the input",
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT
    ]
)
answer_relevancy.measure(test_case1)
print(answer_relevancy.score)

print(answer_relevancy.reason)

print(test_case1.actual_output)
