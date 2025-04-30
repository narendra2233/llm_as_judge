from deepeval.metrics import AnswerRelevancyMetric,GEval
from deepeval.test_case import LLMTestCase,LLMTestCaseParams
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm=ChatOpenAI(model='gpt-4o-mini')
text="write a query to find maximum salary of the employee whose age is greater than 30"
test_case=LLMTestCase(
    input=text,
    actual_output="""SELECT MIN(salary) AS max_salary
FROM employees
WHERE age < 30"""
)
answer_relevancy=GEval(
    name="Correctness",
    criteria="Check whather the actual output is correct according to the input",
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT
    ],
    verbose_mode=True
    
)
answer_relevancy.measure(test_case)

print(answer_relevancy.score)

print(test_case.actual_output)
print(answer_relevancy.reason)
