# 유형별 분류 및 요약을 위한 함수
from openai import OpenAI
client = OpenAI(api_key="")

def validater(context, query):

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": f"""
                Using the provided past memories and the current conversation, 
                output only the numbers of past memories relevant to the current conversation.
                The answer format is as follows.
                Output: 1, 2, 5
                
                past memories:
                {context}
                
                current conversation:
                {query}
                
                Output:
                
                """
            },
        ],
        temperature=0,
        )
    return response.choices[0].message.content

# rewrite없이 과거기억이 필요한지 아닌지 뭍는 코드 
def check_memory(text):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": f"""
                Please classify the following dialogue as requiring past memory or not and print either 1 or 0 accordingly.
                If past memory is required, print 1.
                If past memory is not required, print 0.

                Text Example:
                “How was your trip last year?”

                Output:
                1

                Text Example:
                “The weather is nice today.”

                Output:
                0
                
                Provided dialogue content:
                {text}
                
                Output:
                
                """
            },
        ],
        temperature=0,
        )
    return response.choices[0].message.content

def query_rewriter(text):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": f"""
                Please classify the provided dialogue based on whether it requires past memory or not.
                If past memory is required, use [Past Information Processing].

                [Past Information Processing]:

                “1. If a past date can be inferred using the current date and the provided dialogue, write down the inferred past date. If the date is not important or cannot be inferred, leave it blank.

                2. If there is a specific topic being discussed or a summary-worthy content, summarize it. If the sentence is vague or difficult to retrieve from the vector DB, leave it as is or leave it blank.”

                The output format should follow the example below.

                If no past memory is required, return **None**.

                If past memory is required, use **Past Information Processing**.

                Text Example:

                “[2020/6/21][Speaker1: How was your trip last year?]”

                Output:

                “[2019/6/21][Speaker][Trip]”

                Text Example:

                “[2020/6/21][Speaker2: The weather is nice today.]”

                Output:

                None

                Text Example:

                “[2020/6/21][Speaker1: Do you still like science?]”

                Output:

                “[][Likes science]”

                Text Example:

                “[2020/6/20][Speaker1: Why did you act that way yesterday?]”

                Output:

                “[2020/6/19][]”
                
                Provided dialogue content:
                {text}
                
                Output:
                
                """
            },
        ],
        temperature=0,
        )
    return response.choices[0].message.content