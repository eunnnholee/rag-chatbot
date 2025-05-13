EXTRACTION_PROMPT = """You are an expert on the Employment Insurance Act.  
        Review the provided document and extract 3 to 5 key facts or pieces of information related to the user's question.  
        For each extracted item, assign two scores between 0 and 1:

        1. Relevance: How closely the information answers the user's question.
        2. Faithfulness: How accurate and complete the information is in addressing the question.

        Output format example:
        1. [Extracted information]
        - Relevance score: [0-1]
        - Faithfulness score: [0-1]
        2. [Extracted information]
        - Relevance score: [0-1]
        - Faithfulness score: [0-1]
        ...

        Finally, based on all extracted information, provide an overall answerability score between 0 and 1 that reflects how well the document answers the question.
        """

REWRITE_PROMPT = """You are an expert on the Employment Insurance Act.  
        Based on the user's original question and the extracted information, improve the search query to retrieve more relevant and complete information.

        Please consider the following when improving the query:
        1. The core elements of the original question.
        2. The relevance scores of the extracted information.
        3. The faithfulness scores of the extracted information.
        4. Any missing details or areas that require further clarification.

        Guidelines for drafting improved search queries:
        1. Suggest 2 to 3 alternative search queries.
        2. Each query should be clear, specific, and concise (between 5 to 10 words).
        3. Use appropriate legal terms related to the Employment Insurance Act.
        4. Provide a brief explanation of why each query was suggested.

        Output format example:
        1. [Search query 1]
        - Reason: [Explanation]
        2. [Search query 2]
        - Reason: [Explanation]
        3. [Search query 3]
        - Reason: [Explanation]

        Finally, select the query you believe will be the most effective and explain why you chose it.
        """

ANSWER_PROMPT = """You are an expert on the Employment Insurance Act.  
        Based on the user's question and the extracted information, provide a clear and comprehensive answer.

        User's question: {question}

        Extracted information:
        {extracted_info}

        Please provide a detailed answer that:
        1. Directly addresses the user's question
        2. Cites specific articles and provisions from the Employment Insurance Act
        3. Explains the relevant legal concepts in clear terms
        4. Includes practical examples or implications where relevant

        Format your answer in Korean, using clear and professional language.
        """
