"""
Golden test dataset for RAG pipeline evaluation.
Questions are grounded in Equal Experts case studies with known expected answers.
"""

TEST_CASES = [
    {
        "id": "tc_001",
        "question": "What work did Equal Experts do for IG Group?",
        "expected_answer": (
            "Equal Experts helped IG Group improve DevOps practices through an embedded "
            "partnership. Key outcomes included reducing change failure rate from 33% to zero, "
            "increasing code coverage from 40% to 90%, and reducing deployment time from "
            "60 days to a matter of hours."
        ),
        "relevant_source": "IG Group",
    },
    {
        "id": "tc_002",
        "question": "What were the key findings of the Forrester study on Equal Experts?",
        "expected_answer": (
            "The Forrester Total Economic Impact study found $61 million in benefits from "
            "reduced costs and increased revenues, an 11 month reduction in time to market, "
            "and over $500,000 in losses avoided over three years."
        ),
        "relevant_source": "Forrester",
    },
    {
        "id": "tc_003",
        "question": "How did Equal Experts help Spirit Super?",
        "expected_answer": (
            "Equal Experts helped Spirit Super introduce event-driven architecture and "
            "cloud-native microservices. Results included reducing third-party integration "
            "time from 8 weeks to 2 days, achieving zero downtime, and increasing team "
            "capacity for new project work from 5% to 66%."
        ),
        "relevant_source": "Spirit Super",
    },
    {
        "id": "tc_004",
        "question": "What did Equal Experts build for Travelopia using GenAI?",
        "expected_answer": (
            "Equal Experts built a GenAI chatbot called Athena for Travelopia's Enchanting "
            "Travels brand. The chatbot reduced customer response times from 24 hours to "
            "instant and was rolled out to the entire team of Travel Coordinators within "
            "a few weeks."
        ),
        "relevant_source": "Travelopia",
    },
    {
        "id": "tc_005",
        "question": "What were the results of the HMRC Illuminate platform?",
        "expected_answer": (
            "The Illuminate platform identified millions of pounds in confirmed tax at risk "
            "and saved tens of millions by stopping fraudulent claims before they were paid. "
            "It enabled bulk searching of unstructured data across millions of documents."
        ),
        "relevant_source": "HMRC",
    },
    {
        "id": "tc_006",
        "question": "How did Equal Experts reduce time to market for John Lewis?",
        "expected_answer": (
            "Equal Experts helped John Lewis build a cloud-based digital platform using "
            "microservices and a You Build It You Run It model. Deployments increased from "
            "10 per year to 4000 per year and the new search algorithm generated an "
            "estimated £30 million in extra revenue."
        ),
        "relevant_source": "John Lewis",
    },
    {
        "id": "tc_007",
        "question": "What approach did Equal Experts take with the Australian superannuation security architecture client?",
        "expected_answer": (
            "Equal Experts embedded a Security Architecture Consultant who streamlined "
            "requirements sharing and introduced a fast-track process for smaller low-risk "
            "projects. This reduced security architecture decision time from 12 weeks to "
            "3 weeks."
        ),
        "relevant_source": "Australian superannuation fund",
    },
]
