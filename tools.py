
import os
import arxiv
from dotenv import load_dotenv
from langchain_tavily import TavilySearch

load_dotenv()

# Tool 1: Web search
tavily = TavilySearch(max_results=5)

# Tool 2: Academic paper search
def search_arxiv(query: str) -> str:
    """Search arXiv for relevant academic papers."""
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=5,
        sort_by=arxiv.SortCriterion.Relevance
    )

    papers = []
    for paper in client.results(search):
        papers.append(
            f"Title: {paper.title}\n"
            f"Summary: {paper.summary}\n"
            f"URL: {paper.entry_id}\n"
        )

    return "\n---\n".join(papers) if papers else "No papers found."


print("Tavily:", tavily.invoke({"query": "LangGraph AI agents"}))
print("arXiv:", search_arxiv("large language model agents"))
