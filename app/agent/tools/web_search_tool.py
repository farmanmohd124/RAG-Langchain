from langchain.tools import tool                        # decorator to register this as an agent tool
from duckduckgo_search import DDGS                      # DuckDuckGo search client (no API key needed)


@tool
def web_search(query: str) -> str:
    """
    Search the web using DuckDuckGo to find information not available in the local knowledge base.
    Use this tool when the user asks about recent events, general knowledge, or topics not covered
    in the uploaded documents. Do not use this for document-specific questions.
    """
    try:
        results = []

        # DDGS() opens a DuckDuckGo search session
        # 'with' ensures the session is properly closed after use (like closing a browser tab)
        with DDGS() as ddgs:
            # .text() performs a text search and returns an iterator of result dicts
            # max_results=4 limits to top 4 results to keep the context concise
            for result in ddgs.text(query, max_results=4):
                title = result.get("title", "No title")    # page title
                href = result.get("href", "")              # URL of the page
                body = result.get("body", "")              # short snippet/summary from the page

                # format each result as a readable block for the LLM
                results.append(
                    f"Title: {title}\nURL: {href}\nSummary: {body}"
                )

        # if DuckDuckGo returned nothing (rare but possible)
        if not results:
            return "No web results found for this query."

        # join all results with a separator so the LLM can clearly read each one
        return "\n\n---\n\n".join(results)

    except Exception as e:
        # catch any network errors or rate limits and return a clean message
        # instead of crashing the whole agent
        return f"Web search failed: {str(e)}"
