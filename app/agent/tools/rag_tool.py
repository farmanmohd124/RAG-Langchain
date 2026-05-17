from langchain.tools import tool          # decorator that turns a normal function into a LangChain tool
from app.services.vectorstore import VectorStore  # our vector store service from Step 5

# create a single shared instance of the vector store
# this is module-level so it is created once when the file is imported, not on every tool call
_store = VectorStore()


@tool
def rag_search(query: str) -> str:
    """
    Search the local knowledge base (PDFs and text files) to answer questions.
    Use this tool when the user asks about topics that may be covered in the uploaded documents.
    Returns relevant text chunks along with their source file and page number.
    """
    # query the vector store — returns a list of dicts with score, text, source, page
    results = _store.query(query, top_k=4)

    # if no results were found, return a message so the agent knows to try something else
    if not results:
        return "No relevant documents found in the knowledge base."

    # format each result into a readable block the LLM can understand
    formatted = []
    for i, result in enumerate(results, start=1):        # enumerate gives us index starting at 1
        source = result["source"]                        # filename like "DBMS_Notes.pdf"
        page = result["page"]                            # page number inside the file
        score = round(result["score"], 3)                # round to 3 decimal places for readability
        text = result["text"].strip()                    # remove leading/trailing whitespace

        block = (
            f"[Result {i}] Source: {source} | Page: {page} | Score: {score}\n"
            f"{text}"                                    # the actual chunk content
        )
        formatted.append(block)

    # join all result blocks with a separator so the LLM can clearly see where one ends and next begins
    return "\n\n---\n\n".join(formatted)
