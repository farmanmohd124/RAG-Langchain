from typing import Dict, List                                    # type hints
from langchain.agents import create_tool_calling_agent, AgentExecutor  # agent builder + runner
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # prompt structure
from langchain_core.messages import HumanMessage, AIMessage     # message types for chat history
from app.services.llm_factory import get_llm                    # our LLM switcher from Step 3
from app.agent.tools.rag_tool import rag_search                 # RAG search tool from Step 6
from app.agent.tools.web_search_tool import web_search          # web search tool from Step 7


# system prompt tells the agent who it is and how to behave
# this text is injected at the start of every conversation
SYSTEM_PROMPT = """You are a helpful AI assistant with access to two tools:

1. rag_search: Search the local knowledge base (uploaded PDFs and documents).
   Always try this first when the question might be answered by the documents.

2. web_search: Search the web for information not available in the documents.
   Use this as a fallback when rag_search returns no relevant results.

After using a tool, always cite your sources by mentioning the document name and page number.
Be concise, accurate, and helpful. If you are unsure, say so.
"""

# list of all tools the agent can use
# adding a new tool later is as simple as appending to this list
TOOLS = [rag_search, web_search]


class RAGAgent:
    def __init__(self):
        self.llm = get_llm()           # get the configured LLM (Groq or Claude) from factory

        # the prompt template structures every conversation
        # MessagesPlaceholder is a slot that gets filled with actual messages at runtime
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),                  # fixed system instructions at the top
            MessagesPlaceholder("chat_history"),         # slot for past conversation messages
            ("human", "{input}"),                        # slot for the current user message
            MessagesPlaceholder("agent_scratchpad"),     # slot for the agent's internal reasoning steps
        ])

        # create_tool_calling_agent builds a ReAct agent that can call our tools
        # it combines the LLM + tools + prompt into an agent "brain"
        agent = create_tool_calling_agent(
            llm=self.llm,
            tools=TOOLS,
            prompt=self.prompt,
        )

        # AgentExecutor is the runtime loop that:
        # 1. sends input to agent
        # 2. executes any tool calls the agent makes
        # 3. feeds tool results back to agent
        # 4. repeats until agent gives a final answer
        self.executor = AgentExecutor(
            agent=agent,
            tools=TOOLS,
            verbose=True,           # prints the agent's reasoning steps to terminal (great for debugging)
            max_iterations=5,       # stop after 5 tool calls to prevent infinite loops
            handle_parsing_errors=True,  # if LLM output is malformed, recover gracefully instead of crashing
        )

        # stores conversation history per session
        # key = session_id (string), value = list of HumanMessage / AIMessage objects
        self.sessions: Dict[str, List] = {}

    def _get_history(self, session_id: str) -> List:
        # return existing history for this session, or an empty list if new session
        return self.sessions.get(session_id, [])

    def _save_history(self, session_id: str, human_msg: str, ai_msg: str):
        # if this session doesn't exist yet, create an empty list for it
        if session_id not in self.sessions:
            self.sessions[session_id] = []

        # append the new human message and AI response to this session's history
        self.sessions[session_id].append(HumanMessage(content=human_msg))
        self.sessions[session_id].append(AIMessage(content=ai_msg))

        # keep only the last 10 messages (5 turns) to avoid the context window getting too large
        self.sessions[session_id] = self.sessions[session_id][-10:]

    def chat(self, query: str, session_id: str = "default") -> Dict:
        # get the conversation history for this session
        history = self._get_history(session_id)

        # run the agent with the current query and past history
        response = self.executor.invoke({
            "input": query,              # the current user question
            "chat_history": history,     # all previous messages in this session
        })

        # extract the final text answer from the response dict
        answer = response.get("output", "I could not generate a response.")

        # save this turn to memory so next message has context
        self._save_history(session_id, query, answer)

        # return a structured dict — the API route will use this to build the JSON response
        return {
            "answer": answer,
            "session_id": session_id,
        }

    def clear_session(self, session_id: str):
        # delete conversation history for a session (used when user clicks "New Chat")
        if session_id in self.sessions:
            del self.sessions[session_id]
