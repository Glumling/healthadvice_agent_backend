"""
tools_extra.py – adds search, REST fetch, calculator, Python REPL,
and PDF-QA tools. Works on BOTH new split-package LangChain (≥0.1.18)
and older monolithic installs.

Tip: to silence “legacy import” warnings entirely:
    pip install -U langchain langchain-community langchain-experimental langchain-openai
"""

from __future__ import annotations
import math, glob, warnings
from typing import List

# ── Flexible imports ───────────────────────────────────────────────
try:                                        # modern stack
    from langchain_core.tools import tool
    from langchain_community.utilities import SerpAPIWrapper
    from langchain_community.tools import RequestsGetTool, DuckDuckGoSearchRun
    from langchain_community.utilities.requests import TextRequestsWrapper
    from langchain_community.document_loaders import UnstructuredPDFLoader
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings
    from langchain_experimental.tools.python.tool import PythonREPLTool
    try:
        from langchain_community.tools.calculator import Calculator
    except ModuleNotFoundError:
        Calculator = None
except ModuleNotFoundError:                 # legacy ≤0.1.16
    warnings.warn("Using legacy langchain imports (consider upgrading).")
    from langchain.tools import tool                         # type: ignore
    from langchain.utilities import SerpAPIWrapper           # type: ignore
    from langchain.tools import RequestsGetTool              # type: ignore
    from langchain.utilities.requests import TextRequestsWrapper  # type: ignore
    from langchain.tools import DuckDuckGoSearchRun          # type: ignore
    from langchain.document_loaders import UnstructuredPDFLoader  # type: ignore
    from langchain.vectorstores import FAISS                 # type: ignore
    from langchain.embeddings.openai import OpenAIEmbeddings # type: ignore
    try:
        from langchain_experimental.tools.python.tool import PythonREPLTool
    except ModuleNotFoundError:
        from langchain.utilities import PythonREPL as PythonREPLTool   # type: ignore
    try:
        from langchain.tools import Calculator                         # type: ignore
    except ImportError:
        Calculator = None

# ── 1) Web search wrapper (SerpAPI → fallback DuckDuckGo) ─────────
#    OpenAI-function agents can only accept Callables or BaseTool
#    objects with a valid schema. We therefore wrap the underlying
#    search run() method in our own @tool-decorated function.

try:                       # prefer SerpAPI if key is set
    _raw_search = SerpAPIWrapper()          # needs SERPAPI_API_KEY
except Exception:
    from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
    from langchain_community.tools import DuckDuckGoSearchRun
    _raw_search = DuckDuckGoSearchRun(
        api_wrapper=DuckDuckGoSearchAPIWrapper()
    )

from langchain_core.tools import tool as _lc_tool

@_lc_tool
def web_search(query: str) -> str:          # <- new tool the agent will see
    """Search the web and return the first result snippet."""
    try:
        return _raw_search.run(query)
    except Exception as e:                  # noqa: BLE001
        return f"Search error: {e}"

# ── 2) Generic REST/JSON GET tool ─────────────────────────────────
# ── 2) Generic REST/JSON GET tool ─────────────────────────────────
try:
    from langchain_community.utilities.requests import TextRequestsWrapper
except ModuleNotFoundError:
    from langchain.utilities.requests import TextRequestsWrapper          # legacy

api_get_tool = RequestsGetTool(            # ← this variable MUST exist
    requests_wrapper=TextRequestsWrapper(),
    allow_dangerous_requests=True,
)

# ── 3) Calculator tool (built-in or custom safe-eval) ──────────────
if Calculator is not None:                  # new stack
    calc_tool = Calculator()
else:                                       # fallback
    @tool
    def calc_tool(expression: str) -> str:
        """Evaluate math expressions (sin, sqrt, etc.)."""
        try:
            return str(eval(expression, {"__builtins__": {}}, vars(math)))
        except Exception as e:              # noqa: BLE001
            return f"Error: {e}"

# ── 4) Python REPL sandbox ─────────────────────────────────────────
python_repl_tool = PythonREPLTool()

# ── 5) Local PDF Q-A retriever (lazy load) ─────────────────────────
def _build_pdf_retriever():
    pdfs = glob.glob("docs/*.pdf")
    if not pdfs:
        return None
    docs = []
    for path in pdfs:
        docs += UnstructuredPDFLoader(path).load()
    vect = FAISS.from_documents(docs, OpenAIEmbeddings())
    return vect.as_retriever(search_kwargs={"k": 3})

_PDF_RET = _build_pdf_retriever()

@tool
def docs_qa(query: str) -> str:
    """Answer questions from PDFs in ./docs (top-3 snippets)."""
    if _PDF_RET is None:
        return "No PDFs found in ./docs."
    chunks: List = _PDF_RET.invoke(query)   # type: ignore[arg-type]
    return "\n---\n".join(c.page_content[:500] for c in chunks)
