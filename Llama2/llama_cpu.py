from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate

MODEL_PATH = "llama-2-13b-chat.ggmlv3.q4_0.bin"

