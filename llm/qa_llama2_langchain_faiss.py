# !pip install accelerate bitsandbytes einops faiss-gpu
# !pip install langchain sentence-transformers transformers xformers

import langchain
import torch
import transformers


model_id = "meta-llama/Llama-2-7b-chat-hf"

device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"

# set quantization configuration to load large model with less GPU memory
# this requires the `bitsandbytes` library
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# begin initializing HF items, you need an access token
access_token = "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"
model_config = transformers.AutoConfig.from_pretrained(
    model_id, token=access_token,
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map="auto",
    token=access_token,
)

# enable evaluation mode to allow model inference
model.eval()

print(f"Model loaded on {device}")


tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id, token=access_token,
)

stop_list = ['\nHuman:', '\n```\n']

stop_token_ids = [tokenizer(x)['input_ids'] for x in stop_list]
print(stop_token_ids)

stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]
print(stop_token_ids)


# define custom stopping criteria object
class StopOnTokens(transformers.StoppingCriteria):
    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs
    ):
        for stop_ids in stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                return True
        return False


stopping_criteria = transformers.StoppingCriteriaList([StopOnTokens()])


generate_text = transformers.pipeline(
    model=model,
    tokenizer=tokenizer,
    return_full_text=True,
    task="text-generation",
    stopping_criteria=stopping_criteria,
    temperature=0.1,
    max_new_tokens=512,
    repetition_penalty=1.1,
)

res = generate_text(
    "Explain the difference between a database and a datalake."
)
print(res[0]["generated_text"])


llm = langchain.llms.HuggingFacePipeline(pipeline=generate_text)

# checking again that everything is working fine
print(llm(prompt="Explain the difference between a database and a datalake."))


web_links = [
    "https://www.databricks.com/",
    "https://help.databricks.com",
    "https://databricks.com/try-databricks",
    "https://help.databricks.com/s/",
    "https://docs.databricks.com",
    "https://kb.databricks.com/",
    "http://docs.databricks.com/getting-started/index.html",
    "http://docs.databricks.com/introduction/index.html",
    "http://docs.databricks.com/getting-started/tutorials/index.html",
    "http://docs.databricks.com/release-notes/index.html",
    "http://docs.databricks.com/ingestion/index.html",
    "http://docs.databricks.com/exploratory-data-analysis/index.html",
    "http://docs.databricks.com/data-preparation/index.html",
    "http://docs.databricks.com/data-sharing/index.html",
    "http://docs.databricks.com/marketplace/index.html",
    "http://docs.databricks.com/workspace-index.html",
    "http://docs.databricks.com/machine-learning/index.html",
    "http://docs.databricks.com/sql/index.html",
    "http://docs.databricks.com/delta/index.html",
    "http://docs.databricks.com/dev-tools/index.html",
    "http://docs.databricks.com/integrations/index.html",
    "http://docs.databricks.com/administration-guide/index.html",
    "http://docs.databricks.com/security/index.html",
    "http://docs.databricks.com/data-governance/index.html",
    "http://docs.databricks.com/lakehouse-architecture/index.html",
    "http://docs.databricks.com/reference/api.html",
    "http://docs.databricks.com/resources/index.html",
    "http://docs.databricks.com/whats-coming.html",
    "http://docs.databricks.com/archive/index.html",
    "http://docs.databricks.com/lakehouse/index.html",
    "http://docs.databricks.com/getting-started/quick-start.html",
    "http://docs.databricks.com/getting-started/etl-quick-start.html",
    "http://docs.databricks.com/getting-started/lakehouse-e2e.html",
    "http://docs.databricks.com/getting-started/free-training.html",
    "http://docs.databricks.com/sql/language-manual/index.html",
    "http://docs.databricks.com/error-messages/index.html",
    "http://www.apache.org/",
    "https://databricks.com/privacy-policy",
    "https://databricks.com/terms-of-use",
]

loader = langchain.document_loaders.WebBaseLoader(web_links)
documents = loader.load()

text_splitter = langchain.text_splitter.RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=20,
)
all_splits = text_splitter.split_documents(documents)

embeddings = langchain.embeddings.HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={"device": "cuda"},
)

# storing embeddings in the vector store
vectorstore = langchain.vectorstores.FAISS.from_documents(
    all_splits, embeddings,
)

chain = langchain.chains.ConversationalRetrievalChain.from_llm(
    llm, vectorstore.as_retriever(), return_source_documents=True,
)

chat_history = []

query = "What is data lakehouse architecture in Databricks?"
result = chain({"question": query, "chat_history": chat_history})

print(result["answer"])

chat_history = [(query, result["answer"])]

query = "What are data governance and interoperability in it?"
result = chain({"question": query, "chat_history": chat_history})

print(result["answer"])

print(result["source_documents"])
