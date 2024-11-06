## **Section 1: RAG, LlamaIndex, and Vector Storage**

### **What is the RAG (Retrieval-Augmented Generation)?**

- RAG (Retrieval-Augmented Generation) : Integrates the retrieval (searching) into LLM text generation. RAG helps the model to “look up” external information to improve its responses. [cite](https://towardsdatascience.com/rag-vs-finetuning-which-is-the-best-tool-to-boost-your-llm-application-94654b1eaba7) [25 Aug 2023]

  <!-- <img src="../files/RAG.png" alt="sk" width="400"/> -->

- In a 2020 paper, Meta (Facebook) came up with a framework called retrieval-augmented generation to give LLMs access to information beyond their training data. [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401): [[cnt](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=arxiv%3A+2005.11401)] [22 May 2020]

  1. RAG-sequence — We retrieve k documents, and use them to generate all the output tokens that answer a user query.
  1. RAG-token— We retrieve k documents, use them to generate the next token, then retrieve k more documents, use them to generate the next token, and so on. This means that we could end up retrieving several different sets of documents in the generation of a single answer to a user’s query.
  1. Of the two approaches proposed in the paper, the RAG-sequence implementation is pretty much always used in the industry. It’s cheaper and simpler to run than the alternative, and it produces great results. [cite](https://towardsdatascience.com/add-your-own-data-to-an-llm-using-retrieval-augmented-generation-rag-b1958bf56a5a) [30 Sep 2023]

### **Retrieval-Augmented Generation: Research Papers**

- [RAG for LLMs](https://arxiv.org/abs/2312.10997): [[cnt](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=arxiv%3A+2312.10997)] 🏆Retrieval-Augmented Generation for Large Language Models: A Survey: `Three paradigms of RAG Naive RAG > Advanced RAG > Modular RAG`

- <details open>

  <summary>Expand: Research Papers</summary>

  - [Benchmarking Large Language Models in Retrieval-Augmented Generation](https://arxiv.org/abs/2309.01431): [[cnt](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=arxiv%3A+2309.01431)]: Retrieval-Augmented Generation Benchmark (RGB) is proposed to assess LLMs on 4 key abilities [4 Sep 2023]:
    - <details open>
      <summary>Expand</summary>
      1. Noise robustness (External documents contain noises, struggled with noise above 80%)
      1. Negative rejection (External documents are all noises, Highest rejection rate was only 45%)
      1. Information integration (Difficulty in summarizing across multiple documents, Highest accuracy was 60-67%)
      1. Counterfactual robustness (Failed to detect factual errors in counterfactual external documents.)
      </details>
  - [Active Retrieval Augmented Generation](https://arxiv.org/abs/2305.06983) : [[cnt](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=arxiv%3A+2305.06983)]: Forward-Looking Active REtrieval augmented generation (FLARE): FLARE iteratively generates a temporary next sentence and check whether it contains low-probability tokens. If so, the system retrieves relevant documents and regenerates the sentence. Determine low-probability tokens by `token_logprobs in OpenAI API response`. [git](https://github.com/jzbjyb/FLARE/blob/main/src/templates.py) [11 May 2023]
  - [Hyde](https://arxiv.org/abs/2212.10496): Hypothetical Document Embeddings. `zero-shot (generate a hypothetical document) -> embedding -> avg vectors -> retrieval` [20 Dec 2022]
  - [Self-RAG](https://arxiv.org/pdf/2310.11511.pdf): [[cnt](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=arxiv%3A+2310.11511)] 1. `Critic model C`: Generates reflection tokens (IsREL (relevant,irrelevant), IsSUP (fullysupported,partially supported,nosupport), IsUse (is useful: 5,4,3,2,1)). It is pretrained on data labeled by GPT-4. 2. `Generator model M`: The main language model that generates task outputs and reflection tokens. It leverages the data labeled by the critic model during training. 3. `Retriever model R`: Retrieves relevant passages. The LM decides if external passages (retriever) are needed for text generation. [git](https://github.com/AkariAsai/self-rag) [17 Oct 2023]
  - [A Survey on Retrieval-Augmented Text Generation](https://arxiv.org/abs/2202.01110): [[cnt](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=arxiv%3A+2202.01110)]: This paper conducts a survey on retrieval-augmented text generation, highlighting its advantages and state-of-the-art performance in many NLP tasks. These tasks include Dialogue response generation, Machine translation, Summarization, Paraphrase generation, Text style transfer, and Data-to-text generation. [2 Feb 2022]
  - [Retrieval meets Long Context LLMs](https://arxiv.org/abs/2310.03025): [[cnt](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=arxiv%3A+2310.03025)]: We demonstrate that retrieval-augmentation significantly improves the performance of 4K context LLMs. Perhaps surprisingly, we find this simple retrieval-augmented baseline can perform comparable to 16K long context LLMs. [4 Oct 2023]
  - [FreshLLMs](https://arxiv.org/abs/2310.03214): [[cnt](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=arxiv%3A+2310.03214)]: Fresh Prompt, Google search first, then use results in prompt. Our experiments show that FreshPrompt outperforms both competing search engine-augmented prompting methods such as Self-Ask (Press et al., 2022) as well as commercial systems such as Perplexity.AI. [git](https://www.github.com/freshllms/freshqa) [5 Oct 2023]
  - [RECOMP: Improving Retrieval-Augmented LMs with Compressors](https://arxiv.org/abs/2310.04408): [[cnt](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=arxiv%3A+2310.04408)]: 1. We propose RECOMP (Retrieve, Compress, Prepend), an intermediate step which compresses retrieved documents into a textual summary prior to prepending them to improve retrieval-augmented language models (RALMs). 2. We present two compressors – an `extractive compressor` which selects useful sentences from retrieved documents and an `abstractive compressor` which generates summaries by synthesizing information from multiple documents. 3. Both compressors are trained. [6 Oct 2023]
  - [Retrieval-Augmentation for Long-form Question Answering](https://arxiv.org/abs/2310.12150): [[cnt](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=arxiv%3A+2310.12150)]: 1. The order of evidence documents affects the order of generated answers 2. the last sentence of the answer is more likely to be unsupported by evidence. 3. Automatic methods for detecting attribution can achieve reasonable performance, but still lag behind human agreement. `Attribution in the paper assesses how well answers are based on provided evidence and avoid creating non-existent information.` [18 Oct 2023]
  - [INTERS: Unlocking the Power of Large Language Models in Search with Instruction Tuning](https://arxiv.org/abs/2401.06532): INTERS covers 21 search tasks across three categories: query understanding, document understanding, and query-document relationship understanding. The dataset is designed for instruction tuning, a method that fine-tunes LLMs on natural language instructions. [git](https://github.com/DaoD/INTERS) [12 Jan 2024]
  - [RAG vs Fine-tuning](https://arxiv.org/abs/2401.08406): Pipelines, Tradeoffs, and a Case Study on Agriculture. [16 Jan 2024]
  - [The Power of Noise: Redefining Retrieval for RAG Systems](https://arxiv.org/abs/2401.14887): No more than 2-5 relevant docs + some amount of random noise to the LLM context maximizes the accuracy of the RAG. [26 Jan 2024]
  - [Corrective Retrieval Augmented Generation (CRAG)](https://arxiv.org/abs/2401.15884): Retrieval Evaluator assesses the retrieved documents and categorizes them as Correct, Ambiguous, or Incorrect. For Ambiguous and Incorrect documents, the method uses Web Search to improve the quality of the information. The refined and distilled documents are then used to generate the final output. [29 Jan 2024] CRAG implementation by LangGraph [git](https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_crag.ipynb)
  - [RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval](https://arxiv.org/abs/2401.18059): Introduce a novel approach to retrieval-augmented language models by constructing a recursive tree structure from documents. [git](https://github.com/run-llama/llama_index/blob/main/llama-index-packs/llama-index-packs-raptor/README.md) `pip install llama-index-packs-raptor` / [git](https://github.com/profintegra/raptor-rag) [31 Jan 2024]
  - [CRAG: Comprehensive RAG Benchmark](https://arxiv.org/abs/2406.04744): a factual question answering benchmark of 4,409 question-answer pairs and mock APIs to simulate web and Knowledge Graph (KG) search [ref](https://www.aicrowd.com/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024) [7 Jun 2024]
  - [PlanRAG](https://arxiv.org/abs/2406.12430): Decision Making. Decision QA benchmark, DQA. Plan -> Retrieve -> Make a decision (PlanRAG) [git](https://github.com/myeon9h/PlanRAG) [18 Jun 2024]
  - [Searching for Best Practices in Retrieval-Augmented Generation](https://arxiv.org/abs/2407.01219): `Best Performance Practice`: Query Classification, Hybrid with HyDE (retrieval), monoT5 (reranking), Reverse (repacking), Recomp (summarization). `Balanced Efficiency Practice`: Query Classification, Hybrid (retrieval), TILDEv2 (reranking), Reverse (repacking), Recomp (summarization). [1 Jul 2024]
  - [Retrieval Augmented Generation or Long-Context LLMs?](https://arxiv.org/abs/2407.16833): Long-Context consistently outperforms RAG in terms of average performance. However, RAG's significantly lower cost remains a distinct advantage. [23 Jul 2024]
  - [Graph Retrieval-Augmented Generation: A Survey](https://arxiv.org/abs/2408.08921) [15 Aug 2024]
  - [Adaptive-RAG](https://arxiv.org/abs/2403.14403): Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity [git](https://github.com/starsuzi/Adaptive-RAG) [21 Mar 2024]
  - [OP-RAG: Order-preserve RAG](https://arxiv.org/abs/2409.01666): Unlike traditional RAG, which sorts retrieved chunks by relevance, we keep them in their original order from the text.  [3 Sep 2024]
  - [Retrieval Augmented Generation (RAG) and Beyond](https://arxiv.org/abs/2409.14924):🏆The paper classifies user queries into four levels—`explicit, implicit, interpretable rationale, and hidden rationale`—and highlights the need for external data integration and fine-tuning LLMs for specialized tasks. [23 Sep 2024]
  - [Astute RAG](https://arxiv.org/abs/2410.07176): adaptively extracts essential information from LLMs, consolidates internal and external knowledge with source awareness, and finalizes answers based on reliability. [9 Oct 2024]

  </details>

### **Advanced RAG**

- RAG Pipeline
  1. Indexing Stage: Preparing a knowledge base.
  2. Querying Stage: Querying the indexed data to retrieve relevant information.
  3. Responding Stage: Generating responses based on the retrieved information. [ref](https://learn.microsoft.com/en-us/azure/machine-learning/concept-retrieval-augmented-generation)
- How to optimize RAG pipeline: [Indexing optimization](https://newsletter.theaiedge.io/p/how-to-optimize-your-rag-pipelines) [24 Oct 2023]
- Advanced RAG Patterns: How to improve RAG peformance [ref](https://cloudatlas.me/why-do-rag-pipelines-fail-advanced-rag-patterns-part1-841faad8b3c2) / [ref](https://cloudatlas.me/how-to-improve-rag-peformance-advanced-rag-patterns-part2-0c84e2df66e6) [17 Oct 2023]
  1. Data quality: Clean, standardize, deduplicate, segment, annotate, augment, and update data to make it clear, consistent, and context-rich.
  2. Embeddings fine-tuning: Fine-tune embeddings to domain specifics, adjust them according to context, and refresh them periodically to capture evolving semantics.
  3. Retrieval optimization: Refine chunking, embed metadata, use query routing, multi-vector retrieval, re-ranking, hybrid search, recursive retrieval, query engine, [HyDE](https://arxiv.org/abs/2212.10496) [20 Dec 2022], and vector search algorithms to improve retrieval efficiency and relevance.
  4. Synthesis techniques: Query transformations, prompt templating, prompt conditioning, function calling, and fine-tuning the generator to refine the generation step.
  - HyDE: Implemented in [LangChain: HypotheticalDocumentEmbedder](https://github.com/langchain-ai/langchain/blob/master/cookbook/hypothetical_document_embeddings.ipynb). A query generates hypothetical documents, which are then embedded and retrieved to provide the most relevant results. `query -> generate n hypothetical documents -> documents embedding - (avg of embeddings) -> retrieve -> final result.` [ref](https://www.jiang.jp/posts/20230510_hyde_detailed/index.html)
- Demystifying Advanced RAG Pipelines: An LLM-powered advanced RAG pipeline built from scratch [git](https://github.com/pchunduri6/rag-demystified) [19 Oct 2023]
- [9 Effective Techniques To Boost Retrieval Augmented Generation (RAG) Systems](https://towardsdatascience.com/9-effective-techniques-to-boost-retrieval-augmented-generation-rag-systems-210ace375049) [doc](9-effective-rag-techniques.png): ReRank, Prompt Compression, Hypothetical Document Embedding (HyDE), Query Rewrite and Expansion, Enhance Data Quality, Optimize Index Structure, Add Metadata, Align Query with Documents, Mixed Retrieval (Hybrid Search) [2 Jan 2024]
- [cite](https://twitter.com/yi_ding/status/1721728060876300461) [7 Nov 2023] `OpenAI has put together a pretty good roadmap for building a production RAG system.` Naive RAG -> Tune Chunks -> Rerank & Classify -> Prompt Engineering. In `llama_index`... [Youtube](https://www.youtube.com/watch?v=ahnGLM-RC1Y)  <br/>
  <img src="../files/oai-rag-success-story.jpg" width="500">
- [Graph RAG (by NebulaGraph)](https://medium.com/@nebulagraph/graph-rag-the-new-llm-stack-with-knowledge-graphs-e1e902c504ed): NebulaGraph proposes the concept of Graph RAG, which is a retrieval enhancement technique based on knowledge graphs. [demo](https://www.nebula-graph.io/demo) [8 Sep 2023]
- [Evaluation with Ragas](https://towardsdatascience.com/visualize-your-rag-data-evaluate-your-retrieval-augmented-generation-system-with-ragas-fc2486308557): UMAP (often used to reduce the dimensionality of embeddings) with Ragas metrics for visualizing RAG results. [Mar 2024] / `Ragas provides metrics`: Context Precision, Context Relevancy, Context Recall, Faithfulness, Answer Relevance, Answer Semantic Similarity, Answer Correctness, Aspect Critique [git](https://github.com/explodinggradients/ragas) [May 2023]
- [Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval): Contextual Retrieval enhances traditional RAG by using Contextual Embeddings and Contextual BM25 to maintain context during retrieval. [19 Sep 2024]

### **The Problem with RAG**

- The Problem with RAG
  1. A question is not semantically similar to its answers. Cosine similarity may favor semantically similar texts that do not contain the answer.
  1. Semantic similarity gets diluted if the document is too long. Cosine similarity may favor short documents with only the relevant information.
  1. The information needs to be contained in one or a few documents. Information that requires aggregations by scanning the whole data.
- [Seven Failure Points When Engineering a Retrieval Augmented Generation System](https://arxiv.org/abs/2401.05856): 1. Missing Content, 2. Missed the Top Ranked Documents, 3. Not in Context, 4. Not Extracted, 5. Wrong Format, 6. Incorrect Specificity, 7. Lack of Thorough Testing [11 Jan 2024]
- Solving the core challenges of Retrieval-Augmented Generation [ref](https://towardsdatascience.com/12-rag-pain-points-and-proposed-solutions-43709939a28c) [Feb 2024] <br/>
  <img src="../files/rag-12-pain-points-solutions.jpg" width="500">

### **RAG Solution Design & Application**

- RAG Solution Design
  - [Azure: Designing and developing a RAG solution](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-solution-design-and-evaluation-guide)
    - [Announcing cost-effective RAG at scale with Azure AI Search](https://aka.ms/AAqfqla)
    - [Advanced RAG with Azure AI Search and LlamaIndex](https://techcommunity.microsoft.com/t5/ai-azure-ai-services-blog/advanced-rag-with-azure-ai-search-and-llamaindex/ba-p/4115007)
    - [GPT-RAG](https://github.com/Azure/GPT-RAG): Enterprise RAG Solution Accelerator [Jun 2023]
    - [Azure OpenAI chat baseline architecture in an Azure landing zone](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/architecture/azure-openai-baseline-landing-zone)
    - Azure Reference Architectures: [x-ref](aoai.md/#azure-reference-architectures)
  - [RAG at scale](https://medium.com/@neum_ai/retrieval-augmented-generation-at-scale-building-a-distributed-system-for-synchronizing-and-eaa29162521) [28 Sep 2023]
  - [LangChain RAG from scratch](https://github.com/langchain-ai/rag-from-scratch) [Jan 2024]
  - [LlamIndex Building Performant RAG Applications for Production](https://docs.llamaindex.ai/en/stable/optimizing/production_rag/#building-performant-rag-applications-for-production)
  - [Advanced RAG on Hugging Face documentation using LangChain](https://huggingface.co/learn/cookbook/advanced_rag)
  - RAG context relevancy metric: Ragas, TruLens, DeepEval [ref](https://towardsdatascience.com/the-challenges-of-retrieving-and-evaluating-relevant-context-for-rag-e362f6eaed34) [Jun 2024]
    - `Context Relevancy (in Ragas) = S / Total number of sentences in retrieved context`
    - `Contextual Relevancy (in DeepEval) = Number of Relevant Statements / Total Number of Statements`
  - [Papers with code](https://paperswithcode.com/method/rag): RAG
  - [What AI Engineers Should Know about Search](https://softwaredoug.com/blog/2024/06/25/what-ai-engineers-need-to-know-search) [25 Jun 2024]
  - [GraphRAG (by Microsoft)](https://arxiv.org/abs/2404.16130): Original Documents -> Knowledge Graph (Group Summaries) -> Partial Responses -> Final Response. local and global search.
[ref](https://microsoft.github.io/graphrag) [git](https://github.com/microsoft/graphrag) [24 Apr 2024]
    - [GraphRAG Implementation with LlamaIndex](https://github.com/run-llama/llama_index/blob/main/docs/docs/examples/cookbooks/GraphRAG_v1.ipynb) [15 Jul 2024]
    - ["From Local to Global" GraphRAG with Neo4j and LangChain](https://neo4j.com/developer-blog/global-graphrag-neo4j-langchain/) [09 Jul 2024]
    - [LightRAG](https://github.com/HKUDS/LightRAG): Utilizing graph structures for text indexing and retrieval processes. [8 Oct 2024]
  - [Learn RAG with LangChain](https://www.sakunaharinda.xyz/ragatouille-book): Online book [May 2024]
  - [Advanced RAG Techniques](https://github.com/NirDiamant/RAG_Techniques):🏆Showcases various advanced techniques for Retrieval-Augmented Generation (RAG) [Jul 2024]
  - [A Practical Approach to Retrieval Augmented Generation (RAG) Systems](https://github.com/mallahyari/rag-ebook): Online book [Dec 2023]
  - [Galileo eBook](https://www.rungalileo.io/mastering-rag): 200 pages content. Mastering RAG. [doc](../files/archive/Mastering%20RAG-compressed.pdf) [Sep 2024]
  - [Introduction to Large-Scale Similarity Search: HNSW, IVF, LSH](https://blog.gopenai.com/introduction-to-large-scale-similarity-search-part-one-hnsw-ivf-lsh-677bf193ab07) [28 Sep 2024]
  - [5 Chunking Strategies For RAG](https://blog.dailydoseofds.com/p/5-chunking-strategies-for-rag) [19 Oct 2024]
  - [Genie: Uber’s Gen AI On-Call Copilot](https://www.uber.com/blog/genie-ubers-gen-ai-on-call-copilot/) [10 Oct 2024]
- RAG Application / Framework
  - [RAG capabilities of LlamaIndex to QA about SEC 10-K & 10-Q documents](https://github.com/run-llama/sec-insights): A real world full-stack application using LlamaIndex [Sep 2023]
  - [RAGxplorer](https://github.com/gabrielchua/RAGxplorer): Visualizing document chunks and the queries in the embedding space. [Jan 2024]
  - [PrivateGPT](https://github.com/imartinez/privateGPT): 100% privately, no data leaks 1. The API is built using FastAPI and follows OpenAI's API scheme. 2. The RAG pipeline is based on LlamaIndex. [May 2023]
  - [Danswer](https://github.com/danswer-ai/danswer): Ask Questions in natural language and get Answers backed by private sources: Slack, GitHub, Confluence, etc. [Apr 2023]
  - [Verba](verba.weaviate.io) Retrieval Augmented Generation (RAG) chatbot powered by Weaviate [git](https://github.com/weaviate/Verba) [Jul 2023]
  - [llm-answer-engine](https://github.com/developersdigest/llm-answer-engine): Build a Perplexity-Inspired Answer Engine Using Next.js, Groq, Mixtral, LangChain, OpenAI, Brave & Serper [Mar 2024]
  - [turboseek](https://github.com/Nutlope/turboseek): An AI search engine inspired by Perplexity [May 2024]
  - [quivr](https://github.com/QuivrHQ/quivr): A personal productivity assistant (RAG). Chat with your docs (PDF, CSV, ...) [May 2023]
  - [RAGApp](https://github.com/ragapp): Agentic RAG. custom GPTs, but deployable in your own cloud infrastructure using Docker. [Apr 2024]
  - [Cognita](https://github.com/truefoundry/cognita): RAG (Retrieval Augmented Generation) Framework for building modular, open source applications [Jul 2023]
  - Open Source AI Searches: [Perplexica](https://github.com/ItzCrazyKns/Perplexica): Open source alternative to Perplexity AI [Apr 2024] / [Marqo](https://github.com/marqo-ai/marqo) / [txtai](https://github.com/neuml/txtai) / [Typesense](https://github.com/typesense/typesense)  / [Morphic](https://github.com/miurla/morphic)
  - [AutoRAG](https://github.com/Marker-Inc-Korea/AutoRAG): RAG AutoML tool for automatically finds an optimal RAG pipeline for your data. [Jan 2024]
  - [RAGflow](https://github.com/infiniflow/ragflow): Streamlined RAG workflow. Focusing on Deep document understanding [Dec 2023]
  - [MindSearch](https://github.com/InternLM/MindSearch): an open-source AI Search Engine Framework [Jul 2024]
  - [RAGFoundry](https://github.com/IntelLabs/RAGFoundry): a library designed to improve LLMs ability to use external information by fine-tuning models on specially created RAG-augmented datasets. [5 Aug 2024]
  - [Haystack](https://github.com/deepset-ai/haystack): LLM orchestration framework to build customizable, production-ready LLM applications. [5 May 2020]
  - [RAGChecker](https://arxiv.org/abs/2408.08067): A Fine-grained Framework For Diagnosing RAG [git](https://github.com/amazon-science/RAGChecker) [15 Aug 2024]
  - [HybridRAG](https://arxiv.org/abs/2408.04948): Integrating VectorRAG and GraphRAG with financial earnings call transcripts in Q&A format. [9 Aug 2024]
  - [MedGraphRAG](https://arxiv.org/abs/2408.04187): MedGraphRAG outperforms the previous SOTA model, [Medprompt](https://arxiv.org/abs/2311.16452), by 1.1%. [git](https://github.com/medicinetoken/medical-graph-rag) [8 Aug 2024]
  - [STORM](https://github.com/stanford-oval/storm): Wikipedia-like articles from scratch based on Internet search. [Mar 2024]
  - [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG): A Python Toolkit for Efficient RAG Research [Mar 2024]
  - [Canopy](https://github.com/pinecone-io/canopy): open-source RAG framework and context engine built on top of the Pinecone vector database. [Aug 2023]
  - [kotaemon](https://github.com/Cinnamon/kotaemon): open-source clean & customizable RAG UI for chatting with your documents. [Mar 2024]
  - [PaperQA2](https://github.com/Future-House/paper-qa): High accuracy RAG for answering questions from scientific documents with citations [Feb 2023]
  - [RAG Builder](https://github.com/KruxAI/ragbuilder): Automatically create an optimal production-ready Retrieval-Augmented Generation (RAG) setup for your data. [Jun 2024]
  - [SWIRL AI Connect](https://github.com/swirlai/swirl-search): SWIRL AI Connect enables you to perform Unified Search and bring in a secure AI Co-Pilot. [Apr 2022]
- Applications, Frameworks, and User Interface (UI/UX): [x-ref](agent_app.md/#applications-frameworks-and-user-interface-uiux)

### **LlamaIndex**

- LlamaIndex (formerly GPT Index) is a data framework for LLM applications to ingest, structure, and access private or domain-specific data. The high-level API allows users to ingest and query their data in a few lines of code. High-Level Concept: [ref](https://docs.llamaindex.ai/en/latest/getting_started/concepts.html) / doc:[ref](https://gpt-index.readthedocs.io/en/latest/index.html) / blog:[ref](https://www.llamaindex.ai/blog) / [git](https://github.com/run-llama/llama_index) [Nov 2022]

  > Fun fact this core idea was the initial inspiration for GPT Index (the former name of LlamaIndex) 11/8/2022 - almost a year ago!. [cite](https://twitter.com/jerryjliu0/status/1711817419592008037) / [Walking Down the Memory Maze: Beyond Context Limit through Interactive Reading](https://arxiv.org/abs/2310.05029)
  >
  > 1.  Build a data structure (memory tree)
  > 1.  Transverse it via LLM prompting

- LlamaIndex Toolkits: `LlamaHub`: A library of data loaders for LLMs [git](https://github.com/run-llama/llama-hub) [Feb 2023] / `LlamaIndex CLI`: a command line tool to generate LlamaIndex apps [ref](https://llama-2.ai/llamaindex-cli/) [Nov 2023] / `LlamaParse`: A unique parsing tool for intricate documents [git](https://github.com/run-llama/llama_parse) [Feb 2024]

  <details open>
    <summary>High-Level Concepts</summary>

    - Query engine vs Chat engine

      1. The query engine wraps a `retriever` and a `response synthesizer` into a pipeline, that will use the query string to fetch nodes (sentences or paragraphs) from the index and then send them to the LLM (Language and Logic Model) to generate a response
      1. The chat engine is a quick and simple way to chat with the data in your index. It uses a `context manager` to keep track of the conversation history and generate relevant queries for the retriever. Conceptually, it is a `stateful` analogy of a Query Engine.

    - Storage Context vs Settings (p.k.a. Service Context)

      - Both the Storage Context and Service Context are data classes.

      <!-- ```python
      index = load_index_from_storage(storage_context, service_context=service_context)
      ``` -->

      1. Introduced in v0.10.0, ServiceContext is replaced to Settings object.
      1. Storage Context is responsible for the storage and retrieval of data in Llama Index, while the Service Context helps in incorporating external context to enhance the search experience.
      1. The Service Context is not directly involved in the storage or retrieval of data, but it helps in providing a more context-aware and accurate search experience.

     ```python
     # The storage context container is a utility container for storing nodes, indices, and vectors.
     class StorageContext:
       docstore: BaseDocumentStore
       index_store: BaseIndexStore
       vector_store: VectorStore
       graph_store: GraphStore
     ```

     ```python
     # The service context container is a utility container for LlamaIndex index and query classes.
     class ServiceContext:
       llm_predictor: BaseLLMPredictor
       prompt_helper: PromptHelper
       embed_model: BaseEmbedding
       node_parser: NodeParser
       llama_logger: LlamaLogger
       callback_manager: CallbackManager
     ```

    ```python
    @dataclass
    class _Settings:
      # lazy initialization
      _llm: Optional[LLM] = None
      _embed_model: Optional[BaseEmbedding] = None
      _callback_manager: Optional[CallbackManager] = None
      _tokenizer: Optional[Callable[[str], List[Any]]] = None
      _node_parser: Optional[NodeParser] = None
      _prompt_helper: Optional[PromptHelper] = None
      _transformations: Optional[List[TransformComponent]] = None
    ```

     </details>

- [LlamaIndex Overview (Japanese)](https://dev.classmethod.jp/articles/llamaindex-tutorial-001-overview-v0-7-9/) [17 Jul 2023]
- [LlamaIndex Tutorial](https://nanonets.com/blog/llamaindex/): A Complete LlamaIndex Guide [18 Oct 2023]
<!-- - [CallbackManager (Japanese)](https://dev.classmethod.jp/articles/llamaindex-tutorial-003-callback-manager/) [27 May 2023] / [Customize TokenTextSplitter (Japanese)](https://dev.classmethod.jp/articles/llamaindex-tutorial-002-text-splitter/) [27 May 2023] / --> 
- [Chat engine ReAct mode](https://gpt-index.readthedocs.io/en/stable/examples/chat_engine/chat_engine_react.html), [FLARE Query engine](https://docs.llamaindex.ai/en/stable/examples/query_engine/flare_query_engine.html) 
- Multimodal RAG Pipeline [ref](https://blog.llamaindex.ai/multi-modal-rag-621de7525fea) [Nov 2023]
- From Simple to Advanced RAG [ref](https://twitter.com/jerryjliu0/status/1711419232314065288) / [ref](https://aiconference.com/speakers/jerry-liu-2023/) [10 Oct 2023] <br/>
  <img src="../files/advanced-rag.png" width="430">
- [Building and Productionizing RAG](https://docs.google.com/presentation/d/1rFQ0hPyYja3HKRdGEgjeDxr0MSE8wiQ2iu4mDtwR6fc/edit#slide=id.p): [doc](../files/archive/LlamaIndexTalk_PyDataGlobal.pdf): Optimizing RAG Systems 1. Table Stakes 2. Advanced Retrieval: Small-to-Big 3. Agents 4. Fine-Tuning 5. Evaluation [Nov 2023]
- [A Cheat Sheet and Some Recipes For Building Advanced RAG](https://blog.llamaindex.ai/a-cheat-sheet-and-some-recipes-for-building-advanced-rag-803a9d94c41b) RAG cheat sheet shared above was inspired by [RAG survey paper](https://arxiv.org/abs/2312.10997). [doc](../files/advanced-rag-diagram-llama-index.png) [Jan 2024]
- [Fine-Tuning a Linear Adapter for Any Embedding Model](https://medium.com/llamaindex-blog/fine-tuning-a-linear-adapter-for-any-embedding-model-8dd0a142d383): Fine-tuning the embeddings model requires you to reindex your documents. With this approach, you do not need to re-embed your documents. Simply transform the query instead. [7 Sep 2023]
- 4 RAG techniques implemented in `llama_index` / [cite](https://x.com/ecardenas300/status/1704188276565795079) [20 Sep 2023] / [git](https://github.com/weaviate/recipes)
<details open>
<summary>Expand: 4 RAG techniques</summary>

  1. SQL Router Query Engine: Query router that can reference your vector database or SQL database

  2. Sub Question Query Engine: Break down the complex question into sub-questions

  3. Recursive Retriever + Query Engine: Reference node relationships, rather than only finding a node (chunk) that is most relevant.
  
  4. Self Correcting Query Engines: Use an LLM to evaluate its own output.
</details>

### **Vector Database Comparison**

- [Not All Vector Databases Are Made Equal](https://towardsdatascience.com/milvus-pinecone-vespa-weaviate-vald-gsi-what-unites-these-buzz-words-and-what-makes-each-9c65a3bd0696): Printed version for "Medium" limits. [doc](../files/vector-dbs.pdf) [2 Oct 2021]
- [Faiss](https://faiss.ai/): Facebook AI Similarity Search (Faiss) is a library for efficient similarity search and clustering of dense vectors. It is used as an alternative to a vector database in the development and library of algorithms for a vector database. It is developed by Facebook AI Research. [git](https://github.com/facebookresearch/faiss) [Feb 2017]
- Milvus (A cloud-native vector database) Embedded [git](https://github.com/milvus-io/milvus) [Sep 2019]: Alternative option to replace PineCone and Redis Search in OSS. It offers support for multiple languages, addresses the limitations of RedisSearch, and provides cloud scalability and high reliability with Kubernetes.
- [Pinecone](https://docs.pinecone.io): A fully managed cloud Vector Database. Commercial Product [Jan 2021]
- [Weaviate](https://github.com/weaviate/weaviate): Store both vectors and data objects. [Jan 2021]
- [Chroma](https://github.com/chroma-core/chroma): Open-source embedding database [Oct 2022]
- [Qdrant](https://github.com/qdrant/qdrant): Written in Rust. Qdrant (read: quadrant) [May 2020]
- [Redis extension for vector search, RedisVL](https://github.com/RedisVentures/redisvl): Redis Vector Library (RedisVL) [Nov 2022]
- [A SQLite extension for efficient vector search, based on Faiss!](https://github.com/asg017/sqlite-vss) [Jan 2023]
- [pgvector](https://github.com/pgvector/pgvector): Open-source vector similarity search for Postgres [Apr 2021] / [pgvectorscale](https://github.com/timescale/pgvectorscale): 75% cheaper than pinecone [Jul 2023]
- [lancedb](https://github.com/lancedb/lancedb): LanceDB's core is written in Rust and is built using Lance, an open-source columnar format.  [Feb 2023]
- [A Comprehensive Survey on Vector Database](https://arxiv.org/abs/2310.11703): Categorizes search algorithms by their approach, such as hash-based, tree-based, graph-based, and quantization-based. [18 Oct 2023]

#### **Vector Database Options for Azure**

- [Pgvector extension on Azure Cosmos DB for PostgreSQL](https://azure.microsoft.com/en-us/updates/generally-available-pgvector-extension-on-azure-cosmos-db-for-postgresql/): [ref](https://python.langchain.com/docs/modules/data_connection/vectorstores/integrations/pgvector) [13 Jun 2023]
- [Vector Search in Azure Cosmos DB for MongoDB vCore](https://devblogs.microsoft.com/cosmosdb/introducing-vector-search-in-azure-cosmos-db-for-mongodb-vcore/) [23 May 2023]
- [Vector search - Azure AI Search](https://github.com/Azure/azure-search-vector-samples): [ref](https://python.langchain.com/docs/modules/data_connection/vectorstores/integrations/azuresearch) Rebranded from Azure Cognitive Search [Oct 2019] to Azure AI Search [Nov 2023]
- [Azure Cache for Redis Enterprise](https://techcommunity.microsoft.com/t5/azure-developer-community-blog/introducing-vector-search-similarity-capabilities-in-azure-cache/ba-p/3827512): Enterprise [Redis Vector Search Demo](https://ecommerce.redisventures.com/) [22 May 2023 ]
- [Azure SQL's support for natively storing and querying vectors](https://devblogs.microsoft.com/azure-sql/announcing-eap-native-vector-support-in-azure-sql-database/) [21 May 2024]

**Note**: Azure Cache for Redis Enterprise: Enterprise Sku series are not able to deploy by a template such as Bicep and ARM.

[![Deploy to Azure](https://aka.ms/deploytoazurebutton)](https://portal.azure.com/#create/Microsoft.Template/uri/https%3A%2F%2Fraw.githubusercontent.com%2Fkimtth%2Fazure-openai-elastic-vector-langchain%2Fmain%2Finfra%2Fdeployment.json)

#### **Embedding**

- Azure Open AI Embedding API, `text-embedding-ada-002`, supports 1536 dimensions. Elastic search, Lucene based engine, supports 1024 dimensions as a max. Open search can insert 16,000 dimensions as a vector storage. Open search is available to use as a vector database with Azure Open AI Embedding API.
- OpenAI Embedding models: `text-embedding-3` [x-ref](chab.md/#openai-products) > `New embedding models`
- [text-embedding-ada-002](https://openai.com/blog/new-and-improved-embedding-model):
  Smaller embedding size. The new embeddings have only 1536 dimensions, one-eighth the size of davinci-001 embeddings,
  making the new embeddings more cost effective in working with vector databases. [15 Dec 2022]
- However, one exception to this is that the maximum dimension count for the Lucene engine is 1,024, compared with
  16,000 for the other engines. [ref](https://opensearch.org/docs/latest/search-plugins/knn/approximate-knn/)
<!-- - LlamaIndex `ElasticsearchReader` class:
  The name of the class in LlamaIndex is `ElasticsearchReader`. However, actually, it can only work with open search. -->
- [Vector Search with OpenAI Embeddings: Lucene Is All You Need](https://arxiv.org/abs/2308.14963): Our experiments were based on Lucene 9.5.0, but indexing was a bit tricky
  because the HNSW implementation in Lucene restricts vectors to 1024 dimensions, which was not sufficient for OpenAI’s 1536-dimensional embeddings. Although the resolution of this issue, which is to make vector dimensions configurable on a per codec basis, has been merged to the Lucene source trunk [git](https://github.com/apache/lucene/pull/12436), this feature has not been folded into a Lucene release (yet) as of early August 2023. [29 Aug 2023]
- [Is Cosine-Similarity of Embeddings Really About Similarity?](https://arxiv.org/abs/2403.05440): In linear matrix factorization, the use of regularization can impact, and in some cases, render cosine similarities meaningless. Regularization involves two objectives. The first objective applies L2-norm regularization to the product of matrices A and B, a process similar to dropout. The second objective applies L2-norm regularization to each individual matrix, similar to the weight decay technique used in deep learning. [8 Mar 2024]
- [Contextual Document Embedding (CDE)](https://arxiv.org/abs/2410.02525): Improve document retrieval by embedding both queries and documents within the context of the broader document corpus. [ref](https://pub.aimind.so/unlocking-the-power-of-contextual-document-embeddings-enhancing-search-relevance-01abfa814c76) [3 Oct 2024]
- [Fine-tuning Embeddings for Specific Domains](https://blog.gopenai.com/fine-tuning-embeddings-for-specific-domains-a-comprehensive-guide-5e4298b42185) [1 Oct 2024]