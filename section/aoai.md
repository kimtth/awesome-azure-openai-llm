## **Section 2** : Azure OpenAI and Reference Architecture

### **Microsoft Azure OpenAI relevant LLM Framework**

#### LLM Integration Frameworks

1. [Semantic Kernel](https://devblogs.microsoft.com/semantic-kernel/) (Feb 2023): An open-source SDK for integrating AI services like OpenAI, Azure OpenAI, and Hugging Face with conventional programming languages such as C# and Python. It's an LLM orchestrator, similar to LangChain. / [git](https://github.com/microsoft/semantic-kernel) / [x-ref](sk_dspy.md/#semantic-kernel)
2. [Kernel Memory](https://github.com/microsoft/kernel-memory) (Jul 2023): An open-source service and plugin for efficient dataset indexing through custom continuous data hybrid pipelines.
3. [Azure ML Prompt Flow](https://learn.microsoft.com/en-us/azure/machine-learning/prompt-flow/overview-what-is-prompt-flow) (Jun 2023): A visual designer for prompt crafting using Jinja as a prompt template language. / [ref](https://techcommunity.microsoft.com/t5/ai-machine-learning-blog/harness-the-power-of-large-language-models-with-azure-machine/ba-p/3828459) / [git](https://github.com/microsoft/promptflow)

- A Memory in Semantic Kernel vs Kernel Memory (FKA. Semantic Memory (SM)): Kernel Memory is designed to efficiently handle large datasets and extended conversations. Deploying the memory pipeline as a separate service can be beneficial when dealing with large documents or long bot conversations. [ref](https://github.com/microsoft/chat-copilot/tree/main/memorypipeline)

#### Prompt Optimization

1. [Prompt Engine](https://github.com/microsoft/prompt-engine) (Jun 2022): A tool for crafting prompts for large language models in Python. / [Python](https://github.com/microsoft/prompt-engine-py)
2. [PromptBench](https://github.com/microsoft/promptbench) (Jun 2023): A unified evaluation framework for large language models.
3. [SAMMO](https://github.com/microsoft/sammo) (Apr 2024): A general-purpose framework for prompt optimization. / [ref](https://www.microsoft.com/en-us/research/blog/sammo-a-general-purpose-framework-for-prompt-optimization/)
4. [Prompty](https://github.com/microsoft/prompty) (Apr 2024): A template language for integrating prompts with LLMs and frameworks, enhancing prompt management and evaluation.
5. [guidance](https://github.com/microsoft/guidance) (Nov 2022): A domain-specific language (DSL) for controlling large language models, focusing on model interaction and implementing the "Chain of Thought" technique.
6. [LMOps](https://github.com/microsoft/LMOps) (Dec 2022): A toolkit for improving text prompts used in generative AI models, including tools like Promptist for text-to-image generation and Structured Prompting.
7. [LLMLingua](https://github.com/microsoft/LLMLingua) (Jul 2023): A tool for compressing prompts and KV-Cache, achieving up to 20x compression with minimal performance loss. LLMLingua-2 was released in Mar 2024.
8. [TypeChat](https://microsoft.github.io/TypeChat/blog/introducing-typechat) (Apr 2023): A tool that replaces prompt engineering with schema engineering, designed to build natural language interfaces using types. / [git](https://github.com/microsoft/Typechat)

#### Agent Frameworks

1. [JARVIS](https://github.com/microsoft/JARVIS) (Mar 2023): An interface for LLMs to connect numerous AI models for solving complex AI tasks.
2. [Autogen](https://github.com/microsoft/autogen) (Mar 2023): A customizable and conversable agent framework. / [ref](https://www.microsoft.com/en-us/research/blog/autogen-enabling-next-generation-large-language-model-applications/) / [Autogen Studio](https://www.microsoft.com/en-us/research/blog/introducing-autogen-studio-a-low-code-interface-for-building-multi-agent-workflows/) (June 2024)
3. [TaskWeaver](https://github.com/microsoft/TaskWeaver) (Sep 2023): A code-first agent framework for converting natural language requests into executable code with support for rich data structures and domain-adapted planning.
4. [UFO](https://github.com/microsoft/UFO) (Mar 2024): A UI-focused agent for Windows OS interaction.
5. [Semantic Workbench](https://github.com/microsoft/semanticworkbench) (Aug 2024): A development tool for creating intelligent agents. / [ref](https://techcommunity.microsoft.com/t5/ai-ai-platform-blog/introducing-semantic-workbench-your-gateway-to-agentic-ai/ba-p/4212695)
6. [OmniParser](https://github.com/microsoft/OmniParser) (Sep 2024): A simple screen parsing tool towards pure vision based GUI agent.
7. [RD-Agent](https://github.com/microsoft/RD-Agent): open source R&D automation tool [ref](https://rdagent.azurewebsites.net/) [Apr 2024]

#### Deep learning

1. [DeepSpeed](https://github.com/microsoft/DeepSpeed) (May 2020): A deep learning optimization library for easy, efficient, and effective distributed training and inference, featuring the Zero Redundancy Optimizer.
2. [FLAML](https://github.com/microsoft/FLAML) (Dec 2020): A lightweight Python library for efficient automation of machine learning and AI operations, offering interfaces for AutoGen, AutoML, and hyperparameter tuning.

#### Risk Identification & Ops

1. [PyRIT](https://github.com/Azure/PyRIT) (Dec 2023): Python Risk Identification Tool for generative AI, focusing on LLM robustness against issues like hallucination, bias, and harassment.
2. [AI Central](https://github.com/microsoft/AICentral) (Oct 2023): An AI Control Center for monitoring, authenticating, and providing resilient access to multiple OpenAI services.
3. [Azure AI Evaluation SDK](https://learn.microsoft.com/en-us/azure/ai-studio/how-to/develop/evaluate-sdk) (Oct 2024): Evaluate with the prompt flow SDK has been retired and replaced with Azure AI Evaluation SDK.

#### Data processing

- [Microsoft Fabric](https://learn.microsoft.com/en-us/fabric/): Fabric integrates technologies like Azure Data Factory, Azure Synapse Analytics, and Power BI into a single unified product [May 2023]

### **Microsoft Copilot Product Lineup**

1. Copilot Products
    - `Microsoft Copilot in Windows` vs `Microsoft Copilot` (= Copilot in Windows + Commercial Data Protection) vs `Microsoft 365 Copilot` (= Microsoft Copilot + M365 Integration) [Nov 2023]
    - [Copilot Scenario Library](https://adoption.microsoft.com/en-us/copilot-scenario-library/)
    - [An AI companion for everyone](https://blogs.microsoft.com/blog/2024/10/01/an-ai-companion-for-everyone/): Copilot’s next phase (Copilot Voice, Copilot Daily, Copilot Vision, CoT, and others.) [1 Oct 2024]
   1. Azure
       - [Microsoft Copilot for Azure](https://learn.microsoft.com/en-us/azure/copilot) / [blog](https://techcommunity.microsoft.com/t5/azure-infrastructure-blog/simplify-it-management-with-microsoft-copilot-for-azure-save/ba-p/3981106) [Nov 2023]
       - [Security Copilot](https://learn.microsoft.com/en-us/security-copilot/microsoft-security-copilot) / [blog](https://blogs.microsoft.com/blog/2023/03/28/introducing-microsoft-security-copilot-empowering-defenders-at-the-speed-of-ai/) [March 2023]
       - [Copilot in Azure Quantum](https://learn.microsoft.com/en-us/azure/quantum/get-started-azure-quantum) [June 2023]
   1. Microsoft 365 (Incl. Dynamics 365 and Power Platform)
       - [Microsoft 365 Copilot](https://learn.microsoft.com/en-us/microsoft-365-copilot/microsoft-365-copilot-overview) / [blog](https://blogs.microsoft.com/blog/2023/03/16/introducing-microsoft-365-copilot-your-copilot-for-work/) [Nov 2023]
       - Copilot in Power Platform: [Power App AI Copilot](https://learn.microsoft.com/en-us/power-apps/maker/canvas-apps/ai-overview) [March 2023] / [Power Automate](https://powerautomate.microsoft.com/en-us/blog/copilot-in-power-automate-new-time-saving-experiences-announced-at-microsoft-ignite-2023/): [Copilot in cloud flows](https://learn.microsoft.com/en-us/power-automate/get-started-with-copilot), [Copilot in Process Mining ingestion](https://learn.microsoft.com/en-us/power-automate/process-mining-copilot-in-ingestion), [Copilot in Power Automate for desktop](https://learn.microsoft.com/en-us/power-automate/desktop-flows/copilot-in-power-automate-for-desktop) ... [Nov 2023]
       - [Dynamics 365 Copilot](https://learn.microsoft.com/en-us/microsoft-cloud/dev/copilot/copilot-for-dynamics365) / [blog](https://blogs.microsoft.com/blog/2023/03/06/introducing-microsoft-dynamics-365-copilot/) [March 2023]
         - [Sales Copilot](https://learn.microsoft.com/en-us/microsoft-sales-copilot)
         - [Service Copilot](https://cloudblogs.microsoft.com/dynamics365/it/2023/11/15/announcing-microsoft-copilot-for-service/)
       - Microsoft Viva Copilot [blog](https://www.microsoft.com/en-us/microsoft-365/blog/2023/04/20/introducing-copilot-in-microsoft-viva-a-new-way-to-boost-employee-engagement-and-performance/) [April 2023]
       - Microsoft Fabric and Power BI: [blog](https://powerbi.microsoft.com/en-us/blog/empower-power-bi-users-with-microsoft-fabric-and-copilot/) / [Fabric Copilot](https://learn.microsoft.com/en-us/fabric/get-started/copilot-fabric-overview) / [PowerBI Copilot](https://learn.microsoft.com/en-us/power-bi/create-reports/copilot-introduction) [March 2024]
       - [Copilot Pro](https://support.microsoft.com/en-us/copilot-pro): Copilot Pro offers all the features of Copilot, plus faster responses, priority access to advanced models, personalized GPTs, integration with Microsoft 365 apps, and enhanced AI image creation. [Jan 2024]
       - [Team Copilot](https://www.microsoft.com/en-us/microsoft-365/blog/2024/05/21/new-agent-capabilities-in-microsoft-copilot-unlock-business-value/): Act as a valuable team member (Meeting facilitator, Group collaborator, Project manager) [May 2024]
       - [Copilot Pages](https://techcommunity.microsoft.com/t5/microsoft-365-copilot/announcing-copilot-pages-for-multiplayer-collaboration/ba-p/4242701): Copilot Pages is a dynamic, persistent canvas in Copilot chat designed for multiplayer AI collaboration [16 Sep 2024]
   1. Windows, Bing and so on
       - [Microsoft Copilot](https://copilot.microsoft.com/): FKA. Bing Chat Enterprise [Nov 2023]
       - [Microsoft Clarity Copilot](https://learn.microsoft.com/en-us/clarity/copilot/clarity-copilot): [blog](https://clarity.microsoft.com/blog/clarity-copilot/) [March 2023]
       - [Microsoft Copilot in Windows](https://learn.microsoft.com/en-us/copilot/copilot) [Sep 2023]
       - [Github Copilot](https://docs.github.com/en/copilot/getting-started-with-github-copilot) [Oct 2021]
       - [Copilot+ PC](https://blogs.microsoft.com/blog/2024/05/20/introducing-copilot-pcs/): AI-powered and NPU-equipped Windows PCs [May 2024]
       - [Windows Copilot Runtime](https://blogs.windows.com/windowsdeveloper/2024/05/21/unlock-a-new-era-of-innovation-with-windows-copilot-runtime-and-copilot-pcs/): The set of APIs powered by the 40+ on-device models, a new layer of Windows. [May 2024]
       - [Nuance DAX Copilot](https://www.nuance.com/healthcare/dragon-ai-clinical-solutions/dax-copilot.html): AI assistant for automated clinical documentation [18 Jan 2024]

2. Customize Copilot
    1. Microsoft AI and AI Studio
       - [Microsoft AI](http://microsoft.com/ai)
       - The age of copilots: [blog](https://www.linkedin.com/pulse/age-copilots-satya-nadella-2hllc) [Nov 2023]
       - [Azure AI Studio](https://learn.microsoft.com/en-us/azure/ai-studio/what-is-ai-studio): [Generative AI Developmet Hub](https://azure.microsoft.com/en-us/products/ai-studio) + Promptflow + Azure AI Content safety / [youtube](https://www.youtube.com/watch?v=Qes7p5w8Tz8) / [SDK and CLI](https://learn.microsoft.com/en-us/azure/ai-studio/how-to/sdk-generative-overview)
    1. Copilot Studio
       - The Copilot System: Explained by Microsoft [youtube](https://www.youtube.com/watch?v=E5g20qmeKpg) [Mar 2023]
       - [Microsoft Copilot Studio](https://learn.microsoft.com/en-us/microsoft-copilot-studio/): Customize Copilot for Microsoft 365. FKA. Power Virtual Agents: [ref](https://www.microsoft.com/en-us/copilot/microsoft-copilot-studio) [Nov 2023]
       - [Microsoft Copilot Dashboard](https://insights.cloud.microsoft/#/CopilotDashboard) / [blog](https://techcommunity.microsoft.com/t5/microsoft-viva-blog/new-ways-microsoft-copilot-and-viva-are-transforming-the/ba-p/3982293)
    1. [Microsoft Office Copilot: Natural Language Commanding via Program Synthesis](https://arxiv.org/abs/2306.03460): [[cnt](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=arxiv%3A+2306.03460)]: Semantic Interpreter, a natural language-friendly AI system for productivity software such as Microsoft Office that leverages large language models (LLMs) to execute user intent across application features. [6 Jun 2023]
    1. [NL2KQL](https://arxiv.org/abs/2404.02933): From Natural Language to Kusto Query [3 Apr 2024]
    1. [SpreadsheetLLM](https://arxiv.org/abs/2407.09025): Introduces an efficient method to encode Excel sheets, outperforming previous approaches with 25 times fewer tokens.[12 Jul 2024]
    1. [GraphRAG (by Microsoft)](https://arxiv.org/abs/2404.16130): RAG with a graph-based approach to efficiently answer both specific and broad questions over large text corpora1. [ref](https://microsoft.github.io/graphrag) [git](https://github.com/microsoft/graphrag) [24 Apr 2024]
    1. [AutoGen Studio](https://arxiv.org/abs/2408.15247): A No-Code Developer Tool for Building and Debugging Multi-Agent Systems [9 Aug 2024]
    1. [DRIFT Search](https://www.microsoft.com/en-us/research/blog/introducing-drift-search-combining-global-and-local-search-methods-to-improve-quality-and-efficiency/): DRIFT search (Dynamic Reasoning and Inference with Flexible Traversal). Built upon Microsoft’s GraphRAG, combining characteristics of both global and local search [31 Oct 2024]

### **Azure Reference Architectures**

|                                                                                                                                                        |                                                                                                                           |
| :----------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------: |
|                              [Azure OpenAI Embeddings QnA](https://github.com/Azure-Samples/azure-open-ai-embeddings-qna) [Apr 2023]                              | [Azure Cosmos DB + OpenAI ChatGPT](https://github.com/Azure-Samples/cosmosdb-chatgpt) C# blazor [Mar 2023] |
|                                    <img src="../files/demo-architecture.png" alt="embeddin_azure_csharp" width="200"/>                                    |                              <img src="../files/cosmos-gpt.png" alt="gpt-cosmos" width="200"/>                               |
| [C# Implementation](https://github.com/Azure-Samples/azure-search-openai-demo-csharp) ChatGPT + Enterprise data with Azure OpenAI and Cognitive Search [Apr 2023] |          [Simple ChatGPT UI application](https://github.com/Azure/openai-at-scale) TypeScript, ReactJs and Flask  [Apr 2023]        |
|                                <img src="../files/demo-architecture-csharp2.png" alt="embeddin_azure_csharp" width="200"/>                                |                              <img src="../files/chatscreen.png" alt="gpt-cosmos" width="200"/>                               |
|                                  [Azure Video Indexer demo](https://aka.ms/viopenaidemo) Azure Video Indexer + OpenAI [Apr 2023]                             |        [Miyagi](https://github.com/Azure-Samples/miyagi) Integration demonstrate for multiple langchain libraries [Feb 2023] |
|                                      <img src="../files/demo-videoindexer.png" alt="demo-videoindexer" width="200"/>                                      |                                 <img src="../files/wip-azure.png" alt="miyagi" width="200"/>                                 |
|                                  [ChatGPT + Enterprise data RAG (Retrieval-Augmented Generation)](https://github.com/Azure-Samples/azure-search-openai-demo)🏆 [Feb 2023]                             |        [Chat with your data - Solution accelerator](https://github.com/Azure-Samples/chat-with-your-data-solution-accelerator) [Jun 2023] |
|                                      <img src="../files/chatscreen2.png" alt="demo-videoindexer" height="130"/>                                      |                                 <img src="../files/cwyd-solution-architecture.png" width="200"/>                                 |


---

- Azure OpenAI Application
  - [An open-source template gallery](https://azure.github.io/awesome-azd/?tags=aicollection): 🏆AI template collection
  - ChatGPT + Enterprise data RAG (Retrieval-Augmented Generation) Demo [git](https://github.com/Azure-Samples/azure-search-openai-demo) 🏆/ [ref](https://www.youtube.com/watch?v=tW2EA4aZ_YQ): Can ChatGPT work with your enterprise data? [06 Apr 2023] [8 Feb 2023]
  - [eShopSupport](https://github.com/dotnet/eshopsupport):💡A reference .NET application using AI for a customer support ticketing system [ref](https://devblogs.microsoft.com/semantic-kernel/eshop-infused-with-ai-a-comprehensive-intelligent-app-sample-with-semantic-kernel/) [Apr 2024]
  - GPT-Azure-Search-Engine: [git](https://github.com/pablomarin/GPT-Azure-Search-Engine) `Integration of Azure Bot Service with LangChain` [Feb 2023]
  - ChatGPT Plugin Quickstart using Python and FastAPI: [git](https://github.com/Azure-Samples/openai-plugin-fastapi) [May 2023]
  - [Azure AI CLI](https://github.com/Azure/azure-ai-cli): Interactive command-line tool for ai [Jul 2023]
  - [Smart Components](https://github.com/dotnet-smartcomponents/smartcomponents): Experimental, end-to-end AI features for .NET apps [Mar 2024]
  - [Prompt Buddy](https://github.com/stuartridout/promptbuddy): 🏆Share and upvote favorite AI prompts. free Microsoft Teams Power App using Dataverse for Teams. [Mar 2024]
  - [VoiceRAG](https://github.com/Azure-Samples/aisearch-openai-rag-audio):💡Voice Using Azure AI Search and the GPT-4o Realtime API for Audio [ref](https://techcommunity.microsoft.com/t5/ai-azure-ai-services-blog/voicerag-an-app-pattern-for-rag-voice-using-azure-ai-search-and/ba-p/4259116) [Sep 2024]
  - [ARGUS](https://github.com/Azure-Samples/ARGUS): Hybrid approach with Azure Document Intelligence combined and GPT4-Vision to get better results without any pre-training. [Jun 2024]
  - Semantic Kernel sample application [x-ref](sk_dspy.md/#code-recipes)

- Azure OpenAI Accelerator / samples
  - [GPT-RAG](https://github.com/Azure/GPT-RAG):💡Enterprise RAG Solution Accelerator (GPT-RAG). Retrieval-Augmented Generation pattern running in Azure. [Jun 2023]
  - [Azure-Cognitive-Search-Azure-OpenAI-Accelerator](https://github.com/MSUSAzureAccelerators/Azure-Cognitive-Search-Azure-OpenAI-Accelerator) [May 2023]
  - [Conversational-Azure-OpenAI-Accelerator](https://github.com/MSUSAzureAccelerators/Conversational-Azure-OpenAI-Accelerator) [Feb 2022]
  - Azure OpenAI samples: [ref](https://github.com/Azure/azure-openai-samples) [Apr 2023]
  - The repository for all Azure OpenAI Samples complementing the OpenAI cookbook.: [ref](https://github.com/Azure-Samples/openai) [Apr 2023]
  - Azure-Samples [ref](https://github.com/Azure-Samples)
    - Azure OpenAI with AKS By Terraform: [git](https://github.com/Azure-Samples/aks-openai-terraform) [Jun 2023]
    - Azure OpenAI with AKS By Bicep: [git](https://github.com/Azure-Samples/aks-openai) [May 2023]
  - Enterprise Logging: [git](https://github.com/Azure-Samples/openai-python-enterprise-logging) [Feb 2023] / [Setting up Azure OpenAI with Azure API Management](https://github.com/Azure/enterprise-azureai) [Jan 2024]
  - Azure OpenAI with AKS by Terraform (simple version): [git](https://github.com/Azure-Samples/azure-openai-terraform-deployment-sample) [May 2023]
  - Azure OpenAI Network Latency Test Script
  : [git](https://github.com/wloryo/networkchatgpt/blob/dc76f2264ff8c2a83392e6ae9ee2aaa55ca86f0e/openai_network_latencytest_nocsv_pub_v1.1.py) [Jun 2023]
  - Create an Azure OpenAI, LangChain, ChromaDB, and Chainlit ChatGPT-like application in Azure Container Apps using Terraform [git](https://github.com/Azure-Samples/container-apps-openai/) [Jul 2023]
  - [Azure SQL DB + AOAI](https://github.com/Azure-Samples/SQL-AI-samples): AI applications built on data from an Azure SQL Database
  - [Smart load balancing for AOAI](https://github.com/Azure-Samples/openai-aca-lb): Smart load balancing for OpenAI endpoints
  - [Azure Functions (C#) bindings for OpenAI](https://github.com/Azure/azure-functions-openai-extension)
  - [RAG for Azure Data](https://github.com/microsoft/AzureDataRetrievalAugmentedGenerationSamples): Retrieval Augmented Generation (RAG) for Azure Data
  - [AI-Sentry](https://github.com/microsoft/ai-sentry): A lightweight, pluggable facade layer for AOAI
  - [AI-in-a-Box](https://github.com/Azure/AI-in-a-Box): AI-in-a-Box aims to provide an "Azure AI/ML Easy Button" for common scenarios [Sep 2023]
  - [AI Samples for .NET](https://github.com/dotnet/ai-samples):  official .NET samples demonstrating how to use AI [Feb 2024]
  - [OpenAI Official .NET Library](https://github.com/openai/openai-dotnet/) [Apr 2024]
  - [Azure Multimodal AI + LLM Processing Accelerator](https://github.com/Azure/multimodal-ai-llm-processing-accelerator): Build multimodal data processing pipelines with Azure AI Services + LLMs [Aug 2024]
  - [azure-llm-fine-tuning](https://github.com/Azure/azure-llm-fine-tuning): SLM/LLM Fine-tuning on Azure [May 2024]
  - [OpenAI Chat Application with Microsoft Entra Authentication](https://github.com/Azure-Samples/openai-chat-app-entra-auth-builtin): Microsoft Entra ID for user authentication  [May 2024]
  - [Evaluating a RAG Chat App](https://github.com/Azure-Samples/ai-rag-chat-evaluator): Tools for evaluation of RAG Chat Apps using Azure AI Evaluate SDK [Nov 2023]
  - [Microsoft.Extensions.AI](https://devblogs.microsoft.com/dotnet/introducing-microsoft-extensions-ai-preview/): a unified layer of C# abstractions for interacting with AI services, such as small and large language models (SLMs and LLMs), embeddings, and middleware. [8 Oct 2024]
  - [Fabric notebooks for analyzing chat history stored in CosmosDB](https://github.com/microsoft/fabric-cosmosdb-chat-analytics) [Sep 2024]

- Referece: Use Case and Architecture
  - [AI Feed](https://techcommunity.microsoft.com/t5/artificial-intelligence-and/ct-p/AI) | [AI Platform Blog](https://techcommunity.microsoft.com/t5/ai-ai-platform-blog/bg-p/AIPlatformBlog) 
  - [Azure Command Companion](https://techcommunity.microsoft.com/t5/analytics-on-azure-blog/azure-command-companion/ba-p/4005044): Harnessing the Power of OpenAI GPT-3.5 Turbo for Azure CLI Command Generation [10 Dec 2023 ]
  - [Chat with your Azure DevOps data](https://techcommunity.microsoft.com/t5/fasttrack-for-azure/chat-with-your-azure-devops-data/ba-p/4017784) [10 Jan 2024]
  - [Baseline OpenAI end-to-end chat reference architecture](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/architecture/baseline-openai-e2e-chat)
  - [Build language model pipelines with memory](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/openai/guide/language-model-pipelines)
  - [NL to SQL Architecture Alternative](https://techcommunity.microsoft.com/t5/azure-architecture-blog/nl-to-sql-architecture-alternatives/ba-p/4136387) [14 May 2024] / [Natural Language to SQL Console](https://github.com/microsoft/kernel-memory/tree/NL2SQL/examples/200-dotnet-nl2sql)
  - [Azure OpenAI workshop](https://github.com/microsoft/OpenAIWorkshop): Workshop materials to build intelligent solutions on Open AI [Feb 2023]
  - [Responsible AI Transparency Report](https://www.microsoft.com/en-us/corporate-responsibility/responsible-ai-transparency-report)
  - [Safeguard and trustworthy generative AI applications](https://azure.microsoft.com/en-us/blog/announcing-new-tools-in-azure-ai-to-help-you-build-more-secure-and-trustworthy-generative-ai-applications/) [28 Mar 2024]
  - [Microsoft AI / Responsible AI](https://aka.ms/RAIResources) 🏆
  - [Baseline Agentic AI Systems Architecture](https://techcommunity.microsoft.com/t5/ai-machine-learning-blog/baseline-agentic-ai-systems-architecture/ba-p/4207137) [20 Aug 2024]
  - [AI Agent-Driven Auto Insurance Claims RAG Pipeline](https://techcommunity.microsoft.com/t5/azure-architecture-blog/exploring-ai-agent-driven-auto-insurance-claims-rag-pipeline/ba-p/4233779) [09 Sep 2024]
  - [Grounding LLMs](https://techcommunity.microsoft.com/t5/fasttrack-for-azure/grounding-llms/ba-p/3843857): Retrieval-Augmented Generation (RAG) [09 Jun 2023]
  - [Revolutionize your Enterprise Data with ChatGPT](https://techcommunity.microsoft.com/t5/ai-applied-ai-blog/revolutionize-your-enterprise-data-with-chatgpt-next-gen-apps-w/ba-p/3762087) [09 Mar 2023]
  - [Check Your Facts and Try Again: Improving Large Language Models with External Knowledge and Automated Feedback](https://www.microsoft.com/en-us/research/group/deep-learning-group/articles/check-your-facts-and-try-again-improving-large-language-models-with-external-knowledge-and-automated-feedback/) [07 Mar 2023]
  - [Azure OpenAI Design Patterns](https://github.com/microsoft/azure-openai-design-patterns): A set of design patterns using the Azure OpenAI service [May 2023]
  - [Azure AI Services Landing Zone](https://github.com/FreddyAyala/AzureAIServicesLandingZone) / [ref](https://techcommunity.microsoft.com/t5/azure-architecture-blog/azure-openai-landing-zone-reference-architecture/ba-p/3882102) [24 Jul 2023]
  - [Security Best Practices for GenAI Applications (OpenAI) in Azure](https://techcommunity.microsoft.com/t5/azure-architecture-blog/security-best-practices-for-genai-applications-openai-in-azure/ba-p/4027885) [16 Jan 2024]
  - [Authentication and Authorization in Generative AI applications with Entra ID and Azure AI Search](https://techcommunity.microsoft.com/t5/fasttrack-for-azure/authentication-and-authorization-in-generative-ai-applications/ba-p/4022277) [09 Jan 2024]
  - [Integrate private access to your Azure Open AI Chatbot](https://techcommunity.microsoft.com/t5/fasttrack-for-azure/integrate-private-access-to-your-azure-open-ai-chatbot/ba-p/3994613) [30 Nov 2023]
  - [An Introduction to LLMOps](https://techcommunity.microsoft.com/t5/ai-machine-learning-blog/an-introduction-to-llmops-operationalizing-and-managing-large/ba-p/3910996): Operationalizing and Managing Large Language Models using Azure ML [27 Aug 2023]
  - [Optimize Azure OpenAI Applications with Semantic Caching](https://techcommunity.microsoft.com/t5/azure-architecture-blog/optimize-azure-openai-applications-with-semantic-caching/ba-p/4106867) [09 Apr 2024]
  - [Azure OpenAI and Call Center Modernization](https://techcommunity.microsoft.com/t5/azure-architecture-blog/azure-openai-and-call-center-modernization/ba-p/4107070) [11 Apr2024]
  - [Azure OpenAI Best Practices Insights from Customer Journeys](https://techcommunity.microsoft.com/t5/ai-azure-ai-services-blog/azure-openai-best-practices-insights-from-customer-journeys/ba-p/4166943): LLMLingua, Skeleton Of Thought [12 Jun 2024]
  - [Retrieval Augmented Fine Tuning](https://techcommunity.microsoft.com/t5/ai-ai-platform-blog/retrieval-augmented-fine-tuning-use-gpt-4o-to-fine-tune-gpt-4o/ba-p/4248861): RAFT: Combining the best parts of RAG and fine-tuning (SFT) [25 Sep 2024]
  - [Using keyless authentication with Azure OpenAI](https://techcommunity.microsoft.com/t5/microsoft-developer-community/using-keyless-authentication-with-azure-openai/ba-p/4111521) [12 Apr 2024]
  - [Baseline OpenAI end-to-end chat reference architecture](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/architecture/baseline-openai-e2e-chat)
  - [Designing and developing a RAG solution](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-solution-design-and-evaluation-guide)
  - [Microsoft Copilot Studio Samples](https://github.com/microsoft/CopilotStudioSamples): Samples and artifacts for Microsoft Copilot Studio

#### **Azure AI Search**

- Azure Cognitive Search rebranding Azure AI Search, it supports Vector search and semantic ranker. [16 Nov 2023]
- In the vector databases category within Azure, several alternative solutions are available. However, the only option that provides a range of choices, including a conventional Lucene-based search engine and a hybrid search incorporating vector search capabilities.
- Vector Search Sample Code: [git](https://github.com/Azure/azure-search-vector-samples) [Apr 2023]
- Azure AI Search (FKA. Azure Cognitive Search) supports
  1. Text Search
  1. Pure Vector Search
  1. Hybrid Search (Text search + Vector search)
  1. Semantic Hybrid Search (Text search + Semantic search + Vector search)
- A set of capabilities designed to improve relevance in these scenarios. We use a combination of hybrid retrieval (vector search + keyword search) + semantic ranking as the most effective approach for improved relevance out-of–the-box. `TL;DR: Retrieval Performance; Hybrid search + Semantic rank > Hybrid search > Vector only search > Keyword only` [ref](https://techcommunity.microsoft.com/t5/azure-ai-services-blog/azure-cognitive-search-outperforming-vector-search-with-hybrid/ba-p/3929167) [18 Sep 2023] <br/>
  <img src="../files/acs-hybrid.png" alt="acs" width="300"/>
- Hybrid search using Reciprocal Rank Fusion (RRF): Reciprocal Rank Fusion (RRF) is an algorithm that evaluates the search scores from multiple, previously ranked results to produce a unified result set. In Azure Cognitive Search, RRF is used whenever there are two or more queries that execute in parallel. [ref](https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking)
- [Integrated vectorization](https://techcommunity.microsoft.com/t5/ai-azure-ai-services-blog/integrated-vectorization-with-azure-openai-for-azure-ai-search/ba-p/4206836): Automatically splits documents into chunks, creates embeddings with Azure OpenAI, maps them to an Azure AI Search index, and automates query vectorization. [24 Aug 2024]

### **Azure Enterprise Services**

- Azure OpenAI Service Offerings
  1. Offering: Standard (Default), Batch (Low-cost, Huge workload), Provisioned (High performance)
  1. Offering Region types: Global (World wide), Data_zones (Zone based), Regional (Region based)
- Copilot (FKA. Bing Chat Enterprise) [18 Jul 2023] [Privacy and Protection](https://learn.microsoft.com/en-us/bing-chat-enterprise/privacy-and-protections#protected-by-default)
  1. Doesn't have plugin support
  1. Only content provided in the chat by users is accessible to Bing Chat Enterprise.
- Azure OpenAI Service On Your Data in Public Preview [ref](https://techcommunity.microsoft.com/t5/ai-cognitive-services-blog/introducing-azure-openai-service-on-your-data-in-public-preview/ba-p/3847000) [19 Jun 2023]
- Azure OpenAI Finetuning: Babbage-002 is $34/hour, Davinci-002 is $68/hour, and Turbo is $102/hour. [ref](https://techcommunity.microsoft.com/t5/azure-ai-services-blog/fine-tuning-now-available-with-azure-openai-service/ba-p/3954693) [16 Oct 2023]
- Customer Copyright Commitment: protects customers from certain IP claims related to AI-generated content. [ref](https://learn.microsoft.com/en-us/legal/cognitive-services/openai/customer-copyright-commitment) [16 Nov 2023]
- [Models as a Service (MaaS)](https://www.linkedin.com/pulse/model-service-maas-revolutionizing-ai-azure-shibu-kt): A cloud-based AI approach that provides developers and businesses with access to pre-built, pre-trained machine learning models. [July 2023]
- [Assistants API](https://techcommunity.microsoft.com/t5/ai-azure-ai-services-blog/azure-openai-service-announces-assistants-api-new-models-for/ba-p/4049940): Code Interpreter, Function calling, Knowledge retrieval tool, and Threads (Truncated and optimized conversation history for the model's context length) in Azure [06 Feb 2024]
