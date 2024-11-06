## **Section 7** : Large Language Model: Landscape

### **Large Language Models (in 2023)**

1. Change in perspective is necessary because some abilities only emerge at a certain scale. Some conclusions from the past are invalidated and we need to constantly unlearn intuitions built on top of such ideas.
1. From first-principles, scaling up the Transformer amounts to efficiently doing matrix multiplications with many, many machines.
1. Further scaling (think 10000x GPT-4 scale). It entails finding the inductive bias that is the bottleneck in further scaling.

- [Twitter](https://twitter.com/hwchung27/status/1710003293223821658) / [Video](https://t.co/vumzAtUvBl) / [Slides](https://t.co/IidLe4JfrC) [6 Oct 2023]
- [LLMprices.dev](https://llmprices.dev): Compare prices for models like GPT-4, Claude Sonnet 3.5, Llama 3.1 405b and many more.
- [AI Model Review](https://aimodelreview.com/): Compare 75 AI Models on 200+ Prompts Side By Side.
- [Artificial Analysis](https://artificialanalysis.ai/): Independent analysis of AI models and API providers.
- [Inside language models (from GPT to Olympus)](https://lifearchitect.ai/models/)
- [LLM Pre-training and Post-training Paradigms](https://sebastianraschka.com/blog/2024/new-llm-pre-training-and-post-training.html) [17 Aug 2024] <br/>
  <img src="../files/llm-dev-pipeline-overview.png" width="350" />

### **Evolutionary Tree of Large Language Models**

- Evolutionary Graph of LLaMA Family

  <img src="../files/llama-0628-final.png" width="450" />

- LLM evolutionary tree

  <!-- <img src="../files/qr_version.jpg" alt="llm" width="450"/> -->
  <img src="../files/tree.png" alt="llm" width="450"/>

- [A Survey of Large Language Models](https://arxiv.org/abs/2303.18223): [[cnt](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=arxiv%3A+2303.18223)] /[git](https://github.com/RUCAIBox/LLMSurvey) [31 Mar 2023] contd.

- [LLM evolutionary tree](https://arxiv.org/abs/2304.13712): [[cnt](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=arxiv%3A+2304.13712)]: A curated list of practical guide resources of LLMs (LLMs Tree, Examples, Papers) [git](https://github.com/Mooler0410/LLMsPracticalGuide) [26 Apr 2023]

### **A Taxonomy of Natural Language Processing**

- An overview of different fields of study and recent developments in NLP. [doc](../files/taxonomy-nlp.pdf) [ref](https://towardsdatascience.com/a-taxonomy-of-natural-language-processing-dfc790cb4c01) [24 Sep 2023]

  “Exploring the Landscape of Natural Language Processing Research” [ref](https://arxiv.org/abs/2307.10652) [20 Jul 2023]

  <img src="../files/taxonomy-nlp.png" width="650" />

  NLP taxonomy

  <img src="../files/taxonomy-nlp2.png" width="650" />

  Distribution of the number of papers by most popular fields of study from 2002 to 2022

### **Open-Source Large Language Models**

- [The Open Source AI Definition](https://opensource.org/ai/open-source-ai-definition) [28 Oct 2024]
- [The LLM Index](https://sapling.ai/llm/index): A list of large language models (LLMs)
- [Chatbot Arena](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard): Benchmarking LLMs in the Wild with Elo Ratings
- [LLM Collection](https://www.promptingguide.ai/models/collection): promptingguide.ai
- [Huggingface Open LLM Learboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
- [ollam](https://ollama.com/library?sort=popular): ollama-supported models
- [KoAlpaca](https://github.com/Beomi/KoAlpaca): Alpaca for korean [Mar 2023]
- [Pythia](https://arxiv.org/abs/2304.01373): How do large language models (LLMs) develop and evolve over the course of training and change as models scale? A suite of decoder-only autoregressive language models ranging from 70M to 12B parameters [git](https://github.com/EleutherAI/pythia) [Apr 2023]
- [OLMo](https://arxiv.org/abs/2402.00838): Truly open language model and framework to build, study, and advance LMs, along with the training data, training and evaluation code, intermediate model checkpoints, and training logs. [git](https://github.com/allenai/OLMo) [Feb 2024] / [OLMoE](https://github.com/allenai/OLMoE): fully-open LLM leverages sparse Mixture-of-Experts [Sep 2024]
- [Open-Sora](https://github.com/hpcaitech/Open-Sora): Democratizing Efficient Video Production for All  [Mar 2024]
- [Jamba](https://www.ai21.com/blog/announcing-jamba): AI21's SSM-Transformer Model. Mamba  + Transformer + MoE [28 Mar 2024]
- Meta (aka. Facebook)
  1. Most OSS LLM models have been built on the [Llama](https://github.com/facebookresearch/llama) / [ref](https://ai.meta.com/llama) / [git](https://github.com/meta-llama/llama-models)
  1. [Llama 2](https://huggingface.co/blog/llama2): 1) 40% more data than Llama. 2)7B, 13B, and 70B. 3) Trained on over 1 million human annotations. 4) double the context length of Llama 1: 4K 5) Grouped Query Attention, KV Cache, and Rotary Positional Embedding were introduced in Llama 2 [18 Jul 2023] [demo](https://huggingface.co/blog/llama2#demo)
  1. [Llama 3](https://llama.meta.com/llama3/): 1) 7X more data than Llama 2. 2) 8B, 70B, and 400B. 3) 8K context length [18 Apr 2024]
  1. [MEGALODON](https://github.com/XuezheMax/megalodon): Long Sequence Model. Unlimited context length. Outperforms Llama 2 model. [Apr 2024]
  1. [Llama 3.1](https://ai.meta.com/blog/meta-llama-3-1/): 405B, context length to 128K, add support across eight languages. first OSS model outperforms GTP-4o. [23 Jul 2024] / [Llama 3.2](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/): Multimodal 11B and 90B model support image reasoning. lightweight 1B and 3B models. [25 Sep 2024]
  1. [Llama 3.2](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/): Multimodal. Include text-only models (1B, 3B) and text-image models (11B, 90B), with quantized versions of 1B and 3B [Sep 2024]
  1. [NotebookLlama](https://github.com/meta-llama/llama-recipes/tree/main/recipes/quickstart/NotebookLlama): An Open Source version of NotebookLM [28 Oct 2024]
- Google
  1. [Gemma](http://ai.google.dev/gemma): Open weights LLM from Google DeepMind. [git](https://github.com/google-deepmind/gemma) / Pytorch [git](https://github.com/google/gemma_pytorch) [Feb 2024]
  1. [Gemma 2](https://www.kaggle.com/models/google/gemma-2/) 2B, 9B, 27B [ref: releases](https://ai.google.dev/gemma/docs/releases) [Jun 2024]
  1. [DataGemma](https://blog.google/technology/ai/google-datagemma-ai-llm/) [12 Sep 2024] / [NotebookLM](https://blog.google/technology/ai/notebooklm-audio-overviews/): LLM-powered notebook. free to use, not open-source. [12 Jul 2023]
- Qualcomm
  1. [Qualcomm’s on-device AI models](https://huggingface.co/qualcomm): Bring generative AI to mobile devices [Feb 2024]
- xAI
  - xAI is an American AI company founded by Elon Musk in March 2023
  1. [Grok](https://x.ai/blog/grok-os): 314B parameter Mixture-of-Experts (MoE) model. Released under the Apache 2.0 license. Not includeded training code. Developed by JAX [git](https://github.com/xai-org/grok) [17 Mar 2024]
  1. [Grok-2 and Grok-2 mini](https://x.ai/blog/grok-2) [13 Aug 2024]
- Databricks
  1. [DBRX](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm): MoE, open, general-purpose LLM created by Databricks. [git](https://github.com/databricks/dbrx) [27 Mar 2024]
- Apple
  1. [OpenELM](https://machinelearning.apple.com/research/openelm): Apple released a Transformer-based language model. Four sizes of the model: 270M, 450M, 1.1B, and 3B parameters. [April 2024]
  1. [Apple Intelligence Foundation Language Models](https://machinelearning.apple.com/research/apple-intelligence-foundation-language-models): 1. A 3B on-device model used for language tasks like summarization and Writing Tools. 2. A large Server model used for language tasks too complex to do on-device. [10 Jun 2024]
- Microsoft
  1. phi-series: cost-effective small language models (SLMs) [x-ref](prompt_ft.md/#knowledge-distillation-reducing-model-size-with-textbooks)
- NVIDIA
  1. [Nemotron-4 340B](https://research.nvidia.com/publication/2024-06_nemotron-4-340b): Synthetic Data Generation for Training Large Language Models [14 Jun 2024]
- Mistral
  - Founded in April 2023. French tech.
  1. open-weights models (Mistral 7B, Mixtral 8x7B, Mixtral 8x22B, NeMo) and optimized commercial models (Mistral Small, Mistral Medium, Mistral Large) [ref](https://docs.mistral.ai/getting-started/models/)
  1. [NeMo](https://mistral.ai/news/mistral-nemo/): 12B model with 128k context length that outperforms LLama 3 8B [18 Jul 2024]
- Groq
  - Founded in 2016. low-latency AI inference H/W. American tech.
  1. [Llama-3-Groq-Tool-Use](https://wow.groq.com/introducing-llama-3-groq-tool-use-models/): a model optimized for function calling [Jul 2024]
- Alibaba
  1. [Qwen series](https://github.com/QwenLM) > [Qwen2](https://github.com/QwenLM/Qwen2): 29 languages. 5 sizes, including Qwen2-0.5B, Qwen2-1.5B, Qwen2-7B, Qwen2-57B-A14B, and Qwen2-72B. [Feb 2024]
- Cohere
  - Founded in 2019. Canadian multinational tech.
  1. [Command R+](https://huggingface.co/collections/CohereForAI/c4ai-command-r-plus-660ec4c34f7a69c50ce7f7b9): The performant model for RAG capabilities, multilingual support, and tool use. [Aug 2024]
- Deepseek
  - Founded in 2023, is a Chinese company dedicated to AGI.
  1. A list of models: [git](https://github.com/deepseek-ai)
- GPT for Domain Specific [x-ref](llm.md/#gpt-for-domain-specific)
- MLLM (multimodal large language model) [x-ref](llm.md/#mllm-multimodal-large-language-model)
- Large Language Models (in 2023) [x-ref](llm.md/#large-language-models-in-2023)

<details open>
<summary>Expand: Llama variants emerged in 2023</summary>

- Upstage's 70B Language Model Outperforms GPT-3.5: [ref](https://en.upstage.ai/newsroom/upstage-huggingface-llm-no1) [1 Aug 2023]
- [Falcon LLM](https://falconllm.tii.ae/) Apache 2.0 license [Mar 2023]
- [StableVicuna](https://stability.ai/blog/stablevicuna-open-source-rlhf-chatbot) First Open Source RLHF LLM Chatbot [Apr 2032]
- [Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html): Fine-tuned from the LLaMA 7B model [Mar 2023]
- [vicuna](https://vicuna.lmsys.org/): 90% ChatGPT Quality [Mar 2023]
- [Koala](https://bair.berkeley.edu/blog/2023/04/03/koala/): Focus on dialogue data gathered from the web.  [Apr 2023]
- [dolly](https://www.databricks.com/blog/2023/03/24/hello-dolly-democratizing-magic-chatgpt-open-models.html): Databricks [Mar 2023]
- [Cerebras-GPT](https://www.cerebras.net/blog/cerebras-gpt-a-family-of-open-compute-efficient-large-language-models/): 7 GPT models ranging from 111m to 13b parameters. [Mar 2023]

</details>

### **LLM for Domain Specific**

- [TimeGPT](https://nixtla.github.io/nixtla/): The First Foundation Model for Time Series Forecasting [git](https://github.com/Nixtla/neuralforecast) [Mar 2023]
- [BioGPT](https://arxiv.org/abs/2210.10341): [[cnt](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=arxiv%3A+2210.10341)]: Generative Pre-trained Transformer for Biomedical Text Generation and Mining [git](https://github.com/microsoft/BioGPT) [19 Oct 2022]
- [MeshGPT](https://nihalsid.github.io/mesh-gpt/): Generating Triangle Meshes with Decoder-Only Transformers [27 Nov 2023]
- [BloombergGPT](https://arxiv.org/abs/2303.17564): A Large Language Model for Finance [30 Mar 2023]
- [Galactica](https://arxiv.org/abs/2211.09085): A Large Language Model for Science [16 Nov 2022]
- [EarthGPT](https://arxiv.org/abs/2401.16822): A Universal Multi-modal Large Language Model for Multi-sensor Image Comprehension in Remote Sensing Domain [30 Jan 2024]
- [SaulLM-7B](https://arxiv.org/abs/2403.03883): A pioneering Large Language Model for Law [6 Mar 2024]
- [Huggingface StarCoder: A State-of-the-Art LLM for Code](https://huggingface.co/blog/starcoder): [git](https://huggingface.co/bigcode/starcoder) [May 2023]
- [Code Llama](https://arxiv.org/abs/2308.12950): Built on top of Llama 2, free for research and commercial use. [ref](https://ai.meta.com/blog/code-llama-large-language-model-coding/) / [git](https://github.com/facebookresearch/codellama) [24 Aug 2023]
- [Devin AI](https://preview.devin.ai/): Devin is an AI software engineer developed by Cognition AI [12 Mar 2024]
- [FrugalGPT](https://arxiv.org/abs/2305.05176): LLM with budget constraints, requests are cascaded from low-cost to high-cost LLMs. [git](https://github.com/stanford-futuredata/FrugalGPT) [9 May 2023]
- [DeepSeek-Coder-V2](https://github.com/deepseek-ai/DeepSeek-Coder-V2): Open-source Mixture-of-Experts (MoE) code language model [17 Jun 2024]
- [Qwen2-Math](https://github.com/QwenLM/Qwen2-Math): math-specific LLM / [Qwen2-Audio](https://github.com/QwenLM/Qwen2-Audio): large-scale audio-language model [Aug 2024] / [Qwen 2.5-Coder](https://github.com/QwenLM/Qwen2.5-Coder) [18 Sep 2024 ]
- [Chai-1](https://github.com/chaidiscovery/chai-lab): a multi-modal foundation model for molecular structure prediction [Sep 2024]
- [Prithvi WxC](https://arxiv.org/abs/2409.13598): In collaboration with NASA, IBM is releasing an open-source foundation model for Weather and Climate [ref](https://research.ibm.com/blog/foundation-model-weather-climate) [20 Sep 2024]
- [AlphaChip](https://deepmind.google/discover/blog/how-alphachip-transformed-computer-chip-design/): Reinforcement learning-based model for designing physical chip layouts. [26 Sep 2024]
- [AlphaFold3](https://github.com/Ligo-Biosciences/AlphaFold3): Open source implementation of AlphaFold3 [Nov 2023] / [OpenFold](https://github.com/aqlaboratory/openfold): PyTorch reproduction of AlphaFold 2 [Sep 2021]
- [MechGPT](https://arxiv.org/abs/2310.10445): Language Modeling Strategies for Mechanics and Materials [git](https://github.com/lamm-mit/MeLM) [16 Oct 2023]

### **MLLM (multimodal large language model)**

- [Understanding Multimodal LLMs](https://magazine.sebastianraschka.com/p/understanding-multimodal-llms):💡Two main approaches to building multimodal LLMs: 1. Unified Embedding Decoder Architecture approach; 2. Cross-modality Attention Architecture approach. [3 Nov 2024]
- [Multimodal Foundation Models: From Specialists to General-Purpose Assistants](https://arxiv.org/abs/2309.10020): [[cnt](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=arxiv%3A+2309.10020)]: A comprehensive survey of the taxonomy and evolution of multimodal foundation models that demonstrate vision and vision-language capabilities. Specific-Purpose 1. Visual understanding tasks 2. Visual generation tasks General-Purpose 3. General-purpose interface. [18 Sep 2023]
- [Awesome Multimodal Large Language Models](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models): Latest Papers and Datasets on Multimodal Large Language Models, and Their Evaluation. [Jun 2023]
- [CLIP](https://arxiv.org/abs/2103.00020): [[cnt](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=arxiv%3A+2103.00020)]: CLIP (Contrastive Language-Image Pretraining), Trained on a large number of internet text-image pairs and can be applied to a wide range of tasks with zero-shot learning. [git](https://github.com/openai/CLIP) [26 Feb 2021]
- [LLaVa](https://arxiv.org/abs/2304.08485): [[cnt](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=arxiv%3A+2304.08485)]: Large Language-and-Vision Assistant [git](https://llava-vl.github.io/) [17 Apr 2023]
  - Simple linear layer to connect image features into the word embedding space. A trainable projection matrix W is applied to the visual features Zv, transforming them into visual embedding tokens Hv. These tokens are then concatenated with the language embedding sequence Hq to form a single sequence. Note that Hv and Hq are not multiplied or added, but concatenated, both are same dimensionality.
  - [LLaVA-1.5](https://arxiv.org/abs/2310.03744): [[cnt](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=arxiv%3A+2310.03744)]: is out! [git](https://github.com/haotian-liu/LLaVA): Changing from a linear projection to an MLP cross-modal. [5 Oct 2023]
- [Video-ChatGPT](https://arxiv.org/abs/2306.05424): [[cnt](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=arxiv%3A+2306.05424)]: a video conversation model capable of generating meaningful conversation about videos. / [git](https://github.com/mbzuai-oryx/Video-ChatGPT) [8 Jun 2023]
- [MiniGPT-4 & MiniGPT-v2](https://arxiv.org/abs/2304.10592): [[cnt](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=arxiv%3A+2304.10592)]: Enhancing Vision-language Understanding with Advanced Large Language Models [git](https://minigpt-4.github.io/) [20 Apr 2023]
- [TaskMatrix, aka VisualChatGPT](https://arxiv.org/abs/2303.04671): [[cnt](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=arxiv%3A+2303.04671)]: Microsoft TaskMatrix [git](https://github.com/microsoft/TaskMatrix); GroundingDINO + [SAM](https://arxiv.org/abs/2304.02643) [git](https://github.com/facebookresearch/segment-anything.git) [8 Mar 2023]
- [GroundingDINO](https://arxiv.org/abs/2303.05499): [[cnt](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=arxiv%3A+2303.05499)]: DINO with Grounded Pre-Training for Open-Set Object Detection [git](https://github.com/IDEA-Research/GroundingDINO) [9 Mar 2023]
- [BLIP-2](https://arxiv.org/abs/2301.12597) [30 Jan 2023]: [[cnt](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=arxiv%3A+2301.12597)]: Salesforce Research, Querying Transformer (Q-Former) / [git](https://github.com/salesforce/LAVIS/blob/main/lavis/models/blip2_models/blip2_qformer.py) / [ref](https://huggingface.co/blog/blip-2) / [Youtube](https://www.youtube.com/watch?v=k0DAtZCCl1w) / [BLIP](https://arxiv.org/abs/2201.12086): [[cnt](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=arxiv%3A+2201.12086)]: [git](https://github.com/salesforce/BLIP) [28 Jan 2022]
  - `Q-Former (Querying Transformer)`: A transformer model that consists of two submodules that share the same self-attention layers: an image transformer that interacts with a frozen image encoder for visual feature extraction, and a text transformer that can function as both a text encoder and a text decoder.
  - Q-Former is a lightweight transformer which employs a set of learnable query vectors to extract visual features from the frozen image encoder. It acts as an information bottleneck between the frozen image encoder and the frozen LLM.
  <!--
  https://zhuanlan.zhihu.com/p/635603332
  https://zhuanlan.zhihu.com/p/613247637
  https://zhuanlan.zhihu.com/p/604318703
  https://zhuanlan.zhihu.com/p/104393915
  -->
- [MiniCPM-V](https://github.com/OpenBMB/MiniCPM-V): MiniCPM-Llama3-V 2.5: A GPT-4V Level Multimodal LLM on Your Phone [Jan 2024]
- Vision capability to a LLM [ref](https://cloud.google.com/blog/products/ai-machine-learning/multimodal-generative-ai-search/) [22 Aug 2023]

  - The model has three sub-models:
    1. A model to obtain image embeddings
    1. A text model to obtain text embeddings
    1. A model to learn the relationships between them
  - This is analogous to adding vision capability to a LLM.

    <img src="../files/cocoa.gif" width="200" />

- Meta (aka. Facebook)
  1. [facebookresearch/ImageBind](https://arxiv.org/abs/2305.05665): [[cnt](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=arxiv%3A+2305.05665)]: ImageBind One Embedding Space to Bind Them All [git](https://github.com/facebookresearch/ImageBind) [9 May 2023]
  1. [facebookresearch/segment-anything(SAM)](https://arxiv.org/abs/2304.02643): [[cnt](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=arxiv%3A+2304.02643)]: The repository provides code for running inference with the SegmentAnything Model (SAM), links for downloading the trained model checkpoints, and example notebooks that show how to use the model. [git](https://github.com/facebookresearch/segment-anything) [5 Apr 2023]
  1. [facebookresearch/SeamlessM4T](https://arxiv.org/abs/2308.11596): [[cnt](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=arxiv%3A+2308.11596)]: SeamlessM4T is the first all-in-one multilingual multimodal AI translation and transcription model. This single model can perform speech-to-text, speech-to-speech, text-to-speech, and text-to-text translations for up to 100 languages depending on the task. [ref](https://about.fb.com/news/2023/08/seamlessm4t-ai-translation-model/) [22 Aug 2023]
  1. [Chameleon](https://arxiv.org/abs/2405.09818): Early-fusion token-based mixed-modal models capable of understanding and generating images and text in any arbitrary sequence. The unified approach uses fully token-based representations for both image and textual modalities. [16 May 2024]
  1. [Models and libraries](https://ai.meta.com/resources/models-and-libraries/)
- Microsoft
  1. Language Is Not All You Need: Aligning Perception with Language Models [Kosmos-1](https://arxiv.org/abs/2302.14045): [[cnt](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=arxiv%3A+2302.14045)] [27 Feb 2023]
  2. [Kosmos-2](https://arxiv.org/abs/2306.14824): [[cnt](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=arxiv%3A+2306.14824)]: Grounding Multimodal Large Language Models to the World [26 Jun 2023]
  3. [Kosmos-2.5](https://arxiv.org/abs/2309.11419): [[cnt](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=arxiv%3A+2309.11419)]: A Multimodal Literate Model [20 Sep 2023]
  4. [BEiT-3](https://arxiv.org/abs/2208.10442): [[cnt](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=arxiv%3A+2208.10442)]: Image as a Foreign Language: BEiT Pretraining for Vision and Vision-Language Tasks [22 Aug 2022]
  5. [TaskMatrix.AI](https://arxiv.org/abs/2303.16434): [[cnt](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=arxiv%3A+2303.16434)]: TaskMatrix connects ChatGPT and a series of Visual Foundation Models to enable sending and receiving images during chatting. [29 Mar 2023]
  6. [Florence-2](https://arxiv.org/abs/2311.06242): Advancing a unified representation for various vision tasks, demonstrating specialized models like `CLIP` for classification, `GroundingDINO` for object detection, and `SAM` for segmentation. [ref](https://huggingface.co/microsoft/Florence-2-large) [10 Nov 2023]
- Google
  1. [Gemini 1.5](https://blog.google/technology/ai/google-gemini-next-generation-model-february-2024): 1 million token context window, 1 hour of video, 11 hours of audio, codebases with over 30,000 lines of code or over 700,000 words. [Feb 2024]
  1. [Foundation Models](https://ai.google/discover/our-models/): Gemini, Veo, Gemma etc.
- Anthrophic
  1. [Claude 3 Opus](https://www.anthropic.com/news/claude-3-family), the largest version of the new LLM, outperforms rivals GPT-4 and Google’s Gemini 1.0 Ultra. Three variants: Opus, Sonnet, and Haiku. [Mar 2024]
- Apple
  1. [4M-21](https://arxiv.org/abs/2406.09406): An Any-to-Any Vision Model for Tens of Tasks and Modalities. [13 Jun 2024]
- Benchmarking Multimodal LLMs.
  - LLaVA-1.5 achieves SoTA on a broad range of 11 tasks incl. SEED-Bench.
  - [SEED-Bench](https://arxiv.org/abs/2307.16125): [[cnt](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=arxiv%3A+2307.16125)]: Benchmarking Multimodal LLMs [git](https://github.com/AILab-CVC/SEED-Bench) [30 Jul 2023]
- [Molmo and PixMo](https://arxiv.org/abs/2409.17146): Open Weights and Open Data for State-of-the-Art Multimodal Models [ref](https://molmo.allenai.org/) [25 Sep 2024]

    <!-- <img src="../files/multi-llm.png" width="180" /> -->

- Optimizing Memory Usage for Training LLMs and Vision Transformers: When applying 10 techniques to a vision transformer, we reduced the memory consumption 20x on a single GPU. [ref](https://lightning.ai/pages/community/tutorial/pytorch-memory-vit-llm/) / [git](https://github.com/rasbt/pytorch-memory-optim) [2 Jul 2023]

### **Generative AI Landscape**

- [The Generative AI Revolution: Exploring the Current Landscape](https://pub.towardsai.net/the-generative-ai-revolution-exploring-the-current-landscape-4b89998fcc5f) : [doc](../files/gen-ai-landscape.pdf) [28 Jun 2023]
- [Diffusion Models vs. GANs vs. VAEs: Comparison of Deep Generative Models](https://pub.towardsai.net/diffusion-models-vs-gans-vs-vaes-comparison-of-deep-generative-models-67ab93e0d9ae) [12 May 2023]

| Model | Description | Strengths | Weaknesses |
| --- | --- | --- | --- |
| GANs | Two neural networks, a generator and a discriminator, work together. The generator creates synthetic samples, and the discriminator distinguishes between real and generated samples. | Unsupervised learning, able to mimic data distributions without labeled data, and are versatile in applications like image synthesis, super-resolution, and style transfer | Known for potentially unstable training and less diversity in generation. |
| VAEs | Consists of an encoder and a decoder. The encoder maps input data into a low-dimensional representation, and the decoder reconstructs the original input data from this representation. e.g, `DALLE` | Efficient at learning latent representations and can be used for tasks like data denoising and anomaly detection, in addition to data generation. | Dependent on an approximate loss function. |
| Diffusion Models | Consists of forward and reverse diffusion processes. Forward diffusion adds noise to input data until white noise is obtained. The reverse diffusion process removes the noise to recover the original data.  e.g, `Stable Diffusion` | Capable of producing high-quality, step-by-step samples. | Multi-step (often 1000) generation process. |
