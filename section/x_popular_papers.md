
## AI Agents

*Retrieved: 2025-12-03 11:28:55*

### 1. Mind2Web: Towards a Generalist Agent for the Web

**Authors:** Xiang Deng, Yu Gu, Boyuan Zheng, et al.  
**Year:** 2023 | **Citations:** 725 | **Venue:** Neural Information Processing Systems  
**Year Month:** [Jun 2023]  
**Fields:** Computer Science  
**URL:** [https://www.semanticscholar.org/paper/58f8925a8b87054ad0635a6398a7fe24935b1604](https://www.semanticscholar.org/paper/58f8925a8b87054ad0635a6398a7fe24935b1604)  
**arXiv:** [https://arxiv.org/abs/2306.06070](https://arxiv.org/abs/2306.06070)  

**Abstract:** We introduce Mind2Web, the first dataset for developing and evaluating generalist agents for the web that can follow language instructions to complete complex tasks on any website. Existing datasets for web agents either use simulated websites or only cover a limited set of websites and tasks, thus not suitable for generalist web agents. With over 2,000 open-ended tasks collected from 137 websites spanning 31 domains and crowdsourced action sequences for the tasks, Mind2Web provides three necessary ingredients for building generalist web agents: 1) diverse domains, websites, and tasks, 2) use of real-world websites instead of simulated and simplified ones, and 3) a broad spectrum of user interaction patterns. Based on Mind2Web, we conduct an initial exploration of using large language models (LLMs) for building generalist web agents. While the raw HTML of real-world websites are often too large to be fed to LLMs, we show that first filtering it with a small LM significantly improves the effectiveness and efficiency of LLMs. Our solution demonstrates a decent level of performance, even on websites or entire domains the model has never seen before, but there is still a substantial room to improve towards truly generalizable agents. We open-source our dataset, model implementation, and trained models (https://osu-nlp-group.github.io/Mind2Web) to facilitate further research on building a generalist agent for the web.

---

### 2. SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering

**Authors:** John Yang, Carlos E. Jimenez, Alexander Wettig, et al.  
**Year:** 2024 | **Citations:** 569 | **Venue:** Neural Information Processing Systems  
**Year Month:** [May 2024]  
**Fields:** Computer Science  
**URL:** [https://www.semanticscholar.org/paper/1c3c531fc0fbe79f97f367ed3648de8467caeeaa](https://www.semanticscholar.org/paper/1c3c531fc0fbe79f97f367ed3648de8467caeeaa)  
**arXiv:** [https://arxiv.org/abs/2405.15793](https://arxiv.org/abs/2405.15793)  

**Abstract:** Language model (LM) agents are increasingly being used to automate complicated tasks in digital environments. Just as humans benefit from powerful software applications, such as integrated development environments, for complex tasks like software engineering, we posit that LM agents represent a new category of end users with their own needs and abilities, and would benefit from specially-built interfaces to the software they use. We investigate how interface design affects the performance of language model agents. As a result of this exploration, we introduce SWE-agent: a system that facilitates LM agents to autonomously use computers to solve software engineering tasks. SWE-agent's custom agent-computer interface (ACI) significantly enhances an agent's ability to create and edit code files, navigate entire repositories, and execute tests and other programs. We evaluate SWE-agent on SWE-bench and HumanEvalFix, achieving state-of-the-art performance on both with a pass@1 rate of 12.5% and 87.7%, respectively, far exceeding the previous state-of-the-art achieved with non-interactive LMs. Finally, we provide insight on how the design of the ACI can impact agents' behavior and performance.

---

### 3. Multi-Agent Collaboration: Harnessing the Power of Intelligent LLM Agents

**Authors:** Yashar Talebirad, Amirhossein Nadiri  
**Year:** 2023 | **Citations:** 340 | **Venue:** arXiv.org  
**Year Month:** [Jun 2023]  
**Fields:** Computer Science  
**URL:** [https://www.semanticscholar.org/paper/ead6121fbc787d508dc6a6d7106f72bf0d647d03](https://www.semanticscholar.org/paper/ead6121fbc787d508dc6a6d7106f72bf0d647d03)  
**arXiv:** [https://arxiv.org/abs/2306.03314](https://arxiv.org/abs/2306.03314)  

**Abstract:** In this paper, we present a novel framework for enhancing the capabilities of large language models (LLMs) by leveraging the power of multi-agent systems. Our framework introduces a collaborative environment where multiple intelligent agent components, each with distinctive attributes and roles, work together to handle complex tasks more efficiently and effectively. We demonstrate the practicality and versatility of our framework through case studies in artificial general intelligence (AGI), specifically focusing on the Auto-GPT and BabyAGI models. We also examine the"Gorilla"model, which integrates external APIs into the LLM. Our framework addresses limitations and challenges such as looping issues, security risks, scalability, system evaluation, and ethical considerations. By modeling various domains such as courtroom simulations and software development scenarios, we showcase the potential applications and benefits of our proposed multi-agent system. Our framework provides an avenue for advancing the capabilities and performance of LLMs through collaboration and knowledge exchange among intelligent agents.

---

### 4. Understanding the planning of LLM agents: A survey

**Authors:** Xu Huang, Weiwen Liu, Xiaolong Chen, et al.  
**Year:** 2024 | **Citations:** 321 | **Venue:** arXiv.org  
**Year Month:** [Feb 2024]  
**Fields:** Computer Science  
**URL:** [https://www.semanticscholar.org/paper/7e281e8ab380affd3c5724feae038274df378511](https://www.semanticscholar.org/paper/7e281e8ab380affd3c5724feae038274df378511)  
**arXiv:** [https://arxiv.org/abs/2402.02716](https://arxiv.org/abs/2402.02716)  

**Abstract:** As Large Language Models (LLMs) have shown significant intelligence, the progress to leverage LLMs as planning modules of autonomous agents has attracted more attention. This survey provides the first systematic view of LLM-based agents planning, covering recent works aiming to improve planning ability. We provide a taxonomy of existing works on LLM-Agent planning, which can be categorized into Task Decomposition, Plan Selection, External Module, Reflection and Memory. Comprehensive analyses are conducted for each direction, and further challenges for the field of research are discussed.

---

### 5. ExpeL: LLM Agents Are Experiential Learners

**Authors:** Andrew Zhao, Daniel Huang, Quentin Xu, et al.  
**Year:** 2023 | **Citations:** 317 | **Venue:** AAAI Conference on Artificial Intelligence  
**Year Month:** [Aug 2023]  
**Fields:** Computer Science  
**URL:** [https://www.semanticscholar.org/paper/5e4597eb21a393b23e473cf66cb5ae8b27cab03e](https://www.semanticscholar.org/paper/5e4597eb21a393b23e473cf66cb5ae8b27cab03e)  
**arXiv:** [https://arxiv.org/abs/2308.10144](https://arxiv.org/abs/2308.10144)  

**Abstract:** The recent surge in research interest in applying large language models (LLMs) to decision-making tasks has flourished by leveraging the extensive world knowledge embedded in LLMs. While there is a growing demand to tailor LLMs for custom decision-making tasks, finetuning them for specific tasks is resource-intensive and may diminish the model's generalization capabilities. Moreover, state-of-the-art language models like GPT-4 and Claude are primarily accessible through API calls, with their parametric weights remaining proprietary and unavailable to the public. This scenario emphasizes the growing need for new methodologies that allow learning from agent experiences without requiring parametric updates. To address these problems, we introduce the Experiential Learning (ExpeL) agent. Our agent autonomously gathers experiences and extracts knowledge using natural language from a collection of training tasks. At inference, the agent recalls its extracted insights and past experiences to make informed decisions. Our empirical results highlight the robust learning efficacy of the ExpeL agent, indicating a consistent enhancement in its performance as it accumulates experiences. We further explore the emerging capabilities and transfer learning potential of the ExpeL agent through qualitative observations and additional experiments.

---

### 6. Language Agent Tree Search Unifies Reasoning Acting and Planning in Language Models

**Authors:** Andy Zhou, Kai Yan, Michal Shlapentokh-Rothman, et al.  
**Year:** 2023 | **Citations:** 301 | **Venue:** International Conference on Machine Learning  
**Year Month:** [Oct 2023]  
**Fields:** Computer Science  
**URL:** [https://www.semanticscholar.org/paper/700bd9681f1b9e9e2212e10415d27b11c7e6836b](https://www.semanticscholar.org/paper/700bd9681f1b9e9e2212e10415d27b11c7e6836b)  
**arXiv:** [https://arxiv.org/abs/2310.04406](https://arxiv.org/abs/2310.04406)  

**Abstract:** While language models (LMs) have shown potential across a range of decision-making tasks, their reliance on simple acting processes limits their broad deployment as autonomous agents. In this paper, we introduce Language Agent Tree Search (LATS) -- the first general framework that synergizes the capabilities of LMs in reasoning, acting, and planning. By leveraging the in-context learning ability of LMs, we integrate Monte Carlo Tree Search into LATS to enable LMs as agents, along with LM-powered value functions and self-reflections for proficient exploration and enhanced decision-making. A key feature of our approach is the incorporation of an environment for external feedback, which offers a more deliberate and adaptive problem-solving mechanism that surpasses the constraints of existing techniques. Our experimental evaluation across diverse domains, including programming, interactive question-answering (QA), web navigation, and math, validates the effectiveness and generality of LATS in decision-making while maintaining competitive or improved reasoning performance. Notably, LATS achieves state-of-the-art pass@1 accuracy (92.7%) for programming on HumanEval with GPT-4 and demonstrates gradient-free performance (average score of 75.9) comparable to gradient-based fine-tuning for web navigation on WebShop with GPT-3.5. Code can be found at https://github.com/lapisrocks/LanguageAgentTreeSearch

---

### 7. Pre-Trained Language Models for Interactive Decision-Making

**Authors:** Shuang Li, Xavier Puig, Yilun Du, et al.  
**Year:** 2022 | **Citations:** 299 | **Venue:** Neural Information Processing Systems  
**Year Month:** [Feb 2022]  
**Fields:** Computer Science  
**URL:** [https://www.semanticscholar.org/paper/b9b220b485d2add79118ffdc2aaa148b67fa53ef](https://www.semanticscholar.org/paper/b9b220b485d2add79118ffdc2aaa148b67fa53ef)  
**arXiv:** [https://arxiv.org/abs/2202.01771](https://arxiv.org/abs/2202.01771)  

**Abstract:** Language model (LM) pre-training is useful in many language processing tasks. But can pre-trained LMs be further leveraged for more general machine learning problems? We propose an approach for using LMs to scaffold learning and generalization in general sequential decision-making problems. In this approach, goals and observations are represented as a sequence of embeddings, and a policy network initialized with a pre-trained LM predicts the next action. We demonstrate that this framework enables effective combinatorial generalization across different environments and supervisory modalities. We begin by assuming access to a set of expert demonstrations, and show that initializing policies with LMs and fine-tuning them via behavior cloning improves task completion rates by 43.6% in the VirtualHome environment. Next, we integrate an active data gathering procedure in which agents iteratively interact with the environment, relabel past"failed"experiences with new goals, and update their policies in a self-supervised loop. Active data gathering further improves combinatorial generalization, outperforming the best baseline by 25.1%. Finally, we explain these results by investigating three possible factors underlying the effectiveness of the LM-based policy. We find that sequential input representations (vs. fixed-dimensional feature vectors) and LM-based weight initialization are both important for generalization. Surprisingly, however, the format of the policy inputs encoding (e.g. as a natural language string vs. an arbitrary sequential encoding) has little influence. Together, these results suggest that language modeling induces representations that are useful for modeling not just language, but also goals and plans; these representations can aid learning and generalization even outside of language processing.

---

### 8. Executable Code Actions Elicit Better LLM Agents

**Authors:** Xingyao Wang, Yangyi Chen, Lifan Yuan, et al.  
**Year:** 2024 | **Citations:** 282 | **Venue:** International Conference on Machine Learning  
**Year Month:** [Feb 2024]  
**Fields:** Computer Science  
**URL:** [https://www.semanticscholar.org/paper/78fbb6e7a1c568a04e8c935aa9909d0c942ea5f6](https://www.semanticscholar.org/paper/78fbb6e7a1c568a04e8c935aa9909d0c942ea5f6)  
**arXiv:** [https://arxiv.org/abs/2402.01030](https://arxiv.org/abs/2402.01030)  

**Abstract:** Large Language Model (LLM) agents, capable of performing a broad range of actions, such as invoking tools and controlling robots, show great potential in tackling real-world challenges. LLM agents are typically prompted to produce actions by generating JSON or text in a pre-defined format, which is usually limited by constrained action space (e.g., the scope of pre-defined tools) and restricted flexibility (e.g., inability to compose multiple tools). This work proposes to use executable Python code to consolidate LLM agents' actions into a unified action space (CodeAct). Integrated with a Python interpreter, CodeAct can execute code actions and dynamically revise prior actions or emit new actions upon new observations through multi-turn interactions. Our extensive analysis of 17 LLMs on API-Bank and a newly curated benchmark shows that CodeAct outperforms widely used alternatives (up to 20% higher success rate). The encouraging performance of CodeAct motivates us to build an open-source LLM agent that interacts with environments by executing interpretable code and collaborates with users using natural language. To this end, we collect an instruction-tuning dataset CodeActInstruct that consists of 7k multi-turn interactions using CodeAct. We show that it can be used with existing data to improve models in agent-oriented tasks without compromising their general capability. CodeActAgent, finetuned from Llama2 and Mistral, is integrated with Python interpreter and uniquely tailored to perform sophisticated tasks (e.g., model training) using existing libraries and autonomously self-debug.

---

### 9. OpenHands: An Open Platform for AI Software Developers as Generalist Agents

**Authors:** Xingyao Wang, Boxuan Li, Yufan Song, et al.  
**Year:** 2024 | **Citations:** 281 | **Venue:** International Conference on Learning Representations  
**Year Month:** [Jul 2024]  
**Fields:** Computer Science  
**URL:** [https://www.semanticscholar.org/paper/1d07e5b6f978cf69c0186f3d5f434fa92d471e46](https://www.semanticscholar.org/paper/1d07e5b6f978cf69c0186f3d5f434fa92d471e46)  
**arXiv:** [https://arxiv.org/abs/2407.16741](https://arxiv.org/abs/2407.16741)  

**Abstract:** Software is one of the most powerful tools that we humans have at our disposal; it allows a skilled programmer to interact with the world in complex and profound ways. At the same time, thanks to improvements in large language models (LLMs), there has also been a rapid development in AI agents that interact with and affect change in their surrounding environments. In this paper, we introduce OpenHands (f.k.a. OpenDevin), a platform for the development of powerful and flexible AI agents that interact with the world in similar ways to those of a human developer: by writing code, interacting with a command line, and browsing the web. We describe how the platform allows for the implementation of new agents, safe interaction with sandboxed environments for code execution, coordination between multiple agents, and incorporation of evaluation benchmarks. Based on our currently incorporated benchmarks, we perform an evaluation of agents over 15 challenging tasks, including software engineering (e.g., SWE-BENCH) and web browsing (e.g., WEBARENA), among others. Released under the permissive MIT license, OpenHands is a community project spanning academia and industry with more than 2.1K contributions from over 188 contributors.

---

### 10. Personal LLM Agents: Insights and Survey about the Capability, Efficiency and Security

**Authors:** Yuanchun Li, Hao Wen, Weijun Wang, et al.  
**Year:** 2024 | **Citations:** 252 | **Venue:** arXiv.org  
**Year Month:** [Jan 2024]  
**Fields:** Computer Science  
**URL:** [https://www.semanticscholar.org/paper/06d860a5bbb99a4eafdbbb2d5f6aa8dd5fd32cf4](https://www.semanticscholar.org/paper/06d860a5bbb99a4eafdbbb2d5f6aa8dd5fd32cf4)  
**arXiv:** [https://arxiv.org/abs/2401.05459](https://arxiv.org/abs/2401.05459)  

**Abstract:** Since the advent of personal computing devices, intelligent personal assistants (IPAs) have been one of the key technologies that researchers and engineers have focused on, aiming to help users efficiently obtain information and execute tasks, and provide users with more intelligent, convenient, and rich interaction experiences. With the development of smartphones and IoT, computing and sensing devices have become ubiquitous, greatly expanding the boundaries of IPAs. However, due to the lack of capabilities such as user intent understanding, task planning, tool using, and personal data management etc., existing IPAs still have limited practicality and scalability. Recently, the emergence of foundation models, represented by large language models (LLMs), brings new opportunities for the development of IPAs. With the powerful semantic understanding and reasoning capabilities, LLM can enable intelligent agents to solve complex problems autonomously. In this paper, we focus on Personal LLM Agents, which are LLM-based agents that are deeply integrated with personal data and personal devices and used for personal assistance. We envision that Personal LLM Agents will become a major software paradigm for end-users in the upcoming era. To realize this vision, we take the first step to discuss several important questions about Personal LLM Agents, including their architecture, capability, efficiency and security. We start by summarizing the key components and design choices in the architecture of Personal LLM Agents, followed by an in-depth analysis of the opinions collected from domain experts. Next, we discuss several key challenges to achieve intelligent, efficient and secure Personal LLM Agents, followed by a comprehensive survey of representative solutions to address these challenges.

---

### 11. Agentless: Demystifying LLM-based Software Engineering Agents

**Authors:** Chun Xia, Yinlin Deng, Soren Dunn, et al.  
**Year:** 2024 | **Citations:** 215 | **Venue:** arXiv.org  
**Year Month:** [Jul 2024]  
**Fields:** Computer Science  
**URL:** [https://www.semanticscholar.org/paper/ae50c8e255ba55bdbbb05ea470aa63437534438e](https://www.semanticscholar.org/paper/ae50c8e255ba55bdbbb05ea470aa63437534438e)  
**arXiv:** [https://arxiv.org/abs/2407.01489](https://arxiv.org/abs/2407.01489)  

**Abstract:** Recent advancements in large language models (LLMs) have significantly advanced the automation of software development tasks, including code synthesis, program repair, and test generation. More recently, researchers and industry practitioners have developed various autonomous LLM agents to perform end-to-end software development tasks. These agents are equipped with the ability to use tools, run commands, observe feedback from the environment, and plan for future actions. However, the complexity of these agent-based approaches, together with the limited abilities of current LLMs, raises the following question: Do we really have to employ complex autonomous software agents? To attempt to answer this question, we build Agentless -- an agentless approach to automatically solve software development problems. Compared to the verbose and complex setup of agent-based approaches, Agentless employs a simplistic three-phase process of localization, repair, and patch validation, without letting the LLM decide future actions or operate with complex tools. Our results on the popular SWE-bench Lite benchmark show that surprisingly the simplistic Agentless is able to achieve both the highest performance (32.00%, 96 correct fixes) and low cost ($0.70) compared with all existing open-source software agents! Furthermore, we manually classified the problems in SWE-bench Lite and found problems with exact ground truth patch or insufficient/misleading issue descriptions. As such, we construct SWE-bench Lite-S by excluding such problematic issues to perform more rigorous evaluation and comparison. Our work highlights the current overlooked potential of a simple, interpretable technique in autonomous software development. We hope Agentless will help reset the baseline, starting point, and horizon for autonomous software agents, and inspire future work along this crucial direction.

---

### 12. Exploring Collaboration Mechanisms for LLM Agents: A Social Psychology View

**Authors:** Jintian Zhang, Xin Xu, Ruibo Liu, et al.  
**Year:** 2023 | **Citations:** 212 | **Venue:** arXiv.org  
**Year Month:** [Oct 2023]  
**Fields:** Computer Science  
**URL:** [https://www.semanticscholar.org/paper/9fcdbfdf28245010c875ce85502351fe05c04b49](https://www.semanticscholar.org/paper/9fcdbfdf28245010c875ce85502351fe05c04b49)  
**arXiv:** [https://arxiv.org/abs/2310.02124](https://arxiv.org/abs/2310.02124)  

**Abstract:** As Natural Language Processing (NLP) systems are increasingly employed in intricate social environments, a pressing query emerges: Can these NLP systems mirror human-esque collaborative intelligence, in a multi-agent society consisting of multiple large language models (LLMs)? This paper probes the collaboration mechanisms among contemporary NLP systems by melding practical experiments with theoretical insights. We fabricate four unique `societies' comprised of LLM agents, where each agent is characterized by a specific `trait' (easy-going or overconfident) and engages in collaboration with a distinct `thinking pattern' (debate or reflection). Through evaluating these multi-agent societies on three benchmark datasets, we discern that certain collaborative strategies not only outshine previous top-tier approaches, but also optimize efficiency (using fewer API tokens). Moreover, our results further illustrate that LLM agents manifest human-like social behaviors, such as conformity and consensus reaching, mirroring foundational social psychology theories. In conclusion, we integrate insights from social psychology to contextualize the collaboration of LLM agents, inspiring further investigations into the collaboration mechanism for LLMs. We commit to sharing our code and datasets\footnote{\url{https://github.com/zjunlp/MachineSoM}.}, hoping to catalyze further research in this promising avenue.

---

### 13. Empowering biomedical discovery with AI agents

**Authors:** Shanghua Gao, Ada Fang, Yepeng Huang, et al.  
**Year:** 2024 | **Citations:** 182 | **Venue:** Cell  
**Year Month:** [Apr 2024]  
**Fields:** Medicine, Computer Science  
**URL:** [https://www.semanticscholar.org/paper/8cedeb11139eab187e43414fd7097c5d578dad7c](https://www.semanticscholar.org/paper/8cedeb11139eab187e43414fd7097c5d578dad7c)  
**arXiv:** [https://arxiv.org/abs/2404.02831](https://arxiv.org/abs/2404.02831)  
---

### 14. Identifying the Risks of LM Agents with an LM-Emulated Sandbox

**Authors:** Yangjun Ruan, Honghua Dong, Andrew Wang, et al.  
**Year:** 2023 | **Citations:** 178 | **Venue:** International Conference on Learning Representations  
**Year Month:** [Sep 2023]  
**Fields:** Computer Science  
**URL:** [https://www.semanticscholar.org/paper/0a893657e00fe8ecfadcc65c63bf293e70cb1564](https://www.semanticscholar.org/paper/0a893657e00fe8ecfadcc65c63bf293e70cb1564)  
**arXiv:** [https://arxiv.org/abs/2309.15817](https://arxiv.org/abs/2309.15817)  

**Abstract:** Recent advances in Language Model (LM) agents and tool use, exemplified by applications like ChatGPT Plugins, enable a rich set of capabilities but also amplify potential risks - such as leaking private data or causing financial losses. Identifying these risks is labor-intensive, necessitating implementing the tools, setting up the environment for each test scenario manually, and finding risky cases. As tools and agents become more complex, the high cost of testing these agents will make it increasingly difficult to find high-stakes, long-tailed risks. To address these challenges, we introduce ToolEmu: a framework that uses an LM to emulate tool execution and enables the testing of LM agents against a diverse range of tools and scenarios, without manual instantiation. Alongside the emulator, we develop an LM-based automatic safety evaluator that examines agent failures and quantifies associated risks. We test both the tool emulator and evaluator through human evaluation and find that 68.8% of failures identified with ToolEmu would be valid real-world agent failures. Using our curated initial benchmark consisting of 36 high-stakes tools and 144 test cases, we provide a quantitative risk analysis of current LM agents and identify numerous failures with potentially severe outcomes. Notably, even the safest LM agent exhibits such failures 23.9% of the time according to our evaluator, underscoring the need to develop safer LM agents for real-world deployment.

---

### 15. Language Models as Agent Models

**Authors:** Jacob Andreas  
**Year:** 2022 | **Citations:** 163 | **Venue:** Conference on Empirical Methods in Natural Language Processing  
**Year Month:** [Dec 2022]  
**Fields:** Computer Science  
**URL:** [https://www.semanticscholar.org/paper/4596139b28c3ceacbd7e3c34dc0df079dbf4e96b](https://www.semanticscholar.org/paper/4596139b28c3ceacbd7e3c34dc0df079dbf4e96b)  
**arXiv:** [https://arxiv.org/abs/2212.01681](https://arxiv.org/abs/2212.01681)  

**Abstract:** Language models (LMs) are trained on collections of documents, written by individual human agents to achieve specific goals in an outside world. During training, LMs have access only to text of these documents, with no direct evidence of the internal states of the agents that produced them -- a fact often used to argue that LMs are incapable of modeling goal-directed aspects of human language production and comprehension. Can LMs trained on text learn anything at all about the relationship between language and use? I argue that LMs are models of intentional communication in a specific, narrow sense. When performing next word prediction given a textual context, an LM can infer and represent properties of an agent likely to have produced that context. These representations can in turn influence subsequent LM generation in the same way that agents' communicative intentions influence their language. I survey findings from the recent literature showing that -- even in today's non-robust and error-prone models -- LMs infer and use representations of fine-grained communicative intentions and more abstract beliefs and goals. Despite the limited nature of their training data, they can thus serve as building blocks for systems that communicate and act intentionally.

---

### 16. Do the Rewards Justify the Means? Measuring Trade-Offs Between Rewards and Ethical Behavior in the MACHIAVELLI Benchmark

**Authors:** Alexander Pan, C. Shern, Andy Zou, et al.  
**Year:** 2023 | **Citations:** 162 | **Venue:** International Conference on Machine Learning  
**Year Month:** [Apr 2023]  
**Fields:** Computer Science  
**URL:** [https://www.semanticscholar.org/paper/5da2d404d789aeff266b63a760d07fe8bc31ba23](https://www.semanticscholar.org/paper/5da2d404d789aeff266b63a760d07fe8bc31ba23)  
**arXiv:** [https://arxiv.org/abs/2304.03279](https://arxiv.org/abs/2304.03279)  

**Abstract:** Artificial agents have traditionally been trained to maximize reward, which may incentivize power-seeking and deception, analogous to how next-token prediction in language models (LMs) may incentivize toxicity. So do agents naturally learn to be Machiavellian? And how do we measure these behaviors in general-purpose models such as GPT-4? Towards answering these questions, we introduce MACHIAVELLI, a benchmark of 134 Choose-Your-Own-Adventure games containing over half a million rich, diverse scenarios that center on social decision-making. Scenario labeling is automated with LMs, which are more performant than human annotators. We mathematize dozens of harmful behaviors and use our annotations to evaluate agents' tendencies to be power-seeking, cause disutility, and commit ethical violations. We observe some tension between maximizing reward and behaving ethically. To improve this trade-off, we investigate LM-based methods to steer agents' towards less harmful behaviors. Our results show that agents can both act competently and morally, so concrete progress can currently be made in machine ethics--designing agents that are Pareto improvements in both safety and capabilities.

---

### 17. Evaluating Very Long-Term Conversational Memory of LLM Agents

**Authors:** Adyasha Maharana, Dong-Ho Lee, S. Tulyakov, et al.  
**Year:** 2024 | **Citations:** 160 | **Venue:** Annual Meeting of the Association for Computational Linguistics  
**Year Month:** [Feb 2024]  
**Fields:** Computer Science  
**URL:** [https://www.semanticscholar.org/paper/0bf3a1867f7245b8a702093901c66b08b518eafc](https://www.semanticscholar.org/paper/0bf3a1867f7245b8a702093901c66b08b518eafc)  
**arXiv:** [https://arxiv.org/abs/2402.17753](https://arxiv.org/abs/2402.17753)  

**Abstract:** Existing works on long-term open-domain dialogues focus on evaluating model responses within contexts spanning no more than five chat sessions. Despite advancements in long-context large language models (LLMs) and retrieval augmented generation (RAG) techniques, their efficacy in very long-term dialogues remains unexplored. To address this research gap, we introduce a machine-human pipeline to generate high-quality, very long-term dialogues by leveraging LLM-based agent architectures and grounding their dialogues on personas and temporal event graphs. Moreover, we equip each agent with the capability of sharing and reacting to images. The generated conversations are verified and edited by human annotators for long-range consistency and grounding to the event graphs. Using this pipeline, we collect LoCoMo, a dataset of very long-term conversations, each encompassing 300 turns and 9K tokens on avg., over up to 35 sessions. Based on LoCoMo, we present a comprehensive evaluation benchmark to measure long-term memory in models, encompassing question answering, event summarization, and multi-modal dialogue generation tasks. Our experimental results indicate that LLMs exhibit challenges in understanding lengthy conversations and comprehending long-range temporal and causal dynamics within dialogues. Employing strategies like long-context LLMs or RAG can offer improvements but these models still substantially lag behind human performance.

---

### 18. AgentPoison: Red-teaming LLM Agents via Poisoning Memory or Knowledge Bases

**Authors:** Zhaorun Chen, Zhen Xiang, Chaowei Xiao, et al.  
**Year:** 2024 | **Citations:** 160 | **Venue:** Neural Information Processing Systems  
**Year Month:** [Jul 2024]  
**Fields:** Computer Science  
**URL:** [https://www.semanticscholar.org/paper/b6948a9e8b3eec5a56a80c69727154fcd7ececce](https://www.semanticscholar.org/paper/b6948a9e8b3eec5a56a80c69727154fcd7ececce)  
**arXiv:** [https://arxiv.org/abs/2407.12784](https://arxiv.org/abs/2407.12784)  

**Abstract:** LLM agents have demonstrated remarkable performance across various applications, primarily due to their advanced capabilities in reasoning, utilizing external knowledge and tools, calling APIs, and executing actions to interact with environments. Current agents typically utilize a memory module or a retrieval-augmented generation (RAG) mechanism, retrieving past knowledge and instances with similar embeddings from knowledge bases to inform task planning and execution. However, the reliance on unverified knowledge bases raises significant concerns about their safety and trustworthiness. To uncover such vulnerabilities, we propose a novel red teaming approach AgentPoison, the first backdoor attack targeting generic and RAG-based LLM agents by poisoning their long-term memory or RAG knowledge base. In particular, we form the trigger generation process as a constrained optimization to optimize backdoor triggers by mapping the triggered instances to a unique embedding space, so as to ensure that whenever a user instruction contains the optimized backdoor trigger, the malicious demonstrations are retrieved from the poisoned memory or knowledge base with high probability. In the meantime, benign instructions without the trigger will still maintain normal performance. Unlike conventional backdoor attacks, AgentPoison requires no additional model training or fine-tuning, and the optimized backdoor trigger exhibits superior transferability, in-context coherence, and stealthiness. Extensive experiments demonstrate AgentPoison's effectiveness in attacking three types of real-world LLM agents: RAG-based autonomous driving agent, knowledge-intensive QA agent, and healthcare EHRAgent. On each agent, AgentPoison achieves an average attack success rate higher than 80% with minimal impact on benign performance (less than 1%) with a poison rate less than 0.1%.

---

### 19. Agent Laboratory: Using LLM Agents as Research Assistants

**Authors:** Samuel Schmidgall, Yusheng Su, Ze Wang, et al.  
**Year:** 2025 | **Citations:** 160 | **Venue:** Conference on Empirical Methods in Natural Language Processing  
**Year Month:** [Jan 2025]  
**Fields:** Computer Science  
**URL:** [https://www.semanticscholar.org/paper/394924896e24c9b086d96d0958dae07f54ff9452](https://www.semanticscholar.org/paper/394924896e24c9b086d96d0958dae07f54ff9452)  
**arXiv:** [https://arxiv.org/abs/2501.04227](https://arxiv.org/abs/2501.04227)  

**Abstract:** Historically, scientific discovery has been a lengthy and costly process, demanding substantial time and resources from initial conception to final results. To accelerate scientific discovery, reduce research costs, and improve research quality, we introduce Agent Laboratory, an autonomous LLM-based framework capable of completing the entire research process. This framework accepts a human-provided research idea and progresses through three stages--literature review, experimentation, and report writing to produce comprehensive research outputs, including a code repository and a research report, while enabling users to provide feedback and guidance at each stage. We deploy Agent Laboratory with various state-of-the-art LLMs and invite multiple researchers to assess its quality by participating in a survey, providing human feedback to guide the research process, and then evaluate the final paper. We found that: (1) Agent Laboratory driven by o1-preview generates the best research outcomes; (2) The generated machine learning code is able to achieve state-of-the-art performance compared to existing methods; (3) Human involvement, providing feedback at each stage, significantly improves the overall quality of research; (4) Agent Laboratory significantly reduces research expenses, achieving an 84% decrease compared to previous autonomous research methods. We hope Agent Laboratory enables researchers to allocate more effort toward creative ideation rather than low-level coding and writing, ultimately accelerating scientific discovery.

---

### 20. TPTU: Task Planning and Tool Usage of Large Language Model-based AI Agents

**Authors:** Jingqing Ruan, Yihong Chen, Bin Zhang, et al.  
**Year:** 2023 | **Citations:** 158 | **Venue:** arXiv.org  
**Fields:** Computer Science  
**URL:** [https://www.semanticscholar.org/paper/5ce94181ea702f69c3651dce721d6bd8026b8106](https://www.semanticscholar.org/paper/5ce94181ea702f69c3651dce721d6bd8026b8106)  
---

### 21. Bots with Feelings: Should AI Agents Express Positive Emotion in Customer Service?

**Authors:** Elizabeth Han, Dezhi Yin, Han Zhang  
**Year:** 2022 | **Citations:** 150 | **Venue:** Information systems research  
**Fields:** Computer Science  
**URL:** [https://www.semanticscholar.org/paper/a79707c7646debe27f1a5188996237f11763592c](https://www.semanticscholar.org/paper/a79707c7646debe27f1a5188996237f11763592c)  

**Abstract:** The rise of emotional intelligence technology and the recent debate about the possibility of a “sentient” artificial intelligence (AI) urge the need to study the role of emotion during people’s interactions with AIs. In customer service, human employees are increasingly replaced by AI agents, such as chatbots, and often these AI agents are equipped with emotion-expressing capabilities to replicate the positive impact of human-expressed positive emotion. But is it indeed beneficial? This research explores how, when, and why an AI agent’s expression of positive emotion affects customers’ service evaluations. Through controlled experiments in which the subjects interacted with a service agent (AI or human) to resolve a hypothetical service issue, we provide answers to these questions. We show that AI-expressed positive emotion can influence customers affectively (by evoking customers’ positive emotions) and cognitively (by violating customers’ expectations) in opposite directions. Thus, positive emotion expressed by an AI agent (versus a human employee) is less effective in facilitating service evaluations. We further underscore that, depending on customers’ expectations toward their relationship with a service agent, AI-expressed positive emotion may enhance or hurt service evaluations. Overall, our work provides useful guidance on how and when companies can best deploy emotion-expressing AI agents.

---

### 22. A-MEM: Agentic Memory for LLM Agents

**Authors:** Wujiang Xu, Zujie Liang, Kai Mei, et al.  
**Year:** 2025 | **Citations:** 135 | **Venue:** arXiv.org  
**Year Month:** [Feb 2025]  
**Fields:** Computer Science  
**URL:** [https://www.semanticscholar.org/paper/1f35a15fe9df43d24ec6ea551ec6c9766c17eccf](https://www.semanticscholar.org/paper/1f35a15fe9df43d24ec6ea551ec6c9766c17eccf)  
**arXiv:** [https://arxiv.org/abs/2502.12110](https://arxiv.org/abs/2502.12110)  

**Abstract:** While large language model (LLM) agents can effectively use external tools for complex real-world tasks, they require memory systems to leverage historical experiences. Current memory systems enable basic storage and retrieval but lack sophisticated memory organization, despite recent attempts to incorporate graph databases. Moreover, these systems'fixed operations and structures limit their adaptability across diverse tasks. To address this limitation, this paper proposes a novel agentic memory system for LLM agents that can dynamically organize memories in an agentic way. Following the basic principles of the Zettelkasten method, we designed our memory system to create interconnected knowledge networks through dynamic indexing and linking. When a new memory is added, we generate a comprehensive note containing multiple structured attributes, including contextual descriptions, keywords, and tags. The system then analyzes historical memories to identify relevant connections, establishing links where meaningful similarities exist. Additionally, this process enables memory evolution - as new memories are integrated, they can trigger updates to the contextual representations and attributes of existing historical memories, allowing the memory network to continuously refine its understanding. Our approach combines the structured organization principles of Zettelkasten with the flexibility of agent-driven decision making, allowing for more adaptive and context-aware memory management. Empirical experiments on six foundation models show superior improvement against existing SOTA baselines. The source code for evaluating performance is available at https://github.com/WujiangXu/A-mem, while the source code of the agentic memory system is available at https://github.com/WujiangXu/A-mem-sys.

---

### 23. Agent Q: Advanced Reasoning and Learning for Autonomous AI Agents

**Authors:** Pranav Putta, Edmund Mills, Naman Garg, et al.  
**Year:** 2024 | **Citations:** 132 | **Venue:** arXiv.org  
**Year Month:** [Aug 2024]  
**Fields:** Computer Science  
**URL:** [https://www.semanticscholar.org/paper/b393f619a87c5b6aa63c7abc7118263205b6aa62](https://www.semanticscholar.org/paper/b393f619a87c5b6aa63c7abc7118263205b6aa62)  
**arXiv:** [https://arxiv.org/abs/2408.07199](https://arxiv.org/abs/2408.07199)  

**Abstract:** Large Language Models (LLMs) have shown remarkable capabilities in natural language tasks requiring complex reasoning, yet their application in agentic, multi-step reasoning within interactive environments remains a difficult challenge. Traditional supervised pre-training on static datasets falls short in enabling autonomous agent capabilities needed to perform complex decision-making in dynamic settings like web navigation. Previous attempts to bridge this ga-through supervised fine-tuning on curated expert demonstrations-often suffer from compounding errors and limited exploration data, resulting in sub-optimal policy outcomes. To overcome these challenges, we propose a framework that combines guided Monte Carlo Tree Search (MCTS) search with a self-critique mechanism and iterative fine-tuning on agent interactions using an off-policy variant of the Direct Preference Optimization (DPO) algorithm. Our method allows LLM agents to learn effectively from both successful and unsuccessful trajectories, thereby improving their generalization in complex, multi-step reasoning tasks. We validate our approach in the WebShop environment-a simulated e-commerce platform where it consistently outperforms behavior cloning and reinforced fine-tuning baseline, and beats average human performance when equipped with the capability to do online search. In real-world booking scenarios, our methodology boosts Llama-3 70B model's zero-shot performance from 18.6% to 81.7% success rate (a 340% relative increase) after a single day of data collection and further to 95.4% with online search. We believe this represents a substantial leap forward in the capabilities of autonomous agents, paving the way for more sophisticated and reliable decision-making in real-world settings.

---

### 24. Mental Models of AI Agents in a Cooperative Game Setting

**Authors:** K. Gero, Zahra Ashktorab, Casey Dugan, et al.  
**Year:** 2020 | **Citations:** 131 | **Venue:** International Conference on Human Factors in Computing Systems  
**Fields:** Computer Science, Psychology  
**URL:** [https://www.semanticscholar.org/paper/a109274aa61679a5d95058b4bd20fa7acba0df52](https://www.semanticscholar.org/paper/a109274aa61679a5d95058b4bd20fa7acba0df52)  

**Abstract:** As more and more forms of AI become prevalent, it becomes increasingly important to understand how people develop mental models of these systems. In this work we study people's mental models of AI in a cooperative word guessing game. We run think-aloud studies in which people play the game with an AI agent; through thematic analysis we identify features of the mental models developed by participants. In a large-scale study we have participants play the game with the AI agent online and use a post-game survey to probe their mental model. We find that those who win more often have better estimates of the AI agent's abilities. We present three components for modeling AI systems, propose that understanding the underlying technology is insufficient for developing appropriate conceptual models (analysis of behavior is also necessary), and suggest future work for studying the revision of mental models over time.

---

### 25. R-Judge: Benchmarking Safety Risk Awareness for LLM Agents

**Authors:** Tongxin Yuan, Zhiwei He, Lingzhong Dong, et al.  
**Year:** 2024 | **Citations:** 131 | **Venue:** Conference on Empirical Methods in Natural Language Processing  
**Year Month:** [Jan 2024]  
**Fields:** Computer Science  
**URL:** [https://www.semanticscholar.org/paper/0e0ea3593dda3039cb93d2ec795a87420006ec08](https://www.semanticscholar.org/paper/0e0ea3593dda3039cb93d2ec795a87420006ec08)  
**arXiv:** [https://arxiv.org/abs/2401.10019](https://arxiv.org/abs/2401.10019)  

**Abstract:** Large language models (LLMs) have exhibited great potential in autonomously completing tasks across real-world applications. Despite this, these LLM agents introduce unexpected safety risks when operating in interactive environments. Instead of centering on the harmlessness of LLM-generated content in most prior studies, this work addresses the imperative need for benchmarking the behavioral safety of LLM agents within diverse environments. We introduce R-Judge, a benchmark crafted to evaluate the proficiency of LLMs in judging and identifying safety risks given agent interaction records. R-Judge comprises 569 records of multi-turn agent interaction, encompassing 27 key risk scenarios among 5 application categories and 10 risk types. It is of high-quality curation with annotated safety labels and risk descriptions. Evaluation of 11 LLMs on R-Judge shows considerable room for enhancing the risk awareness of LLMs: The best-performing model, GPT-4o, achieves 74.42% while no other models significantly exceed the random. Moreover, we reveal that risk awareness in open agent scenarios is a multi-dimensional capability involving knowledge and reasoning, thus challenging for LLMs. With further experiments, we find that fine-tuning on safety judgment significantly improve model performance while straightforward prompting mechanisms fail. R-Judge is publicly available at https://github.com/Lordog/R-Judge.

---

### 26. Trial and Error: Exploration-Based Trajectory Optimization for LLM Agents

**Authors:** Yifan Song, Da Yin, Xiang Yue, et al.  
**Year:** 2024 | **Citations:** 126 | **Venue:** Annual Meeting of the Association for Computational Linguistics  
**Year Month:** [Mar 2024]  
**Fields:** Computer Science  
**URL:** [https://www.semanticscholar.org/paper/f95da5b7be2fac2381eb5dfe26dc7dc5bc2d9a90](https://www.semanticscholar.org/paper/f95da5b7be2fac2381eb5dfe26dc7dc5bc2d9a90)  
**arXiv:** [https://arxiv.org/abs/2403.02502](https://arxiv.org/abs/2403.02502)  

**Abstract:** Large Language Models (LLMs) have become integral components in various autonomous agent systems. In this study, we present an exploration-based trajectory optimization approach, referred to as ETO. This learning method is designed to enhance the performance of open LLM agents. Contrary to previous studies that exclusively train on successful expert trajectories, our method allows agents to learn from their exploration failures. This leads to improved performance through an iterative optimization framework. During the exploration phase, the agent interacts with the environment while completing given tasks, gathering failure trajectories to create contrastive trajectory pairs. In the subsequent training phase, the agent utilizes these trajectory preference pairs to update its policy using contrastive learning methods like DPO. This iterative cycle of exploration and training fosters continued improvement in the agents. Our experiments on three complex tasks demonstrate that ETO consistently surpasses baseline performance by a large margin. Furthermore, an examination of task-solving efficiency and potential in scenarios lacking expert trajectory underscores the effectiveness of our approach.

---

### 27. AgentBoard: An Analytical Evaluation Board of Multi-turn LLM Agents

**Authors:** Chang Ma, Junlei Zhang, Zhihao Zhu, et al.  
**Year:** 2024 | **Citations:** 123 | **Venue:** Neural Information Processing Systems  
**Year Month:** [Jan 2024]  
**Fields:** Computer Science  
**URL:** [https://www.semanticscholar.org/paper/cf270bea2fba82bcff83f380c1f100d346b14ecf](https://www.semanticscholar.org/paper/cf270bea2fba82bcff83f380c1f100d346b14ecf)  
**arXiv:** [https://arxiv.org/abs/2401.13178](https://arxiv.org/abs/2401.13178)  

**Abstract:** Evaluating Large Language Models (LLMs) as general-purpose agents is essential for understanding their capabilities and facilitating their integration into practical applications. However, the evaluation process presents substantial challenges. A primary obstacle is the benchmarking of agent performance across diverse scenarios within a unified framework, especially in maintaining partially-observable environments and ensuring multi-round interactions. Moreover, current evaluation frameworks mostly focus on the final success rate, revealing few insights during the process and failing to provide a deep understanding of the model abilities. To address these challenges, we introduce AgentBoard, a pioneering comprehensive benchmark and accompanied open-source evaluation framework tailored to analytical evaluation of LLM agents. AgentBoard offers a fine-grained progress rate metric that captures incremental advancements as well as a comprehensive evaluation toolkit that features easy assessment of agents for multi-faceted analysis. This not only sheds light on the capabilities and limitations of LLM agents but also propels the interpretability of their performance to the forefront. Ultimately, AgentBoard serves as a step towards demystifying agent behaviors and accelerating the development of stronger LLM agents.

---

### 28. AgentHarm: A Benchmark for Measuring Harmfulness of LLM Agents

**Authors:** Maksym Andriushchenko, Alexandra Souly, Mateusz Dziemian, et al.  
**Year:** 2024 | **Citations:** 113 | **Venue:** International Conference on Learning Representations  
**Year Month:** [Oct 2024]  
**Fields:** Computer Science  
**URL:** [https://www.semanticscholar.org/paper/716c6f6a6e653bebfa676402b887fe2927e06c73](https://www.semanticscholar.org/paper/716c6f6a6e653bebfa676402b887fe2927e06c73)  
**arXiv:** [https://arxiv.org/abs/2410.09024](https://arxiv.org/abs/2410.09024)  

**Abstract:** The robustness of LLMs to jailbreak attacks, where users design prompts to circumvent safety measures and misuse model capabilities, has been studied primarily for LLMs acting as simple chatbots. Meanwhile, LLM agents -- which use external tools and can execute multi-stage tasks -- may pose a greater risk if misused, but their robustness remains underexplored. To facilitate research on LLM agent misuse, we propose a new benchmark called AgentHarm. The benchmark includes a diverse set of 110 explicitly malicious agent tasks (440 with augmentations), covering 11 harm categories including fraud, cybercrime, and harassment. In addition to measuring whether models refuse harmful agentic requests, scoring well on AgentHarm requires jailbroken agents to maintain their capabilities following an attack to complete a multi-step task. We evaluate a range of leading LLMs, and find (1) leading LLMs are surprisingly compliant with malicious agent requests without jailbreaking, (2) simple universal jailbreak templates can be adapted to effectively jailbreak agents, and (3) these jailbreaks enable coherent and malicious multi-step agent behavior and retain model capabilities. To enable simple and reliable evaluation of attacks and defenses for LLM-based agents, we publicly release AgentHarm at https://huggingface.co/datasets/ai-safety-institute/AgentHarm.

---

### 29. Tree Search for Language Model Agents

**Authors:** Jing Yu Koh, Stephen McAleer, Daniel Fried, et al.  
**Year:** 2024 | **Citations:** 110 | **Venue:** Trans. Mach. Learn. Res.  
**Year Month:** [Jul 2024]  
**Fields:** Computer Science  
**URL:** [https://www.semanticscholar.org/paper/9345e55a21959948499cee997522aa5eac7ed588](https://www.semanticscholar.org/paper/9345e55a21959948499cee997522aa5eac7ed588)  
**arXiv:** [https://arxiv.org/abs/2407.01476](https://arxiv.org/abs/2407.01476)  

**Abstract:** Autonomous agents powered by language models (LMs) have demonstrated promise in their ability to perform decision-making tasks such as web automation. However, a key limitation remains: LMs, primarily optimized for natural language understanding and generation, struggle with multi-step reasoning, planning, and using environmental feedback when attempting to solve realistic computer tasks. Towards addressing this, we propose an inference-time search algorithm for LM agents to explicitly perform exploration and multi-step planning in interactive web environments. Our approach is a form of best-first tree search that operates within the actual environment space, and is complementary with most existing state-of-the-art agents. It is the first tree search algorithm for LM agents that shows effectiveness on realistic web tasks. On the challenging VisualWebArena benchmark, applying our search algorithm on top of a GPT-4o agent yields a 39.7% relative increase in success rate compared to the same baseline without search, setting a state-of-the-art success rate of 26.4%. On WebArena, search also yields a 28.0% relative improvement over a baseline agent, setting a competitive success rate of 19.2%. Our experiments highlight the effectiveness of search for web agents, and we demonstrate that performance scales with increased test-time compute. We conduct a thorough analysis of our results to highlight improvements from search, limitations, and promising directions for future work. Our code and models are publicly released at https://jykoh.com/search-agents.

---

### 30. AI Agents vs. Agentic AI: A Conceptual Taxonomy, Applications and Challenges

**Authors:** Ranjan Sapkota, Konstantinos I. Roumeliotis, Manoj Karkee  
**Year:** 2025 | **Citations:** 109 | **Venue:** Information Fusion  
**Year Month:** [May 2025]  
**Fields:** Computer Science  
**URL:** [https://www.semanticscholar.org/paper/986e813f4c4f36786c3642cb9c8718586e47bdcf](https://www.semanticscholar.org/paper/986e813f4c4f36786c3642cb9c8718586e47bdcf)  
**arXiv:** [https://arxiv.org/abs/2505.10468](https://arxiv.org/abs/2505.10468)  
---

### 31. AI Agents Under Threat: A Survey of Key Security Challenges and Future Pathways

**Authors:** Zehang Deng, Yongjian Guo, Changzhou Han, et al.  
**Year:** 2024 | **Citations:** 107 | **Venue:** ACM Computing Surveys  
**Year Month:** [Jun 2024]  
**Fields:** Computer Science  
**URL:** [https://www.semanticscholar.org/paper/5aacf780ec16a29bdbe283a14f5a9e6b7e1f292d](https://www.semanticscholar.org/paper/5aacf780ec16a29bdbe283a14f5a9e6b7e1f292d)  
**arXiv:** [https://arxiv.org/abs/2406.02630](https://arxiv.org/abs/2406.02630)  

**Abstract:** An Artificial Intelligence (AI) agent is a software entity that autonomously performs tasks or makes decisions based on pre-defined objectives and data inputs. AI agents, capable of perceiving user inputs, reasoning and planning tasks, and executing actions, have seen remarkable advancements in algorithm development and task performance. However, the security challenges they pose remain under-explored and unresolved. This survey delves into the emerging security threats faced by AI agents, categorizing them into four critical knowledge gaps: unpredictability of multi-step user inputs, complexity in internal executions, variability of operational environments, and interactions with untrusted external entities. By systematically reviewing these threats, this article highlights both the progress made and the existing limitations in safeguarding AI agents. The insights provided aim to inspire further research into addressing the security threats associated with AI agents, thereby fostering the development of more robust and secure AI agent applications.

---

### 32. LLM Agents can Autonomously Exploit One-day Vulnerabilities

**Authors:** Richard Fang, R. Bindu, Akul Gupta, et al.  
**Year:** 2024 | **Citations:** 103 | **Venue:** arXiv.org  
**Year Month:** [Apr 2024]  
**Fields:** Computer Science  
**URL:** [https://www.semanticscholar.org/paper/04bcd16b564e47b019880dd7db65f7fffae7e2a5](https://www.semanticscholar.org/paper/04bcd16b564e47b019880dd7db65f7fffae7e2a5)  
**arXiv:** [https://arxiv.org/abs/2404.08144](https://arxiv.org/abs/2404.08144)  

**Abstract:** LLMs have becoming increasingly powerful, both in their benign and malicious uses. With the increase in capabilities, researchers have been increasingly interested in their ability to exploit cybersecurity vulnerabilities. In particular, recent work has conducted preliminary studies on the ability of LLM agents to autonomously hack websites. However, these studies are limited to simple vulnerabilities. In this work, we show that LLM agents can autonomously exploit one-day vulnerabilities in real-world systems. To show this, we collected a dataset of 15 one-day vulnerabilities that include ones categorized as critical severity in the CVE description. When given the CVE description, GPT-4 is capable of exploiting 87% of these vulnerabilities compared to 0% for every other model we test (GPT-3.5, open-source LLMs) and open-source vulnerability scanners (ZAP and Metasploit). Fortunately, our GPT-4 agent requires the CVE description for high performance: without the description, GPT-4 can exploit only 7% of the vulnerabilities. Our findings raise questions around the widespread deployment of highly capable LLM agents.

---

### 33. RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning

**Authors:** Zihan Wang, Kangrui Wang, Qineng Wang, et al.  
**Year:** 2025 | **Citations:** 101 | **Venue:** arXiv.org  
**Year Month:** [Apr 2025]  
**Fields:** Computer Science  
**URL:** [https://www.semanticscholar.org/paper/1d03586baa32b3d6ff657a180053821543e11abb](https://www.semanticscholar.org/paper/1d03586baa32b3d6ff657a180053821543e11abb)  
**arXiv:** [https://arxiv.org/abs/2504.20073](https://arxiv.org/abs/2504.20073)  

**Abstract:** Training large language models (LLMs) as interactive agents presents unique challenges including long-horizon decision making and interacting with stochastic environment feedback. While reinforcement learning (RL) has enabled progress in static tasks, multi-turn agent RL training remains underexplored. We propose StarPO (State-Thinking-Actions-Reward Policy Optimization), a general framework for trajectory-level agent RL, and introduce RAGEN, a modular system for training and evaluating LLM agents. Our study on four stylized environments reveals three core findings. First, our agent RL training shows a recurring mode of Echo Trap where reward variance cliffs and gradient spikes; we address this with StarPO-S, a stabilized variant with trajectory filtering, critic incorporation, and gradient stabilization. Second, we find the shaping of RL rollouts would benefit from diverse initial states, medium interaction granularity and more frequent sampling. Third, we show that without fine-grained, reasoning-aware reward signals, agent reasoning hardly emerge through multi-turn RL and they may show shallow strategies or hallucinated thoughts. Code and environments are available at https://github.com/RAGEN-AI/RAGEN.

---

