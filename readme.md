# Awesome Robot Memory：机器人长期记忆与 Memory VLA 文献综述

> 更新日期：2026-08-05
> 范围：Vision-Language-Action、长时程操作、部分可观测决策与机器人记忆

## 摘要

长时程机器人操作往往不满足马尔可假设。当前观测可能无法表示已离开视野的对象、早期交互结果、任务进度与隐含规则，因此策略必须建模历史。

本文将现有工作分为 episode 内策略记忆、跨 episode 与系统级记忆、训练与优化范式、数据集与评测四个层次。其中，episode 内记忆再按主要表示与更新机制细分。

## 1. 问题定义与研究边界

本综述关注记忆如何支持机器人的闭环感知与动作决策。核心问题是：在可控的计算、延迟与存储成本下，从历史中保留哪些信息，以何种形式保留，以及何时写入和读取。

本文主要覆盖单 episode 内的感知、动作与任务状态记忆，同时纳入跨示范经验检索、高层规划记忆和终身记忆。纯静态多帧输入只在其构成有意义的历史基线时纳入。

## 2. 分类框架与统一术语

本文以记忆的作用范围为一级分类，以核心表示与更新机制为二级分类。每篇论文按其主要贡献归入一类；梯度路径、记忆模态和读写策略作为横切属性讨论。

| 层次 | 分类 | 关键特征 |
|---|---|---|
| Episode 内 | 固定窗口与密集历史 | 直接编码近期或较长观测序列 |
| Episode 内 | 显式记忆库与压缩 | 维护可检索、可合并或固定容量的记忆 |
| Episode 内 | 事件与关键帧 | 仅保留预计对未来决策有用的稀疏历史 |
| Episode 内 | 递归状态与全历史模型 | 通过循环状态或线性时序模型传播历史 |
| Episode 内 | 结构化记忆 | 以对象、空间、场景、运动或概念为基本单元 |
| Episode 内 | 语言、动作与多模态记忆 | 使用视觉之外的可解释或控制相关信息 |
| 跨 Episode | 经验与系统级记忆 | 检索过去示范、技能、知识或长期环境状态 |

## 3. Episode 内策略记忆

### 3.1 固定窗口与密集历史上下文

这类方法保留规则采样的历史观测或压缩后的逐帧表征。优点是信息路径直接、易于复用预训练模型；限制是成本通常随上下文长度增长，且冗余帧可能稀释关键证据。

| 论文 | 核心方法 | 特点 |
|---|---|---|
| [HAMLET: Switch your VLA Model into a History-Aware Policy](https://arxiv.org/abs/2510.00695) | moment tokens、时间对比初始化与轻量历史聚合 | 固定短窗口 |
| [NativeMEM: Native Memory Compression for Long-Horizon Robotic Manipulation](https://arxiv.org/abs/2607.06678) | 使用 VLA 视觉编码器将每个历史帧—视角压缩为原生 token | 紧凑的逐帧历史 |
| [ContextVLA: Visual Context Compression for Efficient Multi-Frame VLA](https://arxiv.org/abs/2510.04246) | 将历史观测压缩为单个 context token | 极紧凑多帧表示 |
| [CronusVLA: A Vision-Language-Action Model with Multi-Frame Context](https://arxiv.org/abs/2506.19816) | 单帧预训练、多帧后训练与历史特征缓存 | 短期多帧上下文 |
| [Learning Long-Context Diffusion Policies via Past-Token Prediction](https://arxiv.org/abs/2505.09561) | 缓存长上下文特征，并以过去动作 token 预测作为辅助目标 | 长上下文 policy head |
| [Efficient Long-Horizon VLA Models via Static-Dynamic Disentanglement](https://arxiv.org/abs/2602.03983) | 分离静态与动态视觉 token，通过 recache gate 更新静态内容 | 降低多帧冗余 |
| [TempoFit: Plug-and-Play Layer-Wise Temporal KV Memory](https://arxiv.org/abs/2603.07647) | 缓存中间层 prefix K/V，相似度检索并施加时间衰减 | training-free 时间 KV 记忆 |
| [Think at 5 Hz, Act at 20 Hz: Asynchronous Fast-Slow Vision-Language-Action Inference for Closed-Loop Driving](https://arxiv.org/abs/2607.15621) | 慢速 VLM 将视觉历史编码为逐层 KV cache，快速 action expert 融合当前帧读取 | 固定历史长度成本的异步快慢闭环控制 |
| [Difference-Based Relational Learning for Zero-Shot Object-Goal Visual Navigation With Direct Sim-to-Real Transfer (T-DRN)](https://arxiv.org/abs/2607.15642) | 双帧 temporal buffer 与差分关系特征 | 窄视野下维持短期对象连续性并实现 sim-to-real 导航 |
| [TraceVLA: Visual Trace Prompting Enhances Spatial-Temporal Awareness](https://arxiv.org/abs/2412.10345) | 将状态—动作轨迹渲染为当前图像上的视觉 trace | 无需潜在记忆库 |
| [Instruction-driven History-aware Policies for Robotic Manipulations](https://arxiv.org/abs/2209.04899) | 联合编码语言、多视角观测与完整动作历史 | 早期全历史操作策略；CoRL 2022 |
| [FibVLA: An Efficient Temporal Vision-Language-Action Model with Fibonacci Sampling](https://arxiv.org/abs/2607.29596) | 对视觉帧与本体状态进行 logarithmic hindsight sampling | 以少量多尺度历史覆盖长时间范围，并通过 Fibonacci 递归推理结合实时反馈生成远期动作 |
| [How Should Vision-Language-Action Models Use Proprioceptive State?](https://arxiv.org/abs/2608.03052) | 受控比较 5 种状态注入接口与 1–96 帧本体状态历史 | 区分真实时间变化与单纯增加条件容量的收益，并总结状态历史长度和注入位置的 VLA 设计原则 |

### 3.2 显式记忆库、检索与压缩

显式记忆方法将历史作为可管理的外部或潜在容器。其研究重点是容量约束下的写入、合并、替换、检索与当前状态融合。

| 论文 | 核心方法 | 记忆组织 |
|---|---|---|
| [MemoryVLA: Perceptual-Cognitive Memory in VLA Models](https://arxiv.org/abs/2508.19236) | working memory、perceptual-cognitive bank、检索与冗余合并 | 感知—认知记忆库 |
| [MemoAct: Atkinson–Shiffrin-Inspired Memory-Augmented Visuomotor Policy](https://arxiv.org/abs/2603.18494) | 无损短期记忆、压缩长期记忆与相似度合并 | 工作/长期记忆 |
| [Dual Latent Memory in Vision-Language-Action Models for Robotic Manipulation (LaMem-VLA)](https://arxiv.org/abs/2607.07608) | 检索和压缩视觉、语义与动作潜在表征 | 短期视觉与长期语义/动作 vault |
| [VPWEM: Non-Markovian Visuomotor Policy with Working and Episodic Memory](https://arxiv.org/abs/2603.04910) | 近期 working window 与递归 episodic compressor | 工作/情景记忆 |
| [PAM: Resolving State Ambiguity via Adaptive Working Memory Recoding](https://arxiv.org/abs/2512.24638) | 多时间范围 query、working-memory recoding 与历史重建 | 尺度自适应记忆 |
| [MemoryVAM: Integrating Memory into Video Action Model for Robot Manipulation](https://arxiv.org/abs/2606.20679) | Perceiver Recap Compressor 与任务进度 Cue Gate | 视频模型与动作解码器共享记忆 |
| [ELMUR: External Layer Memory with Update/Rewrite for Long-Horizon RL](https://arxiv.org/abs/2510.07151) | 每层外部记忆、双向交叉注意力与替换/混合规则 | 分层外部记忆 |
| [Gated Memory Policy](https://arxiv.org/abs/2604.18933) | 学习何时读取历史以及读取哪些内容 | 门控交叉注意力记忆 |
| [AURA-Mem: Action-Gated Memory for Robot Policies at Constant VRAM](https://arxiv.org/abs/2606.02775) | 在 action-relevant surprise 时写入固定大小 fast-weight state | O(1) 状态显存 |
| [TFP: Temporally Conditioned Memory-Fusion Policies for Visuomotor Learning](https://arxiv.org/abs/2607.08283) | 以显式时间条件融合 latent belief 与记忆 | elapsed-time-conditioned dynamics |
| [HALO: Memory Retrieval in Visuomotor Policies for Long-Horizon Robot Control](https://arxiv.org/abs/2606.25136) | 从图像、本体状态与动作历史中学习稀疏注意力检索 | 分钟级长时程历史读取 |
| [MemoryWAM: Memory-Augmented World Action Model for Long-Horizon Robot Manipulation](https://arxiv.org/abs/2606.20562) | 保留短期细节并递归压缩长期上下文 | 双时间尺度持久记忆 |
| [HELM: Harness-Enhanced Long-horizon Memory for VLA Manipulation](https://arxiv.org/abs/2604.18791) | CLIP 索引关键帧、状态验证与失败回滚 | 策略外执行 harness 记忆 |
| [PrediMem: Predictive Memory for Robotic Manipulation](https://arxiv.org/abs/2605.10921) | 近期/关键帧双缓冲 memory bank 与 predictive coding | 与 RoboMemArena 一同提出 |
| [MemoGuard: An Adaptive Runtime for Guarding Against Memory Traps in Communication-Limited Robot Navigation](https://arxiv.org/abs/2607.15589) | 在复用情景记忆前验证拓扑、资源与历史结果约束 | 拒绝高相似但执行无效的 memory trap，仅在验证失败时调用本地推理 |
| [WorldScape Policy 2.0: Empowering Steerable World Action Modeling with Reasoning-Augmented Memory](https://arxiv.org/abs/2607.18840) | 近期视觉 DiT prefill 与分层事件记忆 | 将历史 VLM 输出组织为 global-history、local-active 和 event-boundary 表示，并按任务进度检索以形成隐式子目标 |
| [DART-VLN: Test-Time Memory Decay and Anti-Loop Regularization for Discrete Vision-Language Navigation](https://arxiv.org/abs/2607.01043) | 读取侧记忆衰减与局部回环惩罚 | training-free 抑制陈旧、冗余历史证据并减少立即折返，不改写记忆或冻结的导航 backbone |
| [E-TTS: A New Embodied Test-Time Scaling Framework for Robotic Manipulation](https://arxiv.org/abs/2606.27268) | history buffer 为 reasoning/action verifiers 提供历史上下文 | training-free 推理—动作联合采样、评分与闭环迭代 refinement |
| [Scene Memory Transformer for Embodied Agents in Long-Horizon Tasks](https://arxiv.org/abs/1903.03878) | 将逐步观测嵌入场景记忆并以注意力读取时空依赖 | 面向长 episode 部分可观测导航的早期全历史记忆策略 |
| [ST-WAM: Semantic-Temporal World Action Model for Robust Manipulation under Visual Distribution Shifts](https://arxiv.org/abs/2607.28993) | Current-Anchored Intent Retrieval 从近期 DINO 历史检索任务相关证据 | 联合语义与像素动态的未来预测，在视觉分布变化下抑制训练域幻觉且部署时无需显式生成未来图像 |
| [Auto-JEPA: A Latent World Model of Continuous Intent for End-to-End Autonomous Driving](https://arxiv.org/abs/2607.29031) | 固定 trajectory memory 与未来驾驶意图检索 | 根据视觉、egomotion 历史和导航指令预测未来轨迹 latent，再检索并排序可执行轨迹；机制与机器人记忆相关，但应用边界为自动驾驶 |

### 3.3 事件驱动与关键帧记忆

这类方法不均匀保留历史，而是根据语义、运动学、视觉差异或预计未来效用选择事件。稀疏记忆可降低计算量，但其性能受事件判定的召回率与时序信用分配影响。

| 论文 | 事件选择方式 | 读写特点 |
|---|---|---|
| [EventVLA: Event-Driven Visual Evidence Memory for Long-Horizon VLA Policies](https://arxiv.org/abs/2606.20092) | 预测历史帧的未来效用 | 结合初始、近期与动态事件关键帧 |
| [KEMO: Event-Driven Keyframe Memory for Long-Horizon Robot Manipulation with VLA Policies](https://arxiv.org/abs/2606.23589) | 运动学事件检测与视觉差异过滤 | 关键帧交叉注意力读取 |
| [Non-Markovian Long-Horizon Robot Manipulation via Keyframe Chaining](https://arxiv.org/abs/2603.01465) | 从全历史选择关键帧 | 历史与当前视觉 token 交错输入 |
| [BPP: Long-Context Robot Imitation Learning by Focusing on Key History Frames](https://arxiv.org/abs/2602.15010) | 利用现成 VLM 问答选择语义关键帧 | 外部模型驱动的稀疏记忆 |
| [WeaveLA: Event Driven Cross-Subtask Latent Memory Weaving](https://arxiv.org/abs/2606.17463) | 在子任务完成时写入 | 将压缩记忆传递给下一子任务 action expert |
| [DiM-WAM: World-Action Modeling with Diverse Historical Event Memory](https://arxiv.org/abs/2606.27677) | 不同历史事件分库存储 | 库内相似度合并与任务进度监督 |
| [Chameleon: Control-Indexed Prospective Memory for Visuomotor Manipulation](https://arxiv.org/abs/2603.24576) | 写入 geometry-grounded embodied event | 可微记忆栈形成面向控制的 prospective state |
| [CycleManip: Enabling Cyclic Task Manipulation via Effective Historical Perception](https://arxiv.org/abs/2512.01022) | cost-aware 历史采样 | 面向重复动作和正确终止时间 |
| [Bi-HIL: Bilateral Control-Based Multimodal Hierarchical Imitation Learning](https://arxiv.org/abs/2603.13315) | 以子任务进度率识别并写入关键帧 | 面向接触丰富操作的层次化记忆 |

### 3.4 递归隐状态、状态空间模型与完整历史

递归方法将历史浓缩为跨控制时刻传播的状态，其推理成本可与 episode 长度解耦。全历史模型则通过 SSM 或 Mamba 顺序扫描轨迹，在保留长范围信息的同时控制复杂度。

| 论文 | 状态表示 | 时序传播 |
|---|---|---|
| [μVLA: On Recurrent Memory for Partially Observable Manipulation](https://arxiv.org/abs/2606.12497) | 少量 recurrent memory tokens | 跨真实控制时刻传递 |
| [ReMem-VLA: Empowering VLA with Memory via Dual-Level Recurrent Queries](https://arxiv.org/abs/2603.12942) | frame-level 与 chunk-level recurrent queries | 固定 EMA 传播 |
| [Chronos: A Physics-Informed Full-History Framework for Non-Markovian Long-Horizon Manipulation](https://arxiv.org/abs/2606.30318) | 每个物理时刻一个 state token | 选择性 SSM 编码完整轨迹 |
| [MTIL: Encoding Full History with Mamba for Temporal Imitation Learning](https://arxiv.org/abs/2505.12410) | Mamba-2 temporal state | 轨迹级顺序更新 |
| [DSSP: Diffusion State Space Policy with Full-History Encoding](https://arxiv.org/abs/2605.14598) | Mamba history encoder | 完整 demonstration 因果扫描 |
| [Recursive Belief Vision-Language-Action Models](https://arxiv.org/abs/2602.20659) | 潜在 belief state | 以自监督 world-model 目标训练递归状态 |
| [Recurrent Action Transformer with Memory](https://arxiv.org/abs/2306.09459) | 专用 memory embeddings | 在 context segment 之间递归传递 |
| [AnoleVLA: Lightweight VLA with Deep State Space Models for Mobile Manipulation](https://arxiv.org/abs/2603.15046) | deep SSM hidden state | 线性复杂度的长上下文建模 |
| [SeedPolicy: Horizon Scaling via Self-Evolving Diffusion Policy](https://arxiv.org/abs/2603.05117) | 固定大小 self-evolving latent state | gated attention 持续更新 |
| [MEMBOT: Memory-Based Robot in Intermittent POMDP](https://arxiv.org/abs/2509.11225) | SSM/LSTM belief encoder 聚合观测与动作序列 | 面向间歇观测缺失的重建预训练与 BC |
| [RoboTTT: Context Scaling for Robot Policies](https://arxiv.org/abs/2607.15275) | 推理时梯度更新形成 fast-weight recurrent state | 将视觉—动作上下文扩展至 8K timesteps，并以 sequence action forcing 与 TBPTT 训练 |
| [Beyond Transformers: Linear Attention Policy for Open-Vocabulary Object Goal Navigation](https://arxiv.org/abs/2607.18794) | Weighted State-Expansion Linear Attention 子状态 | 以线性注意力执行结构化内部状态更新，避免 Transformer 固定窗口在长距离部分可观测导航中的扩展瓶颈 |
| [Beyond Fixed Goal Delivery: Online POMDP Planning for Target Interception in Crowds](https://arxiv.org/abs/2607.18517) | 人类导航意图的在线 belief 与搜索树 | 在固定计算预算下维护多个未来交互假设，并联合搜索转向和速度以完成拥挤环境中的安全拦截 |
| [One Future, Every Robot: Label-Efficient Collective-State Prediction with Decentralized JEPA](https://arxiv.org/abs/2607.28443) | 16 帧局部历史与邻居间 64-float 递归消息 | 各机器人仅凭局部观测和带宽受限通信预测一致的未来群体 token field，无需全局池化、episode 时钟或未来动作 |
| [From Routes to Steps: Separating Semantic Progress from Local Execution in Vision-and-Language Navigation](https://arxiv.org/abs/2608.03143) | 由全局指令与累积视觉历史估计的 step-level progress state | 将语义进度追踪与近期观测驱动的局部动作生成解耦，分别诊断进度记忆错误和执行错误 |

### 3.5 对象中心、空间与场景记忆

结构化方法不将历史视为无差别 token 序列，而是显式维护对象、位姿、轨迹、关系、概念或场景 belief。这类表示更适合处理视野外对象、空间一致性与可解释推理。

| 论文 | 结构化单元 | 主要能力 |
|---|---|---|
| [History-Aware Visuomotor Policy Learning via Point Tracking](https://arxiv.org/abs/2509.17141) | 对象点轨迹 | 全历史运动与任务阶段建模 |
| [Rethinking Progression of Memory State in Robotic Manipulation: An Object-Centric Perspective](https://arxiv.org/abs/2511.11478) | 持久对象槽位 | slot-state-space dynamics 与关系编码 |
| [SOMA: Spatial Memory for Out-of-Vision Manipulation in VLA](https://arxiv.org/abs/2605.22283) | 持久空间语义记忆 | 多视角构建、动态修正与指令检索 |
| [SAM2Act: Integrating Visual Foundation Model with a Memory Architecture for Robotic Manipulation](https://arxiv.org/abs/2501.18564) | 视觉 memory bank | 结合 SAM2 编码器与 attention 的空间记忆 |
| [EvoScene-VLA: Evolving Scene Beliefs Inside the Action Model](https://arxiv.org/abs/2605.21862) | scene belief | 在动作生成路径内持续演化场景状态 |
| [Analytic Concept-Centric Memory for Agentic Embodied Manipulation](https://arxiv.org/abs/2606.29774) | 对象部件、位姿、affordance、状态转移与技能 | 结构化、可解释的概念记忆 |
| [EchoVLA: Synergistic Declarative Memory for VLA-Driven Mobile Manipulation](https://arxiv.org/abs/2511.18112) | scene memory 与 episodic memory | 移动操作中的粗到细检索融合 |
| [RoboStream: Weaving Spatio-Temporal Reasoning with Memory in Vision-Language Models for Robotics](https://arxiv.org/abs/2603.12939) | 绑定视觉证据与 3D 几何的 STF-Tokens、因果时空图 | training-free 持久对象 grounding 与动作触发状态转移追踪，支持遮挡下对象恒常性和长程 VLM 规划 |
| [OASIS-Map: Object-Level Change Detection in Multi-Session Mapping using Semantic Correspondence Matching](https://arxiv.org/abs/2607.14899) | 跨 session 的时空一致对象地图 | 通过稠密 patch 级语义对应检测场景变化，并在遮挡、局部视图和分割误差下增量关联重访对象 |
| [SkillNav: Score-Level Skill Intervention for Zero-Shot Object Goal Navigation](https://arxiv.org/abs/2607.15758) | curiosity value map 上的可写空间行为记忆与短提示语义记忆 | 以分级 score intervention 组合导航技能，无需训练且不增加历史 token 成本 |
| [ChronoFlow-Policy: Unifying Past-Current-Future Interaction Flow](https://arxiv.org/abs/2606.31493) | 对象与夹爪的稀疏 3D keypoints | 统一表示过去、当前与未来交互动态 |
| [VQ-Memory for Robust Long-Horizon Manipulation](https://arxiv.org/abs/2603.09513) | 离散化关节状态 token | 以 VQ-VAE 表示任务阶段与本体感觉历史 |
| [mindmap: Spatial Memory in Deep Feature Maps for 3D Action Policies](https://arxiv.org/abs/2509.20297) | 语义 3D 重建与深层特征图 | 记忆视野外对象并生成 3D 动作轨迹 |
| [Mem-World: Memory-Augmented Action-Conditioned World Models](https://arxiv.org/abs/2606.18960) | 4D wrist-view surfel memory | 持久世界 rollout 与策略评估/改进 |
| [Stigmergic Graph Memory: An Environment-Aware Approach for Many-to-Many Multi-Agent Pickup and Delivery](https://arxiv.org/abs/2607.15182) | 仓库节点与有向边上的执行信号 | 以有界衰减图记忆联合引导多机器人端点选择与路径偏好 |
| [VLN-AVP: Zero-Shot Vision-Language Navigation with Hybrid Long-Short-Term Memory for Autonomous Valet Parking](https://arxiv.org/abs/2607.17767) | 短期语义感知记忆与长期拓扑记忆 | 在无预建地图的开放词汇自主泊车中结合近期视觉线索与历史空间结构 |
| [Closing the Loop in Humanoid VLA: Persistent 3D Object Tokens for Verifiable Loco-Manipulation](https://arxiv.org/abs/2607.18016) | 角色索引的持久 3D 对象记录 | 将 RGB-D 对象状态转换为动作与几何验证共享的 token，跨运动、接触、遮挡和恢复维持对象一致性 |
| [Spatiotemporal Knowledge Graphs as Persistent Scene Memory for Embodied Question Answering](https://arxiv.org/abs/2510.01483) | 跨视频片段的时空知识图谱 | 维持对象身份并以子图检索结合视觉 grounding，使查询延迟不再随观测历史增长 |
| [ReferTrack: Referring Then Tracking for Embodied Visual Tracking](https://arxiv.org/abs/2607.20061) | 历史目标框滑动窗口与 TVBI tokens | 将目标框的时间、视点和几何信息注入视觉历史，持续保留单目具身跟踪中的目标运动线索 |
| [GLAM-SLAM: Efficient Dense Gaussian SLAM for Long-Horizon Large-Scale Scenes](https://arxiv.org/abs/2607.21416) | 稀疏锚点网格、光流稠密化与分区 Gaussian 地图 | 在长距离、大尺度序列中限制地图增长并保持实时稠密重建，为持久空间记忆提供可扩展底座 |
| [Learning Spatiotemporal Decision Priors for Efficient Path Planning under Partial Observability (ImiPath)](https://arxiv.org/abs/2607.22166) | 局部时空观测与历史轨迹决策先验 | 从示范轨迹蒸馏可复用的方向偏好，以 STAPNet 引导异构规划器减少部分可观测长程导航中的冗余搜索 |
| [Memory for Attention: Language-Conditioned Re-Perception with a Vision–Language–Motion Map](https://arxiv.org/abs/2607.23797) | 含对象变化历史、最近观测时间和运动通道的持久 VLMM | 将记忆用于有限感知预算下的重观测调度；其路径规划收益有限，但可优先刷新语言相关且高变化的对象 |
| [Shared Voxel-Map-Based Cooperative Indoor UAV Guidance with a Multi-Agent Soft Actor-Critic Controller](https://arxiv.org/abs/2607.25728) | 多无人机共享的世界坐标 voxel map 与 ego-aligned BEV 切片 | 融合各机 360° LiDAR 历史形成一致空间 belief，在集中训练、分散执行下支持协同室内导航 |
| [Explicit Kinematic Guidance from Analytic Concepts for Vision-Language-Action Models](https://arxiv.org/abs/2607.26513) | 从 3D 视觉基础模型构建的可执行 Analytic Concepts | 动态追踪对象与机构的运动学参数，以稠密程序化奖励和显式空间约束指导 VLA 完成精细操作 |
| [EgoGenesis: Egocentric World-Action Modeling with Online Anchored Projective Memory and Action-3D RoPE](https://arxiv.org/abs/2607.28243) | 首帧 3D 场景锚点与周期更新的近期状态 | Online Anchored Projective Memory 在自回归视频生成中兼顾长期几何一致性与动态刷新，供可控具身数据生成使用 |
| [Write-Safe Flow Field Mapping under Ambiguous Onboard Sensing and Localization Drift](https://arxiv.org/abs/2607.27713) | 带 write-safety score 的持久全局流场地图 | 根据观测歧义、定位漂移和地图参考连续衰减不可靠写入，抑制错位更新累积成长期 ghost structures |
| [HAM-VLN: Harnessing Hierarchical Agentic Memory for Zero-Shot Vision-and-Language Navigation](https://arxiv.org/abs/2607.29600) | 持久 depth-grounded world graph、近期 waypoint 与检索式旧历史 | 在决策调用中同步记录房间、对象、导航进度和失败反思，并按相关性、时效性、显著性及拓扑邻域召回，减少超过 65% 的上下文长度 |
| [Neurosymbolic Reasoning with Incremental Knowledge for Sample-Efficient Hierarchical Reinforcement Learning](https://arxiv.org/abs/2608.02993) | 探索过程中持续更新的符号环境知识 | 高层组件在当前知识表示上执行 D* 规划，低层模块从经验学习运动 primitive，并以 Belief World Tree Search 处理先验不确定性 |
| [SLAMFormer-$\infty$: Infinite SLAM Transformer for Unbounded Frontend and Backend Processing](https://arxiv.org/abs/2608.03429) | 定义局部坐标系与尺度的 memory conditions | 以局部有界计算和全局轨迹—几何优化支持无显式距离上限的空间记忆，在超过 17 km 的序列上保持一致重建 |

### 3.6 语言、动作与多模态记忆

视觉历史并非唯一记忆载体。动作历史可表示已执行的行为与任务进度，语言可将对象位置、子目标与规则外显化，运动表征则强调交互变化而非静态外观。

| 论文 | 主要模态 | 核心方法 |
|---|---|---|
| [Remember What You Did? Learning Behavioral Memories for Partially Observable Object Manipulation](https://arxiv.org/abs/2606.21188) | 动作历史 | 以过去动作的 DCT 重建训练递归行为记忆 |
| [Notes-to-Self: Scratchpad Augmented VLAs for Memory Dependent Manipulation Tasks](https://arxiv.org/abs/2602.21013) | 自然语言 | scratchpad 保存对象位置、计划与子目标进度 |
| [MEM: Multi-Scale Embodied Memory for Vision Language Action Models](https://arxiv.org/abs/2603.03596) | 视频与语言 | 短期视频记忆与长期语言事件记忆 |
| [HiF-VLA: Hindsight, Insight and Foresight through Motion Representation](https://arxiv.org/abs/2512.09928) | 运动 | 用 motion 压缩历史变化，联合 hindsight 与 future-motion reasoning |
| [CDP: Robust Autoregressive Visuomotor Policy Learning via Causal Diffusion](https://arxiv.org/abs/2506.14769) | 动作 | 以历史动作序列条件化 diffusion policy，并缓存跨时刻 attention K/V |
| [LIFT: Never Too Late for Force — Accelerating VLA Post-Training with Reactive Force Injection](https://arxiv.org/abs/2607.14236) | 力觉 | 以近期 6D 末端力构成 causal force memory，通过交叉注意力驱动接触阶段的反应式动作更新 |
| [T-Rex: Tactile-Reactive Dexterous Manipulation](https://arxiv.org/abs/2606.17055) | 高频触觉 | temporal tactile VQ-VAE 与变速率 Mixture-of-Transformers 建模触觉历史并驱动反应式灵巧操作 |
| [FM-VLA: Force-based Memory for Vision-Language-Action Models in Contact-Rich Manipulation](https://arxiv.org/abs/2607.18231) | 力觉与短期状态历史 | 以 VAE 将力时序压缩为 force memory tokens，辨别视觉近似但接触次数或隐状态不同的非马尔可任务 |
| [Temporal Policy: History-Initialized Action Generation for Robotic Learning from Demonstration](https://arxiv.org/abs/2607.29482) | 近期机器人状态历史 | 用历史而非独立高斯噪声初始化 stochastic-interpolant action flow，显式耦合过去状态与未来动作并降低生成路径和推理成本 |

## 4. 跨 Episode 经验与系统级记忆

跨 episode 记忆将可用历史从当前轨迹扩展到过往示范、经验库、任务先验和持续更新的环境知识。这些方法通常在高层规划、技能调度或长期知识管理中使用记忆。

| 论文 | 记忆范围 | 核心方法 |
|---|---|---|
| [MemER: Scaling Up Memory for Robot Control via Experience Retrieval](https://arxiv.org/abs/2510.20328) | 跨轨迹经验 | 关键帧化经验库与检索 |
| [MAP-VLA: Memory-Augmented Prompting for VLA in Robotic Manipulation](https://arxiv.org/abs/2511.09516) | 跨示范任务阶段 | 以可学习 soft prompts 表示并按轨迹相似度检索 |
| [OptimusVLA: Global Prior Meets Local Consistency](https://arxiv.org/abs/2602.20200) | 跨示范先验与 episode 内动作 | Global Prior Memory 与 Local Consistency Memory |
| [RoboMemory: A Brain-inspired Multi-memory Agentic Framework for Lifelong Learning](https://arxiv.org/abs/2508.01415) | 跨任务终身记忆 | spatial、temporal、episodic、semantic memory 与动态知识图谱 |
| [LifelongVLA: Towards Human-like Physical Intelligence via Lifelong Vision-Language-Action Learning](https://arxiv.org/abs/2607.14852) | 跨任务技能保持与持续扩展 | 以短期/长期双时间尺度 LoRA 门控平衡可塑性与稳定性，并通过 cache-efficient stochastic replay 巩固旧技能 |
| [ABot-AgentOS: A General Robotic Agent OS with Lifelong Multi-modal Memory](https://arxiv.org/abs/2607.10350) | 跨场景、跨模态的持久图记忆 | 将对话、视觉、空间、时间与任务轨迹组织为类型化图，并以失败驱动的 gated evo-assets 持续演化；同时提出 EmbodiedWorldBench |
| [HiMe: Hierarchical Embodied Memory for Long-Horizon VLA Control](https://arxiv.org/abs/2607.03449) | 系统级快慢记忆 | Executor、Sentry、Planner 分层与 Add/Update/Delete 知识管理 |
| [LoHo-Manip: Long-Horizon Manipulation via Trace-Conditioned VLA Planning](https://arxiv.org/abs/2604.21924) | 高层任务进度 | VLM 输出 done/remaining 语言记忆与 2D visual trace |
| [Vesta: A Generalist Embodied Reasoning Model](https://arxiv.org/abs/2606.20905) | 高层规划与执行状态 | 历史帧与运行中的文本子任务 cache |
| [AdaManip: Adaptive Articulated Object Manipulation Environments and Policy Learning](https://arxiv.org/abs/2502.11124) | 试探性交互历史 | 根据过往试探结果适应不可观测内部机构 |
| [Retrieval-Augmented Embodied Agents](https://arxiv.org/abs/2404.11699) | 跨示范策略经验 | 外部 policy memory bank、多模态检索与策略生成 |
| [Long-Term Memory for VLA-based Agents in Open-World Task Execution](https://arxiv.org/abs/2604.15671) | 跨执行成功轨迹 | 将成功经验固化为可检索资产并用于后续规划 |
| [Harness VLA: Steering Frozen VLAs via Memory-Guided Agents](https://arxiv.org/abs/2607.08448) | 跨执行规则与失败经验 | 以 execution traces、成功规则和 failure models 约束冻结 VLA |
| [PhysClaw-0: A Symbiotic Agentic System for Robot Autonomy via Language Corrections](https://arxiv.org/abs/2607.14047) | 跨轮次纠错经验 | 将自然语言纠错解析为结构化调整并写入 Corrective Memory，在相同条件下复用以减少重复人工干预 |
| [Experience Memory Graph: One-Shot Error Correction for Agents](https://arxiv.org/abs/2607.13884) | 跨任务成功与失败轨迹 | 将轨迹构造成动作决策图，以图匹配提取成功工作流和纠错 edit path，并在 ALFWorld 等具身任务中单次检索执行 |
| [MEMORA: Embodied Action Memory from Egocentric Videos for Reasoning and Planning](https://arxiv.org/abs/2607.14252) | 第一视角跨任务经验 | 以形成—巩固—检索生命周期维护环境、实体、活动和推断知识四类记忆，用于记忆驱动的机器人规划 |
| [HyperDCM: Dynamic Cluster Memory Replay in Hyperbolic Space for Continual Robotic Navigation Across Scenes](https://arxiv.org/abs/2607.16267) | 跨场景持续导航经验 | 将场景图嵌入双曲空间，以动态聚类和结构敏感更新选择代表性回放样本，缓解灾难性遗忘 |
| [PhyAgentOS: A Self-Evolving Operating System for Embodied Agents with Decoupled Cognitive Planning and Physical Execution](https://arxiv.org/abs/2607.16636) | 跨 embodiment 的验证执行经验 | 以 State-as-a-File 共享状态，并将验证后的结果沉淀为 epistemic memory、可复用知识与纠错经验 |
| [Retriever: Composing Closed-Loop Asynchronous Robot Programs](https://arxiv.org/abs/2607.17213) | 有限记忆的异步闭环程序状态 | 以有状态因果流函数组合感知、belief、规划与控制，并支持日志驱动的确定性重放 |
| [RoboHarness: Memory-Driven Orchestration of Heterogeneous Robot Policies for Long-Horizon Planning](https://arxiv.org/abs/2607.18060) | 跨策略多模态执行轨迹 | 由执行记忆刻画策略能力边界，Memory Bridge 检索下一策略的轨迹并引导机器人进入其分布内状态区域 |
| [Robots Acquire Manipulation Skills in Seconds from a Single Human Video (HOST)](https://arxiv.org/abs/2607.20033) | 单视频即时技能获取与已有技能保持 | 依次估计任务进度、预测机器人未来观测并反推动作，在推理时快速学习新操作程序 |
| [Extreme-RGMT: Continual Learning of Highly Dynamic Skills for Robust Generalist Humanoid Control](https://arxiv.org/abs/2607.20110) | 人形机器人技能获取与能力巩固 | 约束已掌握动作上的策略漂移，并以难度感知采样和 advantage-prioritized trajectory replay 强化高动态片段 |
| [EgoRecovery: Acquiring Failure Recovery Ability Through Human Recovery Demonstration](https://arxiv.org/abs/2607.19745) | 人类与机器人共享的纠错意图经验 | 对齐人类恢复片段和少量机器人示范，由 recovery gate 在检测到失败状态时激活可执行纠错行为 |
| [Courteous Anticipation: Improving Long-Lived Task Planning in Persistent Shared Environments](https://arxiv.org/abs/2607.20289) | 持久多机器人环境中的未来任务影响 | 估计当前计划终态对各机器人后续任务的期望成本，避免长期任务序列中的副作用累积 |
| [Closing the Lab-to-Store Gap: A Data-Efficient Post-Training and Experience-Driven Learning VLA Framework for Retail Humanoids](https://arxiv.org/abs/2607.20345) | 零售真机执行经验 | DEED 结合文本 advantage prefix、视觉语言价值函数和针对性后训练，从真实经验改进 GR00T VLA |
| [Addressing the Orchestration Gap in Generalist Robots via Physical Agency (Pigey)](https://arxiv.org/abs/2607.21725) | 当前任务的规划、执行、验证与恢复状态 | 在冻结 VLA 或参数化技能外构建闭环 orchestrator，持续跟踪子目标结果并在失败后重规划，无需额外收集数据或后训练 |
| [StARS: Socially Appropriate Robot Actions via a Recommender System-Driven Approach](https://arxiv.org/abs/2607.21802) | 跨用户的稀疏社会偏好 | 将用户、场景和候选动作适宜度建模为协同过滤问题，结合场景表征生成个性化机器人动作评分 |
| [WCM: World-Cognition Model for Generalizable Human-Robot Interaction](https://arxiv.org/abs/2607.22999) | 教学 episode、自主 rollout 与系统级知识 | SLAK 显式分离感知、逻辑、动作和记忆，并将人机教学与执行轨迹提炼为持续改进长程任务的监督数据 |
| [Try Once, Then Optimal: De-Redundified Procedure Memory for Cross-Episode Exploration Amortization](https://arxiv.org/abs/2607.23702) | 绑定可识别物体实例的短程序记忆 | 首次交互后记录揭示隐藏状态的操作过程，重访时作为反馈策略的软偏置召回，减少 16%–30% 的重复探索且可从错误记忆中恢复 |
| [A Few Words Go a Long Way: Language Guided Robot Policy Synthesis (ARCHITECT)](https://arxiv.org/abs/2607.23784) | 程序执行轨迹、语言纠错与持久技能库 | 用编码代理合成模块化机器人程序，将人工纠错 grounding 到执行轨迹并沉淀为可复用、可解释技能，逐步降低新任务所需干预 |
| [Not Forgotten: Implementation and Evaluation of a Personalized Episodic Memory for the Humanoid Robot Head Kim](https://arxiv.org/abs/2607.24190) | 跨会话个人交互记忆 | 以向量语义相似度和记忆强度混合评分检索往期交互并注入 LLM，上线研究显示其提升社交性、可信度和温暖感 |
| [Belief-Aware Influence and Trust: Shaping Human Belief During Repeated Human-Robot Interaction (BAIT)](https://arxiv.org/abs/2607.25327) | 重复交互中的快速人类策略与缓慢感知信念 | 用分层粒子滤波持续估计双时间尺度 belief，并以 MPPI 在即时性能约束下平衡长期影响和用户信任 |
| [Decompose and Reorganize: Planning with Primitives and Visuomotor Policies Learned from Demonstrations (DR-LfD)](https://arxiv.org/abs/2607.25397) | 跨示范的原子技能与接触关系 | 按接触关系分解视觉运动策略和对象中心 primitive，再通过 TAMP 门控重组为未见配置下的多阶段长程任务 |
| [Practice Makes Policies: Bootstrapping Closed-Loop Visuomotor Skills from Open-World Robot Experience (HERO)](https://arxiv.org/abs/2607.26809) | 自主练习产生的跨回合交互经验 | 以启发式推理、示例复用和反思式执行启动无人工示范探索，并将反复出现的交互巩固为闭环视觉运动策略 |
| [LabEvolver: Training-Free Experience Evolution for Safe and Grounded Wet-Lab Agents](https://arxiv.org/abs/2607.27690) | 跨任务的技能、策略与安全 episodic memory | 以内层状态 grounding 执行循环完成任务，再由外层演化循环将轨迹提炼为可复用经验，在连续实验中减少时间和安全拦截 |
| [RoboBRIDGE: A Modular Framework for Bridging Policies to Robust Real-World Robotic Agents](https://arxiv.org/abs/2607.27881) | 长时程执行状态、失败和恢复信息 | 在预训练 VLA 外构建模块化 orchestration layer，协调状态追踪、执行验证、故障恢复与跨任务/场景/embodiment 适配 |
| [ETA: A New Agentic Paradigm for Embodied Tasks](https://arxiv.org/abs/2608.03924) | 成功与失败交互、可审计记忆和可重放轨迹 | 以 Planner–Interface–World 闭环逐步执行和核验，并将交互沉淀为可复用经验；OpenETA 提供可替换规划器、组合式工具与仿真/真机统一接口 |
| [Episodic Memory Model for Learning Robotic Manipulation Tasks](https://arxiv.org/abs/2104.10218) | 单次示范经验 | 形成可分解、可重放的状态转移与动作序列 |
| [Deep Episodic Memory: Encoding, Recalling, and Predicting Episodic Experiences](https://arxiv.org/abs/1801.04134) | 视觉—动作 episode | 无监督编码、相似经验检索、重建与未来预测 |

## 5. 训练与优化范式

### 5.1 辅助目标与表征学习

长期记忆的监督不必仅来自当前动作损失。部分方法通过重建过去、预测交互效果、估计任务进度或统一历史与未来运动来约束记忆表征。

| 论文 | 训练信号 | 目的 |
|---|---|---|
| [Action-Effect Memory Pretraining for Robot Manipulation](https://arxiv.org/abs/2606.12499) | 动作及其环境效果 | 学习 interaction consequence 而非仅压缩观测 |
| [Recurrent-Depth VLA](https://arxiv.org/abs/2602.07845) | 动作头迭代 refinement | 深度方向的递归推理，不等同于 episode 时间记忆 |
| [AVA-VLA](https://arxiv.org/abs/2511.18960) | 连续观测短窗口展开 | 学习跨真实控制时刻的递归状态 |
| [Generalizable Task Planning through Representation Pretraining](https://arxiv.org/abs/2205.07993) | 合成场景理解数据中的对象级表示预训练 | 以参数化语义与几何先验提升未见对象上的多步操作规划泛化 |
| [CaP-X: A Framework for Benchmarking and Improving Coding Agents for Robot Manipulation](https://arxiv.org/abs/2603.22435) | 多轮交互、结构化执行反馈、视觉差分与技能合成 | 扩展 agentic test-time computation，并以可验证奖励强化 Code-as-Policy 控制器 |
| [Foresight Residual RL for Long-Horizon Robot Manipulation with Vision-Language-Action Models](https://arxiv.org/abs/2607.16506) | 当前子任务终态对应的下游 rollout 成功率 | 学习视觉 foresight predictor，并以未来子任务成功概率塑造残差策略的交接状态质量 |
| [RoboInter1.5: A Holistic Intermediate Representation Suite for Embodied World Modeling and Robotic Manipulation](https://arxiv.org/abs/2607.18709) | 子任务、技能、对象、接触点与运动轨迹等稠密中间表示 | 以结构化表示同时约束 VLA 动作执行与未来世界状态 rollout，并提供 23 万余条操作 episode 和时空 VQA 评测 |
| [STeP: Signal Temporal Logic for Precise Specifications for Action Generation with Vision Language Models](https://arxiv.org/abs/2607.18580) | 由语言生成的 Signal Temporal Logic 约束 | 以统一形式连接高层任务分解、低层控制、执行监控与重规划，使空间、时间和逻辑要求可检查 |
| [Diffusion ReRoll: Revisable Denoising for Robotic Sequential Prediction](https://arxiv.org/abs/2607.19919) | 对已稳定序列区域选择性重新加噪 | 让早期与未来片段在去噪过程中相互修正，统一改善长程规划、动作序列和视频—动作预测 |
| [Koopman Dreamer: Spectrally Constrained Latent Dynamics for Stable World-Model Imagination](https://arxiv.org/abs/2607.19719) | 带谱约束的 Koopman 潜在动力学 | 通过有界旋转—缩放模态、动作双线性项和多步目标控制长期想象的误差放大与信息保留 |
| [PhysCoRe: Physics-Based World Modeling with Residual Correction for Robotic Manipulation](https://arxiv.org/abs/2607.20653) | 可微 MPM、逐粒子材料参数估计与动力学残差修正 | 从少量交互中学习可校正的物理世界模型，并以不确定性引导探索和操作规划 |
| [GS-Agent: A Multi-Agent System for Physically Plausible 4D World Generation](https://arxiv.org/abs/2607.21522) | 资产、材质、布局、运动与渲染代理协作 | 将语言任务转化为受物理引擎约束的动态 4D 场景，为具身世界模型的数据生成与未来模拟提供工具链 |
| [Action-Conditioned World Model for Goal Plane Probe Guidance in Robotic Ultrasound](https://arxiv.org/abs/2607.21918) | 近期超声帧、探头动作与时间偏移 | 以潜在条件 diffusion 预测未来超声观测，再用冻结世界模型的奖励训练目标条件动作序列并执行真机闭环引导 |
| [ViTacWorld: Scaling Visuo-Tactile World Models for Contact-Rich Robot Manipulation](https://arxiv.org/abs/2607.22530) | 动作条件的视觉—触觉未来轨迹 | 联合预测时间对齐的图像和触觉反馈，以生成 rollout 扩充接触操作训练数据并评估策略结果 |
| [Robot-Factored World Models via Robot Rendering](https://arxiv.org/abs/2607.22535) | 控制器展开的名义轨迹与渲染机器人几何 | 将动作实现和机器人外观从生成模型中因式分解，避免未来状态泄漏，并以统一视觉接口泛化到未见机器人 embodiment |
| [$N_0$-TWAM: Scaling Tactile-Native World-Action Model for Contact-Rich Manipulation](https://arxiv.org/abs/2607.23783) | 未来视觉、接触与动作；NeoForce 统一力觉表示 | 在 6 种 embodiment、450 个任务上进行视觉—触觉联合预训练，并以接触事件推进长程多阶段操作 |
| [LeapBot-WA: World-Anchor Action Models via Predictive Latent Alignments](https://arxiv.org/abs/2607.23969) | foundation latent 空间中的预测语义和物理动态 | 以 JEPA World-Anchor 和 ISAE 替代像素级未来生成；训练时用动力学专家指导动作 diffusion，部署时裁除专家实现零额外开销 |
| [FutureRTC: Real-Time Robot Execution with Anticipatory-Conditioned Action Chunking](https://arxiv.org/abs/2607.24008) | 动作真正执行时的视觉表征与本体状态 | 用状态校正和 motion-aware 特征传输弥补异步 VLA 的预测—执行错位，并以策略一致性损失平滑连续 action chunks |
| [DeVA: Decoupled Video-Action Model with Physical Guidance for Robot Policy Learning](https://arxiv.org/abs/2607.24159) | 视频未来表征、affordance 与深度 | 将视频和动作预测拆为专用专家，通过多层特征传递把时空动态知识注入动作策略，提高小数据学习和真机泛化 |
| [FeelWorld: Visuo-Tactile World Model for Hierarchical Contact Prediction and Planning](https://arxiv.org/abs/2607.24267) | 未来视觉 latent、接触状态、3D 力觉 latent 与滑动状态 | 以 contact-gated attention 在自由空间和接触阶段切换预测路径，并通过自回归 rollout 支持接触感知 CEM 规划 |
| [τ: Learning Touch-Augmented Vision-Language-Action Models from Future Visual Supervision](https://arxiv.org/abs/2607.24485) | 动作条件的时空触觉表示 | 用 JEPA 风格未来视觉 latent 监督触觉表征并融合 VLA；监督仅在训练期间使用，不增加部署计算 |
| [VisualPatchWorld: Code World Models as Latent Structured Representations for Planning](https://arxiv.org/abs/2607.25236) | 主动探测、状态—动作轨迹与可执行动力学代码 | 先选择定性动力学形式再拟合自由参数，得到可检查、可编辑且可 rollout 的代码世界模型供 MPC 使用 |
| [Temporal-Distance JEPA: Plan-Aware Representation Learning for Latent World Model Predictive Control](https://arxiv.org/abs/2607.25337) | 轨迹先后关系、跨轨迹负样本与 rollout consistency | 从无奖励日志学习有方向的时间距离，使 latent 同时表达目标进度并直接充当多步规划 cost |
| [SAM3D-Guided Object-Centric Representation Alignment for Vision-Language-Action Models](https://arxiv.org/abs/2607.25912) | SAM3D 对象级 3D 教师表征 | 训练时将细粒度 3D 对象先验对齐到 π₀ 中间视觉特征，部署仍只用 RGB—语言输入，并改善跨子任务目标切换 |
| [DC-WAM: Dynamic-Centric Visual Supervision and Reasoning for World-Action Models](https://arxiv.org/abs/2607.25918) | temporal-difference flow、轨迹加权与动态相关 future tokens | 将未来监督从外观重建重定向到夹爪、对象和接触区的交互动态，并以 DynaRoute attention bias 聚焦控制相关变化 |
| [INTACT: Isomorphic Intent-to-Action Learning for Search-Free World Models](https://arxiv.org/abs/2607.26056) | 局部物理意图 $z_{t+1}-z_t$ 与目标意图 $z_g-z_t$ | 用共享 JEPA 表示把未来 latent 差分直接映射为动作分布，免除传统世界模型的大规模候选序列搜索 |
| [ContactFlow: Learning Actionable 3D Interaction Flow from Human and Robot Videos](https://arxiv.org/abs/2607.26579) | 人与物体之间、跨 embodiment 的 3D 接触点轨迹 | 用人类与机器人视频训练世界模型，以 propose–imagine–verify–act 闭环预测并验证可执行的交互运动 |
| [Enfold: Distilling Future into the Present for Efficient Visuomotor Policies](https://arxiv.org/abs/2607.26657) | 未来生成器多层中间状态蒸馏得到的当前时刻预测表征 | 仅在训练期使用未来生成器，部署时以当前观测保留未来信息并降低 3.7–10.1 倍推理延迟 |
| [ActSWM: Learning Action-Sensitive World Models for Embodied Agents](https://arxiv.org/abs/2607.26712) | 对替代动作保持可区分性的未来 latent | 通过动作可恢复的局部转移约束缓解 context collapse，使世界模型保留决策所需的动作因果差异 |
| [CheckVLA: Event-Driven World-Model Verification for Vision-Language-Action Policies](https://arxiv.org/abs/2607.26789) | 动作条件世界模型与事件驱动关键帧库 | 以 conformal 阈值核验执行结果、重写动作后缀，并用关键事件记忆维持长程任务进度 |
| [Temporally Centered SIGReg for Long-Horizon Vision-Language-Action Learning](https://arxiv.org/abs/2607.26924) | 相对时间中心的 latent residual | 只正则化时间残差而非完整 latent 边缘分布，减少跨任务表示混叠并提升长程操作性能 |
| [What Can Latent World Models Know? Physical Parameter Identifiability from Multimodal Prediction](https://arxiv.org/abs/2607.27017) | 由视觉、触觉及预测目标共同决定的物理属性 latent | 证明输入决定可识别信息、预测目标决定保留信息；只有要求预测触觉时，表征才可靠编码刚度等隐含物理参数 |
| [TacWAM: Anchor-Guided World Action Model with Mechanics-Aware Tactile Prediction](https://arxiv.org/abs/2607.28391) | 未来视觉、触觉、稠密力场、形变流与动作 | 以 mechanics-aware tactile prediction 表达接触阶段视觉难以观测的力、形变、剪切与滑动，同时阻止未来触觉成为动作生成的特权信息 |
| [World Action Planner: Generalizable Decision-Making with Action-Conditioned World Models](https://arxiv.org/abs/2607.27599) | VLM 动作提案与 pose-image 条件未来 rollout | 在世界模型想象结果上进行优化和搜索，迭代修正初始计划以提升新场景和组合任务中的决策泛化 |
| [QuantWAMs: Calibrating at the Right Granularity for World Action Models](https://arxiv.org/abs/2607.28405) | 面向闭环 rollout 分布的量化校准 | 按模型结构、部署轨迹和任务目标选择校准粒度，降低世界—动作联合预测的迭代去噪与闭环执行成本 |
| [WCM: A World Critic Model for Vision-Language-Action Reinforcement Learning](https://arxiv.org/abs/2607.29613) | 过去 $K$ 帧历史、动作条件下一 latent 与价值联合监督 | 以 instruction-conditioned causal Transformer 形成可更新 predictive state，使 VLA 的 RL critic 在部分可观测任务中学习跨时刻动力学 |
| [FBFM: A Training-Free Asynchronous Feedback Mechanism for Flow-Matching in World-Action Models Execution](https://arxiv.org/abs/2607.29235) | 前一动作 chunk 与执行后的真实图像 | 在下一 chunk 的 flow-matching 生成过程中异步注入细粒度真实反馈，持续 re-ground 并抑制长时程预测漂移 |
| [BWM: A Low-Cost High-Fidelity World Simulator for Robot Learning](https://arxiv.org/abs/2607.29302) | 初始环境锚点、动态视觉历史与时间对齐动作 | 进行有状态自回归未来预测，并作为 imitation-learning 数据引擎和闭环策略评估、风险预判与排名工具 |
| [ValueFormer: A Causal Transformer Value Function with Stage-Aware Labels for Semi-Autonomous Vision-Language-Action Policies](https://arxiv.org/abs/2608.02958) | rollout 时间上下文与阶段感知逐帧价值标签 | 同时输出平滑的优势估计和尖锐的错误检测信号，以因果 critic 表示任务进度并定位长程执行失败 |
| [UniNav: A Unified World-Action Diffusion Model for Visual Navigation](https://arxiv.org/abs/2608.03244) | 历史帧、目标图像、未来视觉与 waypoint 联合监督 | 在同一 diffusion Transformer 中联合生成未来观测和连续导航轨迹，并以 geometry-aware camera tokens 增强空间 grounding |
| [LiLa-WAM: Lightweight Latent Reasoning World-Action Model for Robotic Manipulation](https://arxiv.org/abs/2608.03701) | 未来状态预测与动作生成共同塑造的紧凑 latent | 以 Visual Transition Token 表达无语言任务方向，可在单张 24 GB GPU 上端到端训练世界—动作模型 |
| [EmbodiedVAE: Disentangled Video VAE for Efficient and Controllable Embodied Manipulation](https://arxiv.org/abs/2608.02990) | 解耦机械臂运动与背景的紧凑视频 latent | 通过非对称时空压缩和最优传输一致性约束，为操作世界模型保留可控动作变化与跨帧运动连续性 |
| [Unified Visuomotor Targets: Supervising VLAs Beyond Physical Actions](https://arxiv.org/abs/2608.03563) | 联合编码动作控制与视觉场景转移的 latent target | 不改变架构或增加数据，通过预测统一视觉运动目标提升 VLA 的训练效率、策略性能和环境鲁棒性 |
| [Track4Action: Distilling World-Centric 3D Tracker into Vision-Language-Action Policies](https://arxiv.org/abs/2608.03727) | 未来 $K$ 帧中的世界中心 3D 转移特征 | 训练时将几何、运动、可见性和相机变化蒸馏进当前观测 VLA，部署时移除未来 clip 与 tracker |
| [GORDON: Graph-based Object-centric Rewards for Decomposition of Long-Horizon Manipulation](https://arxiv.org/abs/2608.03753) | 对象—空间关系图与任务进度 latent | 从无动作视频学习稠密进度奖励，并由奖励时间曲线自动发现对象状态转移和长任务子阶段 |

### 5.2 TBPTT、梯度截断与完整轨迹训练

方法的前向历史范围与反向梯度范围并不相同。递归状态可在整个 episode 内传播，同时只对最近若干步保留计算图；也可完全停止跨步梯度，或在轻量时序模块上执行完整轨迹反传。

| 训练范式 | 代表方法 | 说明 |
|---|---|---|
| Temporal TBPTT | [μVLA](https://arxiv.org/abs/2606.12497)、[AVA-VLA](https://arxiv.org/abs/2511.18960) | 状态跨物理时刻传递，梯度只回传固定步数 |
| 极短 horizon/固定更新 | [ReMem-VLA](https://arxiv.org/abs/2603.12942) | 长期传播由固定 EMA 完成，不依赖跨步梯度 |
| Detached streaming state | [VPWEM](https://arxiv.org/abs/2603.04910) | 前向信息可覆盖完整 episode，递归 cache 写入时停止梯度 |
| Full-trajectory backpropagation | [Chronos](https://arxiv.org/abs/2606.30318) | 完整梯度保留在轻量 temporal-action stack |
| Trajectory-wise full scan | [MTIL](https://arxiv.org/abs/2505.12410)、[DSSP](https://arxiv.org/abs/2605.14598) | 对完整轨迹执行因果扫描 |
| Recurrent-depth truncation | [Recurrent-Depth VLA](https://arxiv.org/abs/2602.07845) | 截断动作预测内的迭代深度，不是长期时间记忆 |
| Training-free memory | [TempoFit](https://arxiv.org/abs/2603.07647) | 不学习跨时刻状态，直接复用中间层 KV |

## 6. 数据集与评测基准

记忆方法应在真正需要历史的部分可观测任务上评估，而非仅延长近似马尔可夫任务的操作时间。常见能力包括对象位置、任务进度、短暂证据、隐含规则、程序和周期计数。

### 6.1 机器人操作专用记忆基准

| 基准/论文 | 规模与记忆类型 | 环境 | 评测特点 |
|---|---|---|---|
| [RoboMME](https://arxiv.org/abs/2603.04639) | 16 个任务；temporal、spatial、object、procedural | 仿真；1,600 条示范、约 77 万帧 | 在统一 π0.5 backbone 上比较 14 种表示—融合变体，并提供 no-memory、symbolic oracle 与 memory-budget 对照 |
| [RMBench](https://arxiv.org/abs/2603.01229) | 9 个任务；5 个 `M(1)` 与 4 个 `M(n)` | 仿真 benchmark，另有真机实验 | 区分固定少量历史与重复探索/试错记忆；主要指标为完整任务成功率 |
| [MIKASA / MIKASA-Robo](https://arxiv.org/abs/2502.10550) | 32 个桌面任务、12 个任务组；object、spatial、sequential、capacity | ManiSkill 仿真 | 提供显式 oracle information、可参数化难度及专家轨迹，覆盖 online/offline RL 与 VLA |
| [MemoryBench / SAM2Act+](https://arxiv.org/abs/2501.18564) | 3 个任务；空间记忆与动作回忆 | RLBench/CoppeliaSim 仿真 | Reopen Drawer、Put Block Back、Rearrange Block；任务直观但范围较窄 |
| [LIBERO-Mem](https://arxiv.org/abs/2511.11478) | 10 个非马尔可任务；object、order/sequence、counting | LIBERO 仿真 | 对象级部分可观测，关键证据与后续决策可跨数百帧 |
| [MemoryRTBench / MemoAct](https://arxiv.org/abs/2603.18494) | 4 个主要仿真任务，另有 2 个真机任务；状态跟踪与长期保持 | RoboTwin 2.0 与真机 | 分别诊断无损短期记忆和压缩长期记忆 |
| [RoboTwin-MeM / EventVLA](https://arxiv.org/abs/2606.20092) | 17 个仿真任务与 4 个真实双臂任务 | RoboTwin 与真机 | 关注交互中短暂出现、随后消失的视觉证据 |
| [RuleSafe / VQ-Memory](https://arxiv.org/abs/2603.09513) | key、password 与 logic lock | articulated-object 仿真 | 用规则驱动的多阶段任务测试任务阶段、本体感觉与逻辑记忆 |
| [RoboMemArena](https://arxiv.org/abs/2605.10921) | 26 个任务，平均轨迹超过 1,000 environment steps | 仿真与 5 个配对真机任务 | 68.9% 子任务被标注为 memory-dependent；提供子任务指令和原生关键帧标注，并报告 TSR/CSR |
| [ReMemBench / PRISM](https://arxiv.org/abs/2606.16178) | 8 个家庭操作任务；spatial、prospective、associative、object-set | 仿真与真机适配 | 将短期记忆扩展到约 2 分钟，并研究 history length、注意力和计算成本 |
| [KineBench](https://arxiv.org/abs/2607.19876) | 20 个 ManiSkill3 操作任务；具身世界模型闭环评测 | 从生成视频提取 6D 末端位姿并在物理模拟器执行 | 避免 IDM 误差混淆，联合任务成功、轨迹平滑度和运动学可行性评估未来世界预测 |
| [AXIS: A Scalable Data Engine for Generalist Robot Manipulation](https://arxiv.org/abs/2607.21588) | 207 个任务、超过 5 万条轨迹；可扩展机器人数据基础设施 | 浏览器遥操作与自动化任务生成、验证、过滤、轨迹平滑和增强 | 持续预训练 π0.5 后平均提升 5.8%，相对 RoboCasa365 提升 37.3%，用于检验数据覆盖对通用操作能力的影响 |
| [RoboMME-Interference: Benchmarking Robot Memory Under Interference](https://arxiv.org/abs/2606.22338) | 跨 session 相关示范与可控数量的无关历史 | RoboMME 长上下文扩展 | 感知记忆随干扰 session 增加而持续衰减；新版表明先按视觉相似度检索相关示范可在各干扰级别恢复无干扰成功率 |
| [ACE-Data-0: Human-Centric Ambient Capture as Embodied Data Engine](https://arxiv.org/abs/2607.28625) | 150 小时、1,700 万帧、75,000 个 episode；长程多模态交互 | 200 类任务、50 名参与者、2 个真实家庭环境；同步第一/第三视角、全身和手部运动、对象 6-DoF、音频与触觉 | 分层评测从感知信号、场景组件扩展至完整交互，暴露接触、遮挡、自运动和长时程条件下的能力缺口 |

### 6.2 具身导航、个性化与持久世界状态

这类基准直接评估记忆和行动的结合，但输出通常是导航动作、高层计划或问答，而非机械臂低层控制，因此不与操作基准等权比较。

| 基准 | 主要范围 | 行动与输出边界 |
|---|---|---|
| [FindingDory](https://arxiv.org/abs/2506.15635) | Habitat 中 60 类可扩展长程任务，历史约 400–3,500 帧 | 评测阶段执行离散导航动作；物体操作由历史采集阶段的 oracle 完成 |
| [MEMENTO](https://arxiv.org/abs/2505.16348) | 个性化 object rearrangement；对象语义、用户习惯与联合记忆 | 两阶段 acquisition/utilization，以含糊指令测试 episodic/personalized memory |
| [LMEE-Bench](https://arxiv.org/abs/2601.10744) | HM3D 中 1,982 个长期记忆导航与问答任务 | 输出导航动作、frontier 或答案，不含机械臂控制 |
| [MemoryEQA / MT-HM3D](https://arxiv.org/abs/2505.13948) | 1,587 个跨目标、跨区域 active EQA 任务 | agent 主动导航收集证据后回答问题 |
| [RoboBench](https://arxiv.org/abs/2510.17801) | 5 个维度、14 项能力、25 类任务与 6,092 个 QA；含时空推理和 memory-driven navigation | 评估 MLLM embodied brain 的理解、感知推理、规划、affordance 与失败分析，不直接评测低层闭环控制 |
| [BEHAVIOR Robot Suite](https://arxiv.org/abs/2503.05652) | 真实家庭环境中的双臂、移动和躯干全身操作平台 | 提供长距离导航及 articulated/deformable object 任务；可承载长程记忆评测，但不是专用记忆基准 |
| [MultiON](https://papers.nips.cc/paper/2020/hash/6e01383fd96a17ae51cc3e15447e7533-Abstract.html) | 按指定顺序寻找多个目标 | 经典空间/语义地图记忆与闭环导航基准 |
| [WorldLines](https://arxiv.org/abs/2606.18847) | 跨日 household traces、状态覆盖与用户历史 | 评估 Memory QA 和 embodied task planning，不执行低层闭环动作 |
| [Beyond Episodic Evaluation: Memory Architectural Bottlenecks in Sequential Embodied Question Answering](https://arxiv.org/abs/2607.21571) | 同一场景连续提问并跨查询保留记忆；比较占据图与持久视觉—语义 3D 记忆 | 揭示仅保留二维可通行性不足以回答后续问题；将视觉证据绑定度量 3D 几何可同时改善问答准确率与导航效率 |
| [VoLN: Vision-Only Long-Horizon Navigation](https://arxiv.org/abs/2607.21400) | 7,210 个连续 3D 无人机长程导航 episode；大视点变化与上下文相关信标选择 | 仅用视觉、目标视图、观测历史、检索式视觉—语义 token 和本体状态进行导航，诊断长程视觉记忆与目标识别能力 |
| [Zero-Shot Mission-Level Evaluation for Aerial MLLM Agents (MissionBench)](https://arxiv.org/abs/2607.22014) | 5 个模拟 3D 环境、4 类共 120 个长程空中任务 | 智能体仅依靠第一视角观测与动作历史自主规划、导航和报告；22 个 MLLM 中最强成功率仍低于 35% |
| [Mag4D-SLAM Dataset: A Repeated-Traversal Multi-Modal 4D Geomagnetic Dataset for Localization and Mapping](https://arxiv.org/abs/2607.21986) | 超过 18 km 的昼夜、正反向重复遍历；LiDAR、相机、IMU、磁力计与 GNSS | 面向跨 session 地点识别、磁航向约束、回环检测与视觉退化条件下的长期定位 |
| [HumanCLAW: A Benchmark for Long-Horizon Egocentric Find–Navigate–Interact Tasks](https://arxiv.org/abs/2607.27180) | 41 个真实场景中的 1,218 条长程第一视角任务轨迹 | 联合评测搜索、导航、交互和进度记忆；9 个 VLM 中最佳成功率仅 16.8%，主要瓶颈是身体、碰撞与任务进度的自我状态追踪 |
| [Embodied Agents Take Control: General-Purpose Software Agents for Zero-Shot Visual Navigation](https://arxiv.org/abs/2607.26148) | 持续 perceive–act–verify–correct 的零样本具身控制循环 | 通用软件智能体可直接闭环导航，但长任务中延迟和上下文持续增长会导致性能显著下降，揭示长程记忆与上下文管理瓶颈 |

### 6.3 通用 POMDP 与记忆评测协议

| 基准/协议 | 用途 | 与机器人操作的距离 |
|---|---|---|
| [Memory Maze](https://arxiv.org/abs/2210.13383) | 随机 3D maze、online RL、offline dataset 与 memory probing | 有闭环动作，专门隔离长期定位记忆，但不是操作任务 |
| [POPGym](https://arxiv.org/abs/2303.01859) | 15 个 POMDP 与多种 recurrent/attention baseline | 通用抽象控制，可用于比较记忆架构 |
| [Memory Gym](https://arxiv.org/abs/2309.17207) | 2D POMDP、endless variants、长度泛化与噪声鲁棒性 | 诊断性强，但非具身机器人环境 |
| [POBAX](https://arxiv.org/abs/2508.00046) | 筛选 memory-improvable 环境并比较 observation-only/state gap | 适合作为部分可观测评测方法学参考 |
| [MEMBOT Intermittent-POMDP Protocol](https://arxiv.org/abs/2509.11225) | 向 MetaWorld 与 RoboMimic 的 10 个任务注入 observation dropout | 是基于已有环境的鲁棒性协议，不是独立 benchmark |
| [RoboMME-Interference](https://robotmemorybench.com/) | 在相关 session 前加入 0/1/3/7 个无关 session | 显式测试跨 session 干扰、历史距离和记忆污染 |

### 6.4 非马尔可性判据与统一报告建议

仅增加轨迹长度或输入帧数不能证明任务需要记忆。较严格的 benchmark 应至少满足以下判据：

1. **State-aliasing pair**：存在当前观测近似相同、但因历史不同而要求不同最优动作的状态对。
2. **History intervention**：固定当前状态，仅删除、替换或打乱关键历史时，正确动作或目标随之改变。
3. **Matched controls**：同时报告 no-history、matched-compute recent window、random/shuffled/wrong memory 与 full-history 对照。
4. **Oracle decomposition**：分别提供 privileged-state oracle、最小充分记忆 oracle，以及必要时的 oracle low-level controller，以区分检索、推理与控制失败。
5. **Scaling curves**：按证据延迟、需保存事实数量、遮挡持续时间、memory budget 与 distractor 数量报告性能曲线，而非只报告平均轨迹长度。

建议统一报告完整任务成功率、子任务完成率/CSR、记忆关键决策点准确率、至少 3 个随机种子与置信区间，并披露训练/测试的场景、物体、布局、语言模板和生成器 seed 隔离方式。系统成本应包含 memory token/frame 数、最大与平均 context、显存、延迟和控制频率。可进一步报告：

\[
\text{Memory Gap}=S_{\text{oracle-memory}}-S_{\text{no-memory}},
\]

\[
\text{Normalized Memory Recovery}=
\frac{S_{\text{method}}-S_{\text{no-memory}}}
{S_{\text{oracle-memory}}-S_{\text{no-memory}}}.
\]

常见评测陷阱包括当前画面残留答案、机械臂姿态或固定动作时长泄露阶段/计数信息、语言模板和 GT subtask/keyframe 标注泄露任务进度、不同上下文长度导致计算量不匹配，以及完整任务成功率将记忆错误与感知、规划和低层控制错误混合。

## 7. 方法比较与研究趋势

### 7.1 方法比较维度

| 维度 | 关键问题 |
|---|---|
| 作用范围 | 记忆服务于当前 episode、跨示范检索，还是终身知识管理？ |
| 表示单元 | 历史被表示为帧、token、关键事件、对象、动作、语言还是递归状态？ |
| 容量与成本 | 计算、显存、存储与推理延迟如何随 episode 长度增长？ |
| 写入与遗忘 | 系统如何判定重要性、冗余性、时效性和替换优先级？ |
| 读取与融合 | 历史是通过全量注意力、相似度检索、交叉注意力、门控还是状态传播注入策略？ |
| 训练梯度 | 前向历史范围与反向梯度范围是否一致？ |
| 评测有效性 | 任务是否确实需要早期证据，并能区分记忆收益与模型规模收益？ |

### 7.2 开放问题

1. **重要性与可压缩性的冲突**：短暂出现的证据可能低频但决定任务成败，而重复背景虽高频却信息量低。
2. **前向记忆与长程信用分配**：保留完整前向历史并不意味优化信号能到达早期状态。
3. **多模态一致性**：视觉、动作、语言、本体感觉与空间地图可能具有不同采样频率和遗忘规则。
4. **评测标准化**：尚需统一报告记忆容量、计算开销、推理延迟、历史跨度与不同记忆类型的成功率。
5. **从 episode 内记忆到终身学习**：如何避免长期知识污染、错误累积和不相干经验干扰，仍缺少统一解法。

## 8. 相关综述与资源

| 论文 | 覆盖范围 | 用途 |
|---|---|---|
| [大模型记忆系统分析框架](docs/llm-memory-system-framework.md) | 参数化/显式记忆、系统分层、原子操作、生命周期、分类与机器人映射 | 作为后续论文方法卡片和统一分析维度 |
| [Large VLM-based VLA Models for Robotic Manipulation: A Survey](https://arxiv.org/abs/2508.13073) | 大型 VLM-based VLA 的架构、训练、world model、memory 与效率 | 完善 VLA related-work taxonomy |
| [A Survey on VLA Models for Embodied AI](https://arxiv.org/abs/2405.14093) | VLA 方法、数据、基准与能力 | 持续追踪领域进展与查漏补缺 |
