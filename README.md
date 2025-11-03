# How to Get Started with Embodied AI Research

© PKU EPIC Lab. All rights reserved. Commercial distribution prohibited.<br>
© PKU EPIC Lab. 版权所有。禁止商业传播。

[原文主页](https://jiangranlv.notion.site/How-to-Get-Started-with-Embodied-AI-Research-252a467c9b60804d85dcfeffcc7771fb)

**作者：吕江燃，张嘉曌，邓胜亮，陈嘉毅，严汨，李忆唐**<br>
**指导老师：王鹤，弋力**<br>

## O、前言

随着具身智能的关注度不断升高，越来越多的研究者涌入这一领域，相关论文数量也呈现井喷式增长。然而，其中不少工作的质量令人担忧：有的只是单纯“讲故事”，有的则一味追求“刷榜”，但这些却反而获得了大量的关注和追捧。在这样的科研环境下，为新同学提供正确的引导，帮助他们少走弯路、健康成长，是一个重要的任务。

因此，我们撰写本文的目的，就是希望为刚进入具身智能科研领域的同学提供一个清晰的guide，帮助大家理解具身智能究竟该研究什么，以及如何正确地入门。希望能帮助到初学者积累必要基础知识的同时，建立起正确的研究认知。

本文章也将作为 PKU EPIC Lab 本科生的入门材料，并会在实践和培养的过程中不断更新和完善。

## 一、基础概念 (Basic Concepts)

### 1. 什么是具身智能 (What is Embodied AI)

具身智能（Embodied AI）是指能够在物理或虚拟环境中**通过感知、行动和交互**来学习与完成任务的人工智能。不同于仅在静态数据（文本、图像、语音等）上进行训练和推理的传统 AI，具身智能的智能体（agent）往往有一个“身体”（body）或“化身”（avatar），它们可以与环境交互，改变环境，并随着环境的改变自己作出调整。

典型的具身智能研究对象包括机器人和虚拟环境中的智能体，本文主要面向机器人领域(Robotics)。

**核心特征：**

- 拥有多模态感知能力（视觉、触觉、语音等）
- 能够执行动作并影响环境
- 学习可以通过**与环境交互**而不仅仅是被动监督完成

### 2. 具身智能与其他AI的区别 (Differences from Traditional AI)

具身智能与传统 AI 的主要区别在于它的**主动性、交互性，以及对动作数据的依赖**。传统 AI 可以利用互联网上丰富的图像、文本、语音等大规模数据集进行训练（参考LLM的成功），而具身智能体所需的动作数据必须通过与环境的真实交互来收集，这使得数据获取代价高昂且规模有限。一言以蔽之，数据问题是具身智能目前最大的bottleneck。那么很自然的两个关键问题是，

- 如何scale up机器人数据？
例如：[GraspVLA](https://pku-epic.github.io/GraspVLA-web/)（在仿真中以合成的方式猛猛造）, [pi0](https://www.physicalintelligence.company/blog/pi0)和[AgiBot-World](https://agibot-world.com/)（在真实世界猛猛遥操采）, [UMI](https://umi-gripper.github.io/)和[AirExo](https://airexo.github.io/)（可穿戴设备，如外骨骼的高效数据采集装置）
- 在不能scale up机器人数据的情况下，如何利用好已有的数据实现你的目的？
例如：[Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) (100条机器人数据训一个特定任务的policy）, [Being-H0](https://arxiv.org/pdf/2507.15597)（利用human video参与policy训练），[MimicGen](https://mimicgen.github.io/)、[DemoGen](https://demo-generation.github.io/)、[Robosplat](https://yangsizhe.github.io/robosplat/)（从一条机器人轨迹中augment得到更多数据）

### 3. 研究具身智能的核心原则 (Core Principles)

- **首先把任务定义（task formulation）想清楚，而不是一开始就盯着模型**。在CV领域，研究者之所以可以直接关注模型，是因为任务往往已经被定义得很清晰，数据集也由他人整理好， 比如图像分类就是输入图片输出类别标签，检测就是输出四个数的bounding box；
    
    但在具身智能中，如何合理地建模任务、确定目标与评价指标，往往比模型选择更为关键。说白了，你得知道你想让机器人学会什么样的技能，输入是啥，输出是啥，用的什么传感器？你所研究的问题是否在合理的setting下？有没有有可能通过更好的setting来解决问题（比如机器人头部相机对场景观测不全，那我们可以考虑加装腕部相机，或者使用鱼眼相机）
    
- 必须认识到**用学习（learning）来解决机器人问题并不是理所当然的选择**。在许多场景中，传统的控制（Control）、规划（Planning）或优化方法（Optimization）依然高效且可靠，而学习方法更多是在任务复杂、环境多变(泛化性) 或缺乏解析建模手段时才展现优势。因此，做具身智能研究时，首先要想回答，为什么你研究的这件事传统robotics解决不了？为什么非得用learning？

## 二、前置技能

这些工具是一个当代CS researcher的必备技能（不局限于方向），主要学习资料可以参考
 [https://missing-semester-cn.github.io/](https://missing-semester-cn.github.io/)

- Python, Conda and Pytorch
- Linux Shell, Git, SSH
- LLM, Cursor
- Docker

## 三、AI and Robotics Basis

以下三门课是基础课程，对于初学者希望能详细的掌握内容，不要“不求甚解”，对于课程Lab的project最好做到完整实现，而不仅局限于做“代码填空”。

### 1. [Intro-to-Embodied-AI](https://pku-epic.github.io/Intro2EAI_2025/schedule/)

实验室内部课程，主要内容是robotics的基础概念和基于learning的robotics，源于王鹤老师《具身智能导论》，外界同学自行寻找类似课程替代

### 2. [Intro-to-CV](https://pku-epic.github.io/Intro2CV_2025/schedule/)

实验室内部课程，主要内容是cv基础和deep learning，源于王鹤老师《计算机视觉导论》，外界同学可以学习Stanford CS231N替代

### 3. (Optional) Deep Reinforcement Learning (CS285)

Berkeley的RL课程，涵盖了Imitation Learning，Online RL, Offline RL等Policy Learning范式

## 四、研究平台与工具 (Research Platforms and Tools)

### 1. 模拟环境 (Simulation Environments)

**Simulation的意义**：在大多情况下，真机部署机器人是不方便的，所以simulation提供了一个很好的代替。主要有两大用途，1.作为高效的数据源，解决真实世界机器人数据少的问题 2.真机测试policy不方便，很难复现，作为一个更通用的benchmark evaluation平台

**对于simulation的掌握**：需要至少一种simulation框架，通过阅读tutorial跑他的example，加深对robotics的理解，不要等到上真机才发现有很多坑，会有很大的安全隐患

IssacLab (Recommend)

https://isaac-sim.github.io/IsaacLab/main/index.html

https://playground.mujoco.org/

### 2. 机器人平台 (Robotic Platforms)

真机实验下的机械臂通讯接口，至少熟悉一类常用的API

基于ros1的franka通讯：

[https://github.com/rail-berkeley/serl_franka_controllers](https://github.com/rail-berkeley/serl_franka_controllers)

### 3. Embodied AI Daily ArXiv  (Advanced)

具身智能每日最新的论文，按manipulation，VLA， dexterous，humanoid等关键词进行划分，推荐基础入门后的同学每日最终最新进展，丰富自己的认知和视野

[https://github.com/jiangranlv/robotics_arXiv_daily](https://github.com/jiangranlv/robotics_arXiv_daily)

## 五、对机器人技能的研究 (Research on Robot Skills)

### 1. Grasping

抓取（**Grasping**）是机器人学中最基础且最重要的任务之一，通常指让机器人末端牢牢抓紧物体以达到力闭合（**force closure**），成功完成抓取后可将物体视作机器人的一部分进行后续的移动和操作。

常见任务有（难度依次递增）：
- **Single object grasping（单物体抓取）**：抓取一个物体，物体通常放在桌子上。
- **Clutter scene grasping（堆叠场景抓取）**：抓取堆叠场景中的物体，通常要求清台（全部抓完）。难点在物体的互相遮挡和干扰。
- **Functional grasping（带语义抓取）**：根据语言指令进行抓取。对于单物体抓取而言，语言通常指定物体要抓的part和抓取的手势；对于堆叠场景而言，还可以指定要抓的物体。难点在语言模态的引入。

常用机械手末端有（难度依次递增）：
- **Suction cup（吸盘）**：控制维度最低，除了末端整体的旋转和平移的自由度之外，只有是否施加吸力的0/1控制信号。
- **Parallel gripper（平行夹抓）**：类似吸盘。学术上通常认为吸盘/平行夹抓+堆叠场景抓取已经被[DexNet](https://arxiv.org/pdf/1703.09312)和[GraspNet](https://openaccess.thecvf.com/content_CVPR_2020/papers/Fang_GraspNet-1Billion_A_Large-Scale_Benchmark_for_General_Object_Grasping_CVPR_2020_paper.pdf)两个系列工作几乎解决（思路：大规模仿真抓取位姿 + 学习位姿预测网络 + sim2real）
- **Multi-fingered hand（多指手）**，又称**Dexterous hand（灵巧手）**：更高的可控自由度和更高的潜力，但也极大地增加了数据构造与学习的难度，导致其发展远落后于前两者。大规模仿真抓取位姿的进展/Dataset：[DexGraspNet](https://pku-epic.github.io/DexGraspNet/)、[Dexonomy](https://arxiv.org/pdf/2504.18829)（覆盖多样化手型）。

常见的做法：
- **Open-loop methods（开环执行）**：通过一次性预测抓取位姿并直接执行，不依赖执行过程中的感知反馈。可以直观理解为“看一次决定怎么抓”，执行时全程不再依赖视觉，仅依靠运动规划达到目标位姿。因此开环方法的核心是 **grasping pose estimation**。**Data Source**：Grasp Synthesis，如 [DexNet](https://arxiv.org/pdf/1703.09312)、[GraspNet-1B](https://openaccess.thecvf.com/content_CVPR_2020/papers/Fang_GraspNet-1Billion_A_Large-Scale_Benchmark_for_General_Object_Grasping_CVPR_2020_paper.pdf). **Learning Approaches**：GSNet。
- **Closed-loop methods（闭环执行）**：在执行过程中持续使用视觉或触觉反馈进行动态调整，从而提升抓取的鲁棒性。这类闭环模型可视为 **policy**，持续输入视觉信息并输出机械臂动作。代表工作：[GraspVLA](https://pku-epic.github.io/GraspVLA-web/)。

### 2. Manipulation

操作（**Manipulation**）比抓取的含义更广，允许手和物体间有频繁的接触点变化，不像抓取任务中接触点形成后就固定不变了。通常只要是改变了物体状态的任务就可以叫操作。

- **Articulated Object Manipulation**：铰链物体操作（如开门、拉抽屉、开柜子）。该类任务通常被简化成抓取任务来处理：1.Part理解（[GAPartNet](https://pku-epic.github.io/GAPartNet/)）2.抓取（Grasping）3.抓取后的操作轨迹规划 4.拉取力度控制（Impedance Control）
- **Deformable Object Manipulation**：柔性物体操作（如叠衣服、挂衣服）。难点在于柔性物体自由度极高、难以精确建模和仿真。常见做法通常基于人工设计的原子操作（action primitives），最近也有一些公司（pai，dyna）开始用数采+端到端学习的方式来直接做。
- **Non-prehensile Manipulation**：非抓握操作，指通过推、拨、翻转等方式在无抓握的情况下操控物体至指定姿态。难点在于 **contact-rich** 的动力学特性，机器人、物体与环境存在多重接触与碰撞，如何生成成功的操作轨迹是当前研究重点。
- **Dexterous Manipulation**：灵巧操作，与non-prehensile类似，但通常有更多的contact和更高的控制维度。一个经典的任务是in-hand reorientation，虽然它已经几乎被RL解决，但如何提升学习效率、拓展到更一般的灵巧操作任务上依旧是研究难点。
- **Bimanual Manipulation**：双臂操作，重点在于如何实现双臂的协调与配合。
- **Mobile Manipulation**：移动操作，强调移动系统为操作提供更大、更灵活的工作空间，移动如何为操作服务，两者如何协同

### 3. Navigation

**Navigation** 导航研究机器人如何在物理环境中移动，以完成给定任务。导航能力是一种综合能力，从高层次来看，包括对视觉、深度信息和指令的理解，以及对历史信息（如地图、Tokens 等）的建模；从低层次来看，还包含路径规划与避障。导航通常涉及场景级别的移动，是硬件、传感器与控制算法综合能力的体现。

常见任务包括：

- **Point Goal Navigation (PointNav)**：给定目标点坐标或相对方向，机器人需从起始位置导航至目标点。不涉及语义理解，属于低层任务。
- **Object Goal Navigation (ObjectNav)**：根据目标物体类别（如“椅子”），在未知环境中寻找并导航至目标物体。
- **Vision-Language Navigation (VLN)**：根据自然语言指令（如“走到厨房的桌子旁”），结合视觉感知完成导航任务。
- **Embodied Question Answering (EQA)**：机器人需在环境中探索、感知并回答与场景相关的问题（如“卧室里有几张床？”）。
- **Tracking**：机器人持续感知并跟随动态目标（如人或移动物体）。

常见做法：
- **Map-based Navigation**, 基于地图的导航算法会利用深度图，里程计等信息构建地图，从而基于地图规划路径完成导航任务。基于地图的方法在静态或者易结构化的场景下表现非常好。相关工作包括: [Object Goal Navigation using Goal-Oriented Semantic Exploration
](https://arxiv.org/abs/2007.00643)
- **Prompting-Large-Model Navigation**，通过对物理世界进行解释得到prompting，然后以现成（off-the-shelf）的大模型作为规划决策的中心。这种方法不需要训练复杂的大模型，且可以利用大模型的智能优势实现复杂的导航任务。相关工作包括: [NavGPT](https://arxiv.org/abs/2305.16986), [CogNav](https://yhancao.github.io/CogNav/)
- **Video-based VLM Navigation**, 通过端到端训练基于视频输入的视觉语言大模型，通过tokens来建模导航历史，和用VLM直接输出未来导航动作。相关工作[NaVid](https://pku-epic.github.io/NaVid/)

**Unified Embodied Navigation**：最新研究趋势是将多种导航任务统一建模，常使用纯RGB输入，并将目标描述转换为语言指令。代表性工作：**[Uni-Navid](https://pku-epic.github.io/Uni-NaVid/)**，统一多种导航任务。**[NavFoM](https://pku-epic.github.io/NavFoM-Web/)**,统一导航任务和embodiment。

### 4. Locomotion

**Locomotion** 强调机器人在多样环境中的运动与机动能力。狭义上通常指基于 **Whole-body Control (WBC)** 的控制方法，用于实现 **四足（Quadrupedal）** 与 **双足（Bipedal / Humanoid）** 运动。

技术路线上，2019年以前主要靠传统的MPC控制实现（例如波士顿动力），目前主流的方法是Sim2Real RL, 以下主要讨论这类主流范式。
既然谈及RL，又分为
- **Learning from manually designed reward** (自己写reward提供desired behavior) ([WoCoCo](https://arxiv.org/pdf/2406.06005)【任务目的：通过reward设计让机器人完成某些特定任务】
- **Learning from human data** (data提供desired behavior，也叫做tracking)【主流】 (ASAP)【任务目的：模仿某一段人类数据中的动作（输入：现在的state和目标的state；输出这一步的action）】

如果人形机器人能完成对特定人类动作的tracking，那么接下来就有了一个很主流的研究方向，general motion tracking -> whole-body teleopration，人在做任何一段动作的时候，机器人可以复现人的动作（这里的难点就很多了，动作输入形式的多样性，减少延时，长程复现人的动作，复现的精准度）
这一系列的工作是H2O, OmniH2O, HOMIE, TWIST, CLONE, HOVER, GMT, Unitrack等等，至此Control最基本的问题应该well-defined了

下一个阶段会涉及到一点除了control之外的东西，就是
- 引入【视觉】实现户外自主化（perceptive locomotion）；例如，根据视觉来进行上楼梯，迈台阶，难点：vision sim2real 【visualmimic】
- 引入【物体】实现loco-manipulation；例如人型机器人搬箱子，难点：物体的dynamics【HDMI】
- 对上述两种task的组合
- 强调【语义的泛化性】，希望能根据各种各样的场景/物体【自主决策】做出相应的动作（whole body VLA）【leverb】
- 强调一些特殊的capability（比如HuB做极端平衡，Any2Track受很大的力干扰摔不倒, Hitter做一个特殊的乒乓球task，spi-active做sim2real对齐让机器人能走直线）


## 六、基于learning的主要研究方向

### 1. Imitation Learning

该方向主要聚焦于 **小模型 (small-model)** 场景：在给定数量有限的机器人轨迹数据集上，学习一个策略 (policy) 来完成特定任务，并在一定范围内实现泛化，例如在同一张桌面上对不同物体的泛化操控。

- **传统方法**：[Behavior Cloning](https://cgi.cse.unsw.edu.au/~claude/papers/MI15.pdf)、[DAgger](https://arxiv.org/abs/1011.0686)
- **当前主流方法**：[ACT](https://tonyzhaozh.github.io/aloha/)、[Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)
    
    这些方法通过引入时序建模与生成式策略学习，有效提升了模仿学习在视觉控制任务中的表现。
    

---

### 2. Vision-Language-Action Models (VLA)

该方向属于 **大模型 (foundation model)** 范式，旨在将视觉、语言与动作建模统一在同一框架下，实现通用的具身智能。

- **代表性工作**：
    - [OpenVLA](https://openvla.github.io/)：第一个开源且易于follow的VLA。
    - [Pi0](https://www.physicalintelligence.company/blog/pi0) / [Pi0.5](https://www.physicalintelligence.company/blog/pi05)：目前公认最work的VLA，10K+ hours teleop data训练的。
    - [GraspVLA](https://pku-epic.github.io/GraspVLA-web/)：基于纯仿真数据的抓取任务的VLA。
    - [RDT](https://rdt-robotics.github.io/rdt-robotics/)：纯diffusion的VLA架构

---

### 3. Sim-to-Real Reinforcement Learning (Distillation)

**从仿真到真实 (Sim-to-Real)** 是强化学习在具身智能中的关键挑战之一。

目前最成功的落地应用集中在 **Locomotion（运动控制）**，而在 **Manipulation（操作任务）** 上仍较少见。

核心思路通常包括 **策略蒸馏 (policy distillation)**、**域随机化 (domain randomization)** 与 **现实校准 (real calibration)** 等技术。

---

### 4. Real-World Reinforcement Learning

**Real-world RL** 指直接在现实环境中进行探索式学习。

这类方法通常用于解决高度挑战性的具体任务（如插入 USB），目标是将成功率优化至接近 100%。

- **从零开始的真实世界强化学习**：**Hil-Serl**
- **基于VLA的真实世界微调 (Fine-tuning)**：部分近期工作尝试利用预训练VLA进行现实强化学习微调，但仍处于早期探索阶段。

---


### 5. World Models

**World Model** 最早起源于 **基于模型的强化学习 (Model-based RL)**，旨在通过内部世界建模来提升采样效率。

代表性工作包括 **Dreamer 系列**（Dreamer, DreamerV2, DreamerV3），通过学习潜在动态模型，实现“在脑中想象未来”式的策略更新。

在具身智能的最新语境中，**World Model** 的概念被拓展为 **条件视频生成模型 (conditioned video generation model)**，用于模拟未来观测、预测任务后果，并与规划模块或语言模型结合以实现长期推理。

## 七、相关领域

### 1. Graphics

图形学在机器人与具身智能中的两大重要应用是 **simulation（仿真）** 与 **rendering（渲染）**。

- **Simulation**：用于搭建虚拟的物理交互环境，是机器人强化学习、控制算法和策略验证的重要工具。如上述IsaacLab等
- **Rendering**：用于生成高质量的图像或视频，支撑感知模型（如视觉Transformer）的训练与评估。例如：**Blender**：开源的三维建模与渲染软件。
- 系统性学习图形学推荐课程：**Games 101, 103**

### 2. Hardware

硬件是具身智能的“身体基础”，涵盖操作、感知与反馈等环节。

- **Tele-operation（遥操作）**
    - **末端操作设备**：如 *Space Mouse*，用于控制机械臂的末端姿态。
    - **主从臂系统**：如 *Gello*，实现高精度的力控遥操作。
    - **可穿戴设备**：如 *AirExo* 或 *UMI*，通过外骨骼或手部设备实现自然交互与示教。
- **Sensors（传感器）**
    - **Camera（视觉）**：RGB / RGB-D 相机，如 RealSense、ZED、Azure Kinect。
    - **Force Sensor（力传感器）**：用于检测接触力矩，常安装于末端。
    - **Tactile Sensor（触觉传感器）**：如 GelSight、DIGIT，用于捕捉表面接触信息。
- **Mocap System（动作捕捉系统）**
    
    用于精确追踪人体或机器人位姿，常用于收集示教数据或标定
    

### 3. Models

- **Transformer**
    

- **Diffusion、Flow Matching**
    由于能够有效建模多峰分布的生成模型sota。

### 4. Foundation Models
- **LLM（Large Language Model）**
    通过大规模文本训练获得强大的语言理解与推理能力，是具身智能中语言规划与高层决策的重要基石。代表模型包括：**GPT / Claude / Gemini**：通用语言推理模型。

- **Vision Encoder** 
    - [DINO系列](https://dinov2.metademolab.com/)：通过大规模的**自监督学习 (self-supervised learning)** 提取图像的细粒度语义表示，在机器人视觉任务中常用于特征提取与场景理解。
    - [CLIP](https://arxiv.org/pdf/2103.00020)：通过大规模的图文匹配对上的 **对比学习 (contrastive learning)** ，将图像与文本映射到共享的多模态语义空间，成为视觉语言理解的核心模型。

- **VLM（Vision-Language Model）**
    通过大规模的图文理解数据进行训练，获得强大的视觉语言理解能力，在机器人视觉任务中常用于VLA模型的初始化，或用于场景理解与任务规划。代表模型包括：[Qwen-VL系列](https://github.com/QwenLM/Qwen-VL)、[GPT4-o](https://arxiv.org/pdf/2410.21276)、[Gemini](https://arxiv.org/pdf/2403.05530)。


### 5. 3D Vision
详见Intro-to-CV课程，此处仅给出一些具身任务中常用的三维视觉技术。
- **三维生成与重建**
    - 相机标定：利用标定版构建多组约束，从而求解相机参数，常用于获取机器人坐标系与相机坐标系之间的变换矩阵。
    - 单目三维生成：根据单张RGB图片生成对应物体的三维几何，在real-to-sim中是一种常用的获得物体几何的方法。
    - 单目深度估计：通过单张RGB图片估计场景深度，常用于将互联网或是二维生成模型的输出结果转换为三维视觉信号。
    - 位姿估计与追踪：通过单张或多张RGB图片估计物体或相机的位姿，常用于提取二维图片或视频中的物体或是人手位姿，进一步作为action的一种表征。
- **三维表示**
    - 网格（Mesh）：通过三角形网格表示三维几何，物理仿真中最常用的三维表示方式。
    - 点云（Point Cloud）：通过物体表面的点的集合来表示三维几何。现有的点云处理网络具有很好的捕捉局部几何的能力，因此GraspNet使用点云作为输入，实现了非常鲁棒的抓取位姿预测。
    - Gaussian Splatting：通过高斯分布表示三维几何，由于其可微渲染与快速计算的特点，成为沟通二维与三维的桥梁。在real-to-sim中是一种常用的重建场景几何的表示。
- **三维理解**
    - 包括三维分类、场景分割、实例检测、空间推理等任务，常用于机器人视觉任务中的场景理解与任务规划。


## 八、(Optional) 科研工作中的必备能力

- Sharp Mind：戳穿别人工作的包装，看到本质<br>
- Writing and Presentation： 包装自己的工作，别让别人拆穿<br>
- Warm Mind：不要一味的批评别人的工作，能够欣赏到别人的亮点<br>

## 相关仓库
https://github.com/TianxingChen/Embodied-AI-Guide
