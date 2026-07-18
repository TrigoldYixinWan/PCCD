# PCCD 实验反思与 research routine 追溯

**日期：** 2026-07-17

**性质：** 全项目证据复盘；不改变任何冻结 verdict，不重新解释 lockbox，不启动新实验。

## 1. 总体结论

PCCD 最初试图建立一条完整、强因果、可修复的安全校准故事：一个在基础分布上表现良好的冻结 critic，会在 policy 被本地适配后发生异质校准退化；这种退化主要表现为危险的 false-negative 增长，可以由 KL shift 预测，并能通过 per-policy calibration 修复。

最终证据没有支持这条完整链条。基础校准锚点 P1 稳定成立；发现性 D5 数据显示了平均 ECE 上升和明显的准则异质性，但独立 lockbox 没有复现正的平均退化。FN 非对称方向显著相反，KL 只有关联而没有达到预注册预测门槛，source-only 与 low-shot temperature scaling 也未达到恢复门槛。原始研究中最稳定的信号变成了“不同安全准则的校准变化方向不同，而且相对排序跨两个 D5 seed 稳定”，但该信号因 confirmatory gatekeeping 和 Qwen reference construct-validity 限制，只能作为强 secondary evidence。

随后启动的三-guard BeaverTails 外部研究也没有确认“良好平均校准掩盖最差准则”“正负准则误差抵消”或“mean/worst 导致 guard 排名翻转”。它确认的是一个更窄、不同的命题：对 ShieldGemma-2B/9B，分类别 calibration audit 会随 human reference 与 blind Qwen-32B proxy 标签源系统变化，且变化沿预注册的 objective/subjective 类别轴显著；Llama Guard 不支持同方向效应。因此最终外部 verdict 是 `LABELSOURCE_ONLY`。

这不是一次没有产出的项目。它产出了三个可靠层次的证据：稳定的基础校准锚点；对多个直觉命题的高质量负结果；以及一个预注册、经多重性校正的标签源敏感性正结果。真正的问题是，最初论文故事的野心远大于前期构念验证和独立证据所能承载的范围。

## 2. 初始研究逻辑

最初的逻辑链为：

1. Qwen2.5-32B 作为 label-only teacher，为十项政策 H1-H5/S1-S3/T1-T2 产生三分类 reference labels。
2. Qwen2.5-7B shared-backbone、十个 per-policy head 的 critic 在基础分布上训练一次并冻结。
3. 独立 policy model 经 D1-D6 的 prompt、LoRA、SFT、DPO 适配产生分布偏移。
4. 若冻结 critic 在适配分布上失准，则比较各 policy 的 ECE、F1、FN、FP。
5. 以 KL(adapted||base) 作为统一 adaptation-strength 轴拟合 scaling law。
6. 以 per-policy temperature scaling 作为低成本部署修复。

这条逻辑在概念上完整，但一次性绑定了六个并不必然共同成立的命题：基础校准、平均退化、准则异质性、FN 主导、KL 可预测性、轻量可修复性。任何中间环节失败都会击穿后续叙事，而项目最初没有先证明 measurement construct 和 intervention manipulation 足以支撑这条链。

## 3. 证据演化时间线

| 阶段 | 原计划问题 | 实际证据 | 决策后果 |
|---|---|---|---|
| Gate D / Day 2 | 32B teacher 是否可运行，数据是否完整 | Gate D 吞吐正式 PASS；10,000 条 train/calib/test 标签 JSON 100%，零 leakage；H5 定义 bug 被发现并两次受控重标 | 工程与数据完整性成立，但暴露 taxonomy 设计没有在采样前充分验证 |
| Day 3 G1 | teacher 和十政策空间是否可靠、异质 | fixed prompt repeat 98.85%；order 76.28%、paraphrase 73.56%；L1 PARTIAL；L2 44/45 | teacher 只能作为固定 production oracle，不能宣称措辞/顺序稳健 |
| Day 4 D0 | critic 是否有良好 base anchor | mean ECE 0.02890，adaptive-ECE 0.02971；mean violated-F1 0.8745 | P1 SUPPORTED，且后来独立 lockbox 再次支持 base anchor |
| Day 4 L3 | base critic 是否已呈政策异质 | F1 CV 0.08057，CI [0.06468, 0.11145]，低于 0.15 | L3 FAIL；原 gate 把“base head 不均匀”误当成“适配后变化会异质” |
| D3 diagnostic | safe-SFT 是否产生预期的危险失准 | KL 0.6508；mean ΔFN-ΔFP = -0.303，方向相反；base violated support 仅 3-10 | 识别 support-shift confound 和 intervention 方向错误；D3 保留为 benign control |
| G2 D1-D6 | 平均退化与 FN 非对称是否同时成立 | D5 mean ΔECE 0.02883、RMS 0.04765、SD 0.03794；但 mean ΔFN-ΔFP = -0.27781，CI 完全低于 0 | discovery P2/P3 支持，P5 结论性失败；完整 G2 FAIL |
| G3 | KL 是否预测 ΔECE | 正斜率显著、in-sample R² 0.7755，但 LODO R² 0.6318 < 0.70；hidden-SFT-only 非单调 | P6 FAIL；KL 有关联但不足以作为跨 objective 预测变量 |
| G4 | source-calib temperature 是否迁移 | D5 recovery -0.00090；absolute gain -0.00738；base-enriched ECE 也恶化 | P4 FAIL；不能把“temperature scaling 保持 discrimination”写成“可恢复 target calibration” |
| Divergence | χ²/reverse-KL/TV 是否解释 G3 | χ² LODO R² -17.10；reverse-KL 0.326；TV 0.646，对 KL 改善 CI 跨 0 | 替代散度不救回 P6 |
| P7 low-shot | 少量 target labels 是否足够 | per-policy target-T 有显著相对改善，但 budget 50-500 均未达到 ECE CI upper <=0.05 | P7 NEGATIVE；“有帮助”不等于“恢复至 base regime” |
| Independent confirmation | P2/P3 是否跨 lockbox 和 seed 复现 | mean ΔECE -0.00618，CI [-0.01105, 0.00009]；P2-C FAIL；criterion SD 0.02483，p=1e-4；old/new ranking rho=0.952 | `CORE_NOT_ESTABLISHED`；退休 mean-degradation thesis；P3 pattern 仅为 registered secondary evidence |
| P8-C | structured matrix scaling 是否形成闭环 | 因 P2-C 未通过而 `NOT_REACHED` | 不能写 P8 成功或失败；避免在不存在 primary degradation 时测试 remedy |
| Human audit | 原 Qwen reference 的 construct validity | 800-cell packet、双盲顺序与协议已冻结，但人工标注未完成 | 原 PCCD 仍缺直接 human grounding |
| External pivot | 多 guard、人类 benchmark 是否支持 aggregate-certificate story | AEGIS 人工 response positive support 为 0，metadata gate FAIL；改为 BeaverTails single-benchmark label-source design | metadata-first gate 成功阻止无效昂贵实验，但跨 benchmark generalization 被放弃 |
| Three-guard formal run | H1 aggregate、H2 cancellation、H3 rank、H4 label source | H1/H2/H3 不支持；H4 在 ShieldGemma-2B/9B 经 Holm 校正支持；Llama Guard 方向相反 | 最终 `LABELSOURCE_ONLY`；获得窄正结果，但没有救回原适配主线 |

## 4. 原始命题的最终证据账本

### P1：基础分布校准良好

这是全项目最稳定的原始命题。自然 test split mean ECE 为 0.02890；独立 confirmation lockbox D0 mean ECE 为 0.03943，CI [0.03787, 0.04554]，仍在锁定的 0.05 region 内。P1 可作为后续 stress test 的可信锚点。

### P2：适配导致正的平均校准退化

发现性 D5 支持：mean ΔECE 0.02883，CI [0.02235, 0.03306]。独立新-seed D5 不支持：mean ΔECE -0.00618，CI [-0.01105, 0.00009]。因此 discovery effect 不能升级为一般性或确认性结论。论文必须保留 `CORE_NOT_ESTABLISHED`，不能用旧 D5 覆盖新 lockbox。

### P3：变化在安全准则间异质

发现性 D5 RMS(ΔECE) 0.04765、SD 0.03794。独立 lockbox 的 criterion SD 0.02483，CI [0.02094, 0.02897]，omnibus p=1e-4；old/new D5 per-criterion ranking rho=0.952。现象本身强且可重复，但预注册要求 P2-C 先通过，故 P3-C 未被正式确认。最诚实的表述是“强、注册的 secondary interaction”，不是“confirmed adaptation degradation heterogeneity”。

### P4：source-only per-policy temperature scaling 可恢复

不支持。它改善 source NLL、保持 argmax/F1/AUROC 基本非劣，却没有迁移到 shifted target calibration；这说明 discrimination preservation 与 calibration recovery 是不同命题。

### P5：退化由 false negative 主导

结论性不支持，且方向相反。D5 mean(ΔFN-ΔFP) -0.27781，CI [-0.32242, -0.23245]。hidden-violation objective 和 support 修正后仍失败，因此不允许第三次重定义方向。

### P6：KL 是可预测 scaling law

不支持。KL 与 mean ΔECE 有显著正关联，但六点 LODO R² 0.6318 未达 0.70，objective 内部也非单调。六个异质 intervention 点不足以把一个相关性提升为跨方法预测定律。

### P7：50-500 个 target labels 的 temperature family 足以恢复

不支持锁定的绝对恢复标准。相对 raw/global/source-T 的改善是真实 secondary result，但没有任何预算达到 base-regime tolerance。

### P8：structured matrix scaling 可形成修复闭环

最终是 `NOT_REACHED`，不是 FAIL。因为独立 P2-C 没有确认 primary degradation，按照 gate 不应在该 lockbox 上继续寻找 remedy。

## 5. 外部三-guard 命题的最终账本

### H1：看似良好的平均 ECE 掩盖坏类别

不支持。human reference 下三 guard 的 mean-category ECE 已为 0.2121-0.2359，远高于 eps=0.015；worst-category ECE 为 0.4562-0.4821。坏类别存在，但平均值并没有伪装成良好证书。

### H2：类别间正负 calibration error 发生抵消

不支持。所有可评估 human-reference signed deviation 同向为负，cancellation C=0。静态 guard audit 显示的是一致的 unsafe-probability underprediction，而不是 sign cancellation。

### H3：mean-optimal 与 worst-optimal guard 排名翻转

未建立。共同七个 PRIMARY 类别上，Llama Guard 同时最小化 mean 和 worst ECE；distinct-winner bootstrap probability 为 0。只有三个 guard、其中两个同属 ShieldGemma family，按锁定规则只能写 `INCONCLUSIVE-FOR-RANK`。

### H4：分类别 audit 对标签来源敏感

支持，但边界窄。ShieldGemma-2B 的 subjective-minus-objective absolute label-source effect 为 0.01087，CI [0.00337, 0.01889]，Holm p=0.02990；ShieldGemma-9B 为 0.01547，CI [0.00807, 0.02308]，Holm p=0.01120。Llama Guard 呈相反轴方向，不支持 H4。结论只能是“至少一个 guard family 的 audit 对 human-vs-Qwen-proxy reference source 系统敏感”，不能说 Qwen 等价于人类、不能说人类标签错误，也不能泛化到所有 guard。

## 6. 做得正确的 research routine

### 6.1 冻结与诚实报告真正改变了结论质量

项目没有为了通过 gate 修改数据、阈值或方向。G2 的反向 FN、G3 的 0.632、G4/P7 的不充分恢复、independent P2-C 的负点估计、external H1-H3 的失败都被完整保留。尤其独立 lockbox 阻止了把 discovery D5 的显著正效应写成一般规律，这是全项目最重要的方法学成功。

### 6.2 元数据门和 provenance audit 节省了错误实验

AEGIS audit 在任何 guard scoring 前发现：5,236 个 human-source response 全部为 safe，human unsafe response 为 0，无法满足 response-level criterion support。停止 AEGIS 不是拖延，而是正确执行构念门。类似地，H5 空映射、PKU metadata、ShieldGemma no-map cells、Llama semantic label token 等问题都在适当层面被保留和修正。

### 6.3 将诊断与 gate 分开是正确的

D3 diagnostic 没有冒充确认性结果；G1 diag100 先定位 prompt-structure 与 paraphrase 问题；χ² 分析被明确标为 non-gating；P7/P8 被定义为新窄命题，不能重写 P4。这个分层显著降低了事后解释污染。

### 6.4 工程可复现性很强

模型 revision、数据 hash、checkpoint、logits、teacher outputs、run marker、one-run consumption、bootstrap seed、完整 cell 表和 CHANGES 均被记录。运行中发生的 import、token-position、post-write NameError 等软件问题都按 outcome-blind 原则处理，没有覆盖已完成结果。

## 7. 需要反思的科学设计

### 7.1 初始 thesis 过载

九天计划同时要求证明 effect existence、heterogeneity、harm direction、scaling law 和 remedy。它更像一篇已经知道答案后的完整论文结构，而不是一个从高不确定性起步的研究程序。更合理的顺序应当是先确认一个主效应和一个 construct-validity 条件，再投入机制、预测和修复。

### 7.2 construct validity 被放到了过晚阶段

Qwen teacher 在大规模标注前没有先完成小规模人类双标审计。后来虽发现 fixed-prompt repeat 很高，但 order/paraphrase 敏感明显，H5 还经历了 taxonomy 重定义。由于 critic training target 和 evaluation reference 都来自同一个 fixed Qwen protocol，P1 的良好校准部分是“对该 oracle 校准良好”，并不自动等于对人类安全判断校准良好。

### 7.3 G1 把错误层次的异质性设成 gate

原 L3 要求 base critic 的 per-policy F1 CV>0.15，但理论真正需要的是 adaptation-induced delta 在 policy 间不同。一个优秀 base critic 本应各头都好，低 CV 与 P1 并不矛盾。这个 gate 设计在数据前就把“政策空间非退化”和“退化异质”混为一谈。

### 7.4 intervention 没有先做 manipulation check

D3 safe-SFT 测的是变安全、显式的输出，却被用于探索 hidden violations 导致的漏报；同时 base violated support 只有 3-10，造成 FN support-shift confound。若先用 50-100 个 prompts 验证 adapted violation prevalence、hiddenness、人类可判性和 critic miss rate，便可在正式 grid 前发现方向不匹配。

### 7.5 不同 evaluation estimand 混入叙事

自然 base test 的 mean ECE 2.9% 与 support-enriched G2 D0 的 4.95%不是同一分布；external guard 的 balanced binary category ECE 21%-24%又是另一 estimand。报告后来正确区分了这些数字，但早期叙事容易把“指标同名”误当成“可直接比较”。下一次必须在项目开头画出 estimand matrix：label space、prevalence weighting、support restriction、binning、reference source、population。

### 7.6 六个异质 D 点不足以支撑 scaling law

D1-D6 不只是强度不同，也混合了 system prompt、safe-SFT、hidden-SFT、rank 和 DPO objective。KL 被当作统一 x 轴，但 objective identity 仍强烈影响 y。六点做 LODO 对单个异常点极敏感，hidden-family 又只有三点。它适合机制探索，不适合强 scaling-law claim。

### 7.7 remedy research 启动过早

P4、P7、P8 在 primary effect 尚未独立确认前逐层扩展。虽然每次都有新 preregistration，科学纪律尚在，但资源配置呈现“一个修复失败再增加更强修复”的 rescue ladder。确认主效应失败后，P8 gate 正确停止；这个停止应更早成为项目设计原则。

### 7.8 external pivot 的假设仍受旧现象牵引

旧 lockbox 看到的是 ΔECE 在 criterion 间正负变化；新 H2 却检验静态 guard 的 signed probability deviation 是否正负抵消，两者不是同一统计对象。H1 的 eps=0.015 对 balanced binary per-category guard ECE 也非常苛刻，H3 只有三个 guard 且两个同 family，预注册已承认低 power。说明新设计虽然 outcome-blind，但仍过度从旧故事类比阈值和结构，没有先做纯 metadata/estimator-scale 的 pilot calibration。

### 7.9 多次 reframe 增加叙事风险

项目经历 FN-asymmetry -> heterogeneous degradation -> KL/remedy -> aggregate certificate -> label-source sensitivity。每次都有决策日志和冻结边界，因此没有构成静默 HARKing；但对审稿人而言，若把所有阶段包装成一个从始至终的单一假设，会显得目标漂移。论文必须把它写成 sequential registered research program：哪些是 discovery、哪些失败、哪些是新 preregistered study，而非事后连成必然路径。

## 8. 工程和项目管理反思

### 成功部分

- 大文件统一放在 `/root/autodl-tmp`，解决系统盘问题。
- 32B teacher 双卡 label-only 吞吐有约 76 倍余量。
- TRL/PEFT/Transformers-5 smoke 在正式训练前完成。
- 使用显式一进程一卡，避免隐式 DataParallel。
- 模型下载、HF revision、许可证、verbalizer 和 prompt registry 被冻结。
- Git feature branch、PR、CHANGES、day reports 和 artifact SHA 形成了可靠审计链。

### 可改进部分

- 模型 access、地域网络和 gated-repo 权限应在研究立项首日完成 registry preflight，而不是到 external stage 才成为主要耗时。
- 九天时间线低估了 taxonomy 审查、构念验证、独立确认和模型下载；它适合作为工程 sprint，不适合作为完整科学证据周期。
- 多次 relabel、长分支和跨阶段 pipeline 累积增加软件风险。每个科学阶段应在 merge 后建立 immutable artifact release，而不是继续在超长分支叠加。
- 应在首次正式 run 前增加 post-write control-flow 的端到端测试；Llama Guard `NameError` 虽不影响结果，但暴露了仅测核心 forward、不测完成标记路径的问题。

## 9. 如果从头重做，更合理的 routine

### Phase 0：一句主命题与证伪条件

只保留一个 primary claim，例如：“在固定 human reference 下，policy adaptation 会导致 criterion-specific calibration vector 发生可重复变化。”不要预先绑定平均方向、FN 方向、KL law 和 remedy。预先写出什么结果会令该命题失败。

### Phase 1：measurement validity first

1. 先做 200-500 个样本、两位人工 annotator 的 blinded criterion audit。
2. 比较 human-human、human-Qwen、Qwen repeat/order/paraphrase。
3. 只保留有足够 positive/negative/applicable support 且定义一致的 criteria。
4. 冻结同一个 ECE estimand 和 population weighting，贯穿 discovery 与 confirmation。

### Phase 2：最小 manipulation pilot

只选 base 与一个 adaptation objective、两个强度。先验证：KL 确实分级；violation support 充分；输出确实更 hidden 而非更显式；reference labels 可重复；critic 未被更新。若 manipulation 不成立，不进入大 grid。

### Phase 3：discovery 与 confirmation 同时设计

在看到 discovery outcome 前，就冻结独立 dataset family、training seed、adapter objective 和 one-unseal confirmation。Discovery 用于估计 effect shape；confirmation 只检验一个 primary effect 和一个 interaction，不允许 discovery 后再决定 lockbox 的主检验。

### Phase 4：只有 core effect 确认后才做 mechanism/remedy

- 若 criterion-vector shift 确认，再测试 KL、tail divergence、representation drift 等机制。
- 若 calibration degradation 确认，再测试 source-only、target-aware、matrix scaling 等 remedy。
- 如果 core 不确认，停止 remedy ladder，直接写 boundary/negative result。

### Phase 5：外部效度采用正交复制

至少两个 human-labelled benchmarks、至少三个真正不同的 guard families、至少两个独立 proxy label sources。Guard ranking 假设应先做 power analysis；label-source study应把 family 作为层级而不是把同 family 两个规模当成两个完全独立复制。

## 10. 当前最诚实的论文证据层级

### 可以作为确认性正结果

1. 冻结 critic 在两个基础评估分布上保持良好平均 calibration anchor。
2. BeaverTails 上，ShieldGemma-2B/9B 的 per-category calibration audit 对 human-vs-Qwen proxy label source 呈预注册的 objective/subjective axis sensitivity。

### 可以作为强 secondary evidence

1. 原 PCCD 独立 lockbox 中 criterion-specific ΔECE 的显著 dispersion。
2. 旧/新 D5 seed 上 per-criterion ordering 的高稳定性 rho=0.952。
3. target-aware per-policy temperature scaling 有相对改善，但不足以恢复至 base tolerance。

### 应作为负结果或边界条件

1. 正的平均 adaptation degradation 未独立确认。
2. FN-asymmetric degradation 显著不支持。
3. KL 不是跨 objective 的充分预测变量。
4. source-only temperature scaling 不迁移。
5. χ²/reverse-KL 未优于 KL；TV 改善未识别。
6. external H1 aggregate-hiding、H2 cancellation、H3 rank reversal 未支持。

### 不得声称

1. Qwen teacher 是人工 ground truth。
2. adaptation 普遍导致平均 calibration 恶化。
3. criterion heterogeneity 已获得无条件 P3-C confirmation。
4. hidden violations 导致 FN-dominant failure。
5. P8 structured matrix scaling 已失败或成功。
6. label-source effect 已跨 benchmark、跨所有 guard family 泛化。

## 11. 最终反思

本项目最有价值的部分不是它最终保住了多少最初命题，而是它展示了一个高风险 AI-safety 实证项目如何在连续负结果中保持证据边界。独立确认推翻 discovery 主效应、metadata audit 阻止不合法 benchmark、one-run preregistration 保留不利 cell，这些程序性选择比任何单个正 p-value 更可靠。

但程序严谨不能替代问题聚焦。早期 routine 的根本失误是，在 measurement validity、intervention validity 和 independent replication 之前，就把论文设计成“现象 + 危害机制 + scaling law + remedy”的完整闭环。后期每次转向虽然诚实，却不断缩窄论文中心。下一轮研究应把人类构念验证、统一 estimand、最小 manipulation check 和同步设计的 confirmation 放在第一周；把机制和修复放在 core effect 被确认之后。

当前最可辩护的研究故事不是“我们证明了冻结安全 critic 在适配后按预想方式失效”，而是：“一系列预注册 stress tests 显示，安全 calibration 的结论高度依赖准则、评估分布和 reference-label source；简单平均、单一 shift scalar 和 source-only recalibration 都不足以提供稳健部署保证。其最明确的外部正证据，是两个 ShieldGemma 规模上的分类别 label-source sensitivity。”这一故事严谨、诚实，但仍需要跨 benchmark、跨 guard family 的独立复制，才能成为强主会级结论。
