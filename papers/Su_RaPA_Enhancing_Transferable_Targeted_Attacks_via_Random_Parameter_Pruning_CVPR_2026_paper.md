<!-- page 1 -->
RaPA: Enhancing Transferable Targeted Attacks via
Random Parameter Pruning
Tongrui Su
Qingbin Li
Shengyu Zhu*
Wei Chen*
Xueqi Cheng
State Key Lab of AI Safety, Institute of Computing Technology, Chinese Academy of Sciences
University of Chinese Academy of Sciences
{sutongrui25s, liqingbin24z, zhushengyu, chenwei2022, cxq}@ict.ac.cn
Abstract
Compared to untargeted attacks, targeted transfer-
based attack still suffers from much lower Attack Success
Rates (ASRs), although significant improvements have been
achieved by kinds of methods, such as diversifying input,
stabilizing the gradient, and re-training surrogate models.
In this paper, we find that adversarial examples generated
by existing methods rely heavily on a small subset of sur-
rogate model parameters, which limits their transferabil-
ity to unseen target models. Inspired by this finding, we
propose Random Parameter Pruning Attack (RaPA), which
introduces parameter-level randomization during the at-
tack process. At each optimization step, RaPA randomly
prunes model parameters to generate diverse yet seman-
tically consistent surrogate variants.
We show that this
parameter-level randomization is equivalent to adding an
importance-equalization regularizer, thereby alleviating the
over-reliance issue.
Extensive experiments across both
CNN and Transformer architectures demonstrate that RaPA
substantially enhances transferability. In the challenging
case of transferring from CNN-based to Transformer-based
models, RaPA achieves up to 11.7% higher average ASRs
than state-of-the-art baselines (with 33.3% ASRs), while be-
ing training-free, cross-architecture efficient, and easily in-
tegrated into existing attack frameworks. Code is available
on https://github.com/molarsu/RaPA.
1. Introduction
Deep neural networks have become prevalent in computer
vision applications [9, 17, 19], but are highly vulnerable
to maliciously crafted inputs, called adversarial examples
[12, 42]. A major concern is their transferability, that is, ad-
versarial examples generated using a white-box model can
directly fool other black-box models, without any access to
their architectures, parameters or gradients [14]. Since this
*Corresponding author.
type of attacks, usually referred to as transfer-based attacks,
do not require any interaction with the target model, they
pose severe security risks to real-world machine learning
systems. Therefore, studying effective transfer-based attack
methods is crucial to understand the vulnerabilities and fur-
ther enhance model robustness.
This paper focuses on targeted transfer-based attacks
with a single surrogate model, where the goal is to deceive
black-box models to classify input images into a specific
incorrect category. Due to the high complexity of decision
boundaries, existing methods still have noticeably lower At-
tack Success Rates (ASRs) in the targeted setting than in the
untargeted [2, 60]. A key observation is that the generated
adversarial examples tend to overfit the surrogate model but
fail to generalize to other models. To improve transferabil-
ity, various strategies have been proposed. Observing that
multiple surrogate models can help enhance transferability
but in practice finding proper models for the same task is not
easy [7, 30, 31, 59], model self-ensemble [20, 26, 34, 58]
tries to create multiple models from an accessible model.
Input transformation [1, 2, 27, 49, 51, 52, 56, 61] applies
different transformations to inputs and diversify input pat-
terns to reduce overfitting. A notable method is Clean Fea-
ture Mixup (CFM) [2], which randomly mixes high-level
features with shuffled clean features. Building upon it, Fea-
ture Tuning Mixup (FTM) [27] introduces learnable and
attack-specific feature perturbations, achieving new state-
of-the-art performance in transferability. Despite these pro-
gresses, there is still much room for further improvement.
In this work, we take a different perspective and identify
a previously overlooked cause of the poor transferability:
the generated adversarial perturbations rely excessively on
a small subset of parameters in the surrogate model, which
limits their generalization to other models that have dif-
ferent parameter configurations. In other words, adversar-
ial perturbation in existing methods tends to exploit a few
“shortcut” parameters, leading to strong white-box perfor-
mance but poor black-box transferability.
To mitigate this issue, we propose Random Parameter
This CVPR paper is the Open Access version, provided by the Computer Vision Foundation.
Except for this watermark, it is identical to the accepted version;
the final published version of the proceedings is available on IEEE Xplore.
6538

---

<!-- page 2 -->
Figure 1. An illustration of the proposed method RaPA. We apply
RaPA to selected layers in the surrogate model to create multiple
and diverse variants at each iteration.
Pruning Attack (RaPA) that introduces parameter-level ran-
domization into the attack process. At each optimization
step, RaPA randomly prunes a subset of parameters in the
surrogate model and uses multiple masked variants to up-
date the adversarial example. We show that taking the ex-
pectation over such random masks is equivalent to adding
an importance regularization term that aims to equalize
parameter contributions, thus preventing over-reliance on
a few dominant parameters. Conceptually, RaPA can be
viewed as a self-ensemble method: each randomly pruned
model represents a diverse yet semantically consistent vari-
ant of the surrogate.
Previous self-ensemble approaches
SASD-WS [54], MUP [58], and Ghost Network [26] rely
on training-based model enhancement, deterministic prun-
ing metrics, and structural perturbations, respectively. In
contrast, RaPA is training-free, cross-architecture efficient,
and straightforward to implement.
We evaluate the proposed method across various CNN-
and Transformer-based target models, as well as against
several defense methods. The experimental results show
that RaPA outperforms other state-of-the-art methods. In
particular, in the challenging scenario of transferring from
CNN-based model to Transformer-based models, RaPA
achieves 11.7% and 17.5% higher average ASRs with
ResNet-50 [17] and DenseNet-121 [19] as surrogate mod-
els, respectively. Moreover, RaPA achieves the highest per-
formance gain when scaling the compute for crafting adver-
sarial examples. Specifically, with ResNet-50 as surrogate
model, increasing the optimization iterations from 300 to
500 and number of forward-backward passes per iteration
from 1 to 5 boosts the average ASR by 15.9%.
To summarize, our main contributions are as follows:
• We show that adversarial examples from existing transfer-
based attacks rely heavily on a tiny subset of parameters
in the surrogate model. Alleviating this over-reliance can
in turn enhance the transferability of attack.
• We propose the RaPA, which introduces parameter-level
randomization during attack optimization. We show, both
intuitively and empirically, that random pruning implic-
itly equalizes parameter importance, acting as a regular-
izer to mitigate the over-reliance issue.
• Experiments across diverse surrogate and target models
demonstrate that RaPA consistently outperforms existing
methods. RaPA further benefits from increased computa-
tional budget, achieving larger improvements when scal-
ing optimization iterations or inference steps.
2. Preliminary
This section introduces the background of adversarial at-
tacks and briefly reviews related works on targeted transfer-
based attacks. See Appendix A for more related work.
2.1. Background
Consider a classification task where the model is defined as
a function f : Rn →Y that maps an input x ∈Rn to a
label in the set Y consisting of all the labels. Given a clean
image x with its true label y ∈Y, untargeted attacks aim to
find an adversarial example xadv ∈Rn that is similar to x
but misleads the model to produce an incorrect prediction,
i.e. f(xadv) ̸= y. Here ‘similarity’ is usually measured by
an ℓp-norm, e.g., ∥xadv −x∥p ≤ϵ where ϵ > 0 is a pre-
defined perturbation budget. For targeted attacks, the goal
is to modify the model prediction to a particular target label
ytar, that is, f(xadv) = ytar ̸= y. In this work, we will
focus on targeted attacks.
In the white-box setting where the model is fully acces-
sible, adversarial example can be obtained by the following:
arg max
xadv
L (f(xadv)) , s.t. ∥xadv −x∥p ≤ϵ,
(1)
where L(·) is a loss function (e.g., cross-entropy loss). The
Fast Gradient Sign Method (FGSM) [12] uses the gradient
direction to solve this problem and craft adversarial exam-
ples, while Iterative FGSM (I-FGSM) [23] extends this idea
to an iterative scheme. In particular, at each iteration t, ad-
versarial example is updated by adding a small perturbation:
xt
adv = xt−1
adv + α · sign

∇xt−1
adv L
 f(xt−1
adv)

,
(2)
where x0
adv = x and α > 0 is a step size. To make the gen-
erated adversarial example satisfy the perturbation budget
constraint, a straightforward way is to project xt
adv into the
ϵ-ball of x.
2.2. Related Work
In black-box settings, the gradient information is not avail-
able and we only have limited access to the target model.
Transfer-based method assumes that the adversarial exam-
ples generated on one model to mislead not only that model
but also other models [42].
6539

---

<!-- page 3 -->
Method
DI
RDI
SI
Admix
ODI
BSR
CFM
RaPA
No pruning
98.2
98.7
98.7
98.1
98.9
98.5
98.0
98.2
Pruning bottom 0.5%
98.2
98.7
98.7
98.0
98.9
98.5
98.0
98.0
Pruning top 0.5%
16.0
28.6
31.9
29.9
19.9
36.7
51.3
64.5
Table 1. ASRs (%) before and after pruning the selected subsets of parameters in the surrogate model on the ImageNet-compatible dataset.
Detailed experimental setting can be found in Section 4.1.
To improve transferability, many methods have been pro-
posed. The first class, input-transformation techniques, ap-
plies a transformation T and uses ∇xt−1
adv L
 f(T (xt−1
adv))

as the gradient. Diverse Inputs (DI) [56] and its variant
Resized DI (RDI) [61] apply random transformations to
increase input variation during optimization. Translation-
Invariant (TI) [8] averages gradients over translated inputs
to reduce location sensitivity.
Structure Invariant Attack
(SIA) [52] and Block Shuffle and Rotate (BSR) [49] both
perform block-level local transformations, with SIA apply-
ing diverse transforms and BSR focusing shuffle and rotate
operations. Object-based DI (ODI) [1] generates adversar-
ial examples rendered on 3D objects, while Admix [51]
mixes inputs with random samples from other classes. CFM
[2] extends it to the feature space with competing noises,
while FTM [27] further adds learnable, attack-specific per-
turbations to achieve state-of-the-art transferability.
The second class focuses on stabilizing gradient updates
to improve transferability. Momentum Iterative FGSM (MI-
FGSM) [7] incorporates a momentum term into I-FGSM
to help avoid local optima. Scale-Invariant (SI) optimiza-
tion [29] improves transferability by applying perturbations
across multiple scaled copies of the input, leveraging the
scale-invariance property of deep models.
Beyond the above methods, another type of approaches
re-train the surrogate model to enhance transferability, e.g.,
DSM [57] and SASD-WS [54] improve model generaliza-
tion and transferability through knowledge distillation or
sharpness-aware self-distillation.
Closely related to the present work is self-ensemble [20,
26, 34, 58], which creates multiple models from only one
surrogate model. The self-ensemble method in [34] specif-
ically considers vision Transformer as surrogate model and
is denoted as SE-ViT in this paper. Ghost Network [26] per-
turbs surrogate model to create a set of new models and then
samples one model from the set at each iteration. Masking
Unimportant Parameters (MUP) [58] drops out unimpor-
tant parameters according to a predefined Taylor expansion-
based metric, while Diversity Weight Pruning(DWP) [48]
only prunes the parameters with small absolute values.
However, the inherent differences w.r.t. model architec-
ture, parameter setting, and training procedure between the
surrogate and target models still limit the effectiveness of
transfer-based methods on certain models and datasets. In
the next section, we show that there is a key aspect that ren-
ders the current over-fitting issue of transfer-based methods.
3. Method
In this section, we first conduct a pilot study to show a
key aspect of the overfitting issue in existing transfer-based
methods and then propose a random masking based ap-
proach. Comparison with related methods is also discussed.
3.1. Motivation
We observe that the adversarial perturbations generated by
solving Problem (1) tend to rely heavily on a small subset of
parameters in the surrogate model. These parameters may
stem from specific training schemes, datasets, or architec-
tural choices. As a result, adversarial examples that strongly
depend on these parameters often fail to generalize and mis-
lead other models. Even with state-of-the-art transfer-based
attack methods, this issue remains a key factor that can lead
to the failure of adversarial example transfer.
To quantify the phenomenon, we conduct a pilot study
using the framework of Optimal Brain Damage (OBD)
[24, 33], which quantifies the importance of each model pa-
rameter from the perspective of sensitivity analysis. Specifi-
cally, given an adversarial example xadv and a loss function
L(·). Let θ represent the entire set of model parameters.
The importance of a parameter θi in the surrogate model f
is computed as:
I(θi) = ∂2L (f(xadv))
∂θ2
i
× θ2
i .
(3)
This metric reflects how much the loss would change if a
parameter θi were removed, and can serve as a proxy for its
contribution to the effectiveness of the adversarial example.
Next, we consider pruning two distinct subsets of the sur-
rogate model’s parameters based on this importance metric:
the top 0.5% most important and the bottom 0.5% least im-
portant parameters(see Appendix B.1 for details). For each
adversarial example, we instantiate the model, prune the se-
lected subset, and evaluate the ASR on the resulting model.
Table 1 reports the ASRs after pruning the two subsets of
model parameters. Here we use ResNet50 as the surrogate
model and detailed setting can be found in Section 4.1.
We observe that pruning the most important parameters
leads to a drastic drop in ASR—more than 46%, whereas
pruning the least important parameters yields negligible im-
pact. This observation suggests that adversarial examples
6540

---

<!-- page 4 -->
generated by existing methods are highly dependent on the
most important parameters, validating our observation on
the over-reliance issue. As such, how to further alleviate
this strong dependence on specific parameters would be a
key to improving the transferability of adversarial examples
over existing transfer-based attack methods.
3.2. Alleviating Over-reliance via Random Param-
eter Pruning
As per the pilot study, a direct approach to improving trans-
ferability would be to mask the most important parame-
ters at each optimization step, thereby mitigating the over-
dependency on them. However, accurately identifying im-
portant parameters requires computing second-order deriva-
tives, which is computationally expensive for all parame-
ters. Although we can approximate them with first-order
terms, masking the most important parameters typically
causes the surrogate model’s capacity to degrade rapidly,
and the resulting adversarial examples may fail to fool the
target model—and even the original surrogate model itself
(See Appendix D.1 for further explanations). To address
this problem, we propose to apply random parameter prun-
ing to the surrogate model at each optimize step. This ap-
proach avoids expensive computations while achieving the
goal of reducing over-reliance on specific parameters.
Intuition and Theoretical Explanation
Our core idea is
that randomly pruning parameters at different optimization
steps encourages the generated adversarial examples to be
less dependent on particular parameter subsets. This in turn
improves transferability across different target models.
We define a random binary mask M ∈{0, 1}|θ|, where
each entry is independently sampled from a Bernoulli dis-
tribution: Mi ∼Bernoulli(1 −p). Here p ∈[0, 1] is the
probability of masking a parameter. With a small p, we
would have E[Mi] ≈1. Then the parameter of the model
used in the forward pass becomes M ⊙θ, where ⊙denotes
element-wise multiplication.
Under this setup, the expected loss over random masks
can be approximated using a second-order Taylor expan-
sion:
EM[L(f(xadv; M ⊙θ))]
≈L(f(xadv; θ)) + p(1 −p)
2
X
i
∂2Lf(xadv; θ)
∂θ2
i
θ2
i ,
(4)
which is sum of the original loss plus an importance penalty.
Minimizing this objective while resampling the mask at
each step would force the adversarial example to distribute
the importance over all parameters, making it more robust
to different parameters and thus more transferable.
Practical Implementation with DropConnect
The
above random parameter pruning method is similar to Drop-
Connect method [47] in training neural networks. We no-
tice that DropConnect is mainly effective in terms of lin-
ear layers . We thus apply DropConnect to the weight and
bias parameters of linear layers as well as the transforma-
tion parameters of normalization layers. Both types of lay-
ers are widely used in mainstream architectures including
Transformer [46]. Empirically, our ablation study in Sec-
tion 4 validates the effectiveness of this choice, compared
with convolutional layers.
We now present our attack method, Random Parame-
ter Pruning Attack (RaPA), as summarized in Algorithm 1.
Take linear layer for example. We perform independent ran-
dom masking onto the weight and bias (if present) param-
eters using Bernoulli sampling. Specifically, for surrogate
model f, let W ∈Rdin×dout denote the weight matrix and
b ∈Rdout the bias vector associated with a linear layer. Here
din and dout represent the input and output dimensions, re-
spectively. The corresponding masks for the weight matrix
and bias are
Mw ∼Bernoulli(1 −pw), Mb ∼Bernoulli(1 −pb), (5)
where Mw ∈{0, 1}din×dout and Mb ∈{0, 1}dout are the
random masks, and pw, pb ∈[0, 1] are DropConnect proba-
bilities. Then the masked parameters are computed as
WM = Mw ⊙W, bM = Mb ⊙b,
(6)
where ⊙denotes the element-wise multiplication. For nor-
malization layer, the same operation is applied similarly to
the transformation parameters. The random masks are sam-
pled for each selected layer.
The random masks Mw and Mb in Eq. (5) are re-
generated for each inference, producing diverse variants of
the surrogate model with different parameters if we conduct
multiple inferences at an iteration. In addition, RaPA can be
naturally integrated with existing input transformation and
gradient stabilization methods for crafting adversarial ex-
amples, as shown in Lines 5 and 8 in Algorithm 1.
Analyzing Parameter Importance with Gini Coefficient
To further verify that our random parameter pruning strat-
egy indeed mitigates the over-reliance on a few dominant
parameters, we employ the Gini coefficient to measure the
distribution of parameter importance across layers. A lower
Gini value indicates a more uniform distribution of impor-
tance, implying that the adversarial perturbation depends
less on specific parameters and generalizes better to un-
seen models. The formal definition and detailed compu-
tation process of the Gini coefficient are provided in Ap-
pendix B.3.
We compute the Gini coefficients based on the parame-
ter importance values I(θi) defined in Eq. (3). The over-
all and layer-wise results are summarized in Table 2. As
6541

---

<!-- page 5 -->
Method
DI
RDI
SI
Admix
ODI
BSR
CFM
RaPA
All Layer Average
0.32
0.30
0.21
0.12
0.33
0.25
0.19
0.08
Conv Layer
0.11
0.07
0.03
0.01
0.12
0.05
0.03
0.00
Norm Layer
0.51
0.52
0.37
0.22
0.53
0.44
0.34
0.15
Linear Layer
1.00
0.86
0.59
0.18
1.00
0.75
0.55
0.13
Table 2. Gini coefficients of parameter importance across different layers and methods. Lower values correspond to more uniform param-
eter importance.
Algorithm 1 Random Parameter Pruning Attack(RaPA)
Input: Classifier f; clean image x; loss function L(·); max
iterations T; ℓp bound ϵ; number of inferences per iter-
ation S; DropConnect probabilities pw, pb; linear and
normalization layers L; input transformation T .
Output: Adversarial example xadv
1: x0
adv ←x
2: for t = 1 →T do
3:
for s = 1 →S do
4:
Obtain modified model fM by applying RaPA
to each layer in L according to Eqs. (5) and (6).
5:
gt
s ←∇xt−1
adv L
 fM(T (xt−1
adv))

6:
end for
7:
gt ←1
S
P gt
s
8:
Update xt
adv with gradient gt using iterative meth-
ods (like MI-FGSM [7]).
9:
Project xt
adv into the ϵ-ball of x.
10: end for
11: return xT
adv
shown in Table 2, RaPA achieves the lowest Gini coeffi-
cients among all compared methods, suggesting that it ef-
fectively flattens the importance distribution and suppresses
the over-concentration of sensitivity on a few parameters.
This balanced importance allocation leads to improved ro-
bustness and transferability across different architectures.
3.3. Discussion
In this section, we compare RaPA with related self-
ensemble methods in more details.
RaPA was proposed to reduce over-dependence on spe-
cific parameters, and it turns out to be a self-ensemble
method that constructs multiple new models at each itera-
tion. Existing self-ensemble method [20] targets object de-
tection task, which is different from ours. SE-ViT [34] is
specifically designed for vision Transformer surrogate mod-
els; as shown in Section 4, its ASR is lower than RaPA even
when using ViT as surrogate model. More closely related
are Ghost Network [26] and MUP [58], but are much out-
performed by RaPA (c.f. Appendix D.1 and Section 4.2).
We now analyze the effectiveness of RaPA from model
ensemble perspective. It has been hypothesized that an ad-
versarial image that remains adversarial for multiple models
is more likely to transfer to other models [31]. RaPA gen-
erates independent random masks for each selected layer
and also at each optimization iteration.
In this sense, it
brings in more randomness and further diversification than
Ghost Network, MUP and DWP. Specifically, Ghost Net-
work perturbs only skip connections for residual networks
(like ResNet-50), while MUP and DWP mask unimportant
parameters according to a predefined metricor at each itera-
tion. This observation is also in accordance with [3, 20, 30],
which show that increasing the number of surrogate models
generally enhances the transfer attack performance. On the
other hand, from the perspective of ensemble techniques in
machine learning, each variant model should also be infor-
mative about or useful to the targeted task of image clas-
sification [3, 20]. As empirically shown in Appendix D.5,
RaPA achieves a good tradeoff in terms of model diversity
and utility, thereby enhancing the attack performance.
4. Experiments
This section empirically validates the effectiveness of our
method, using both CNN- and Transformer-based models.
4.1. Experimental Settings
Dataset
We utilize the ImageNet-compatible dataset [22],
served as the official dataset for the NIPS 2017 Attack Chal-
lenge. This dataset contains both ground-truth and targeted
labels, making it well-suited for targeted-attack.
General Setting
We adopt the ℓ∞-norm as the constraint
on perturbation, with budget ϵ = 16/255. The learning rate
is chosen as α = 2/255. We use the Logit loss [60] as our
objective function in Equation (1). By default, we set the
maximum number of optimization iterations to 1, 000 and
the batch size to 32 for all baseline methods, to ensure suf-
ficient optimization. Furthermore, we notice that different
attack methods may take different amounts of computations
per iteration. We hence fix the same number of inferences
for each optimization iteration to maintain fair comparisons
across all attack methods.
Surrogate and Target Models
Our experiments choose
various models commonly used in the literature [2]. These
include 1) CNN-based models:
VGG-16 [41], ResNet-
18 (RN18) [17], ResNet-50 (RN50) [17], DenseNet-121
(DN121) [19], Xception (Xcep) [4], MobileNet-v2 (MBv2)
6542

---

<!-- page 6 -->
Source: RN50
Source: DN121
Attack
ViT
LeViT
ConViT
Twins
PiT
CLIP
Avg.
ViT
LeViT
ConViT
Twins
PiT
CLIP
Avg.
DI
0.4
6.7
0.6
3.8
1.8
0.5
2.3
0.3
4.0
1.2
2.0
2.1
0.3
1.7
RDI
2.8
24.0
4.4
12.6
10.1
1.2
9.2
1.0
12.0
2.1
6.9
8.2
1.4
5.3
SI
8.0
42.9
7.7
25.1
23.3
3.7
18.4
3.9
21.5
4.4
10.2
14.3
2.1
9.4
SIA
3.1
28.4
3.9
16.7
13.5
1.9
11.2
2.4
24.2
2.2
12.9
11.1
1.5
9.0
BSR
6.8
42.4
7.7
25.3
21.9
2.3
17.7
2.7
21.6
2.7
10.6
11.9
1.2
8.5
DWP
3.7
32.0
4.0
17.6
13.3
1.8
12.1
2.7
21.1
2.9
14.3
12.0
2.4
9.2
Admix
7.2
43.4
5.9
22.7
19.4
4.2
17.1
4.2
33.8
4.0
18.0
19.1
3.9
13.8
ODI
15.5
49.3
11.8
31.6
35.1
5.6
24.8
8.5
36.3
8.4
19.8
26.7
4.9
17.4
MUP
7.0
48.3
8.2
30.7
26.5
3.6
20.7
5.2
38.9
5.0
23.2
20.1
4.1
16.1
CFM
17.3
65.8
14.6
47.5
39.9
7.9
32.2
10.4
49.0
8.3
30.1
30.4
5.5
22.3
FTM
18.0
67.6
16.5
47.1
41.5
9.3
33.3
10.7
50.4
9.4
31.1
30.4
4.8
22.8
RaPA
33.8±1.0
75.4±1.8
27.6±0.2
59.5±0.4
57.3±0.7
15.6±0.1
45.0
27.8±0.1
69.4±0.6
23.5±0.2
53.1±0.6
54.0±0.1
14.1±0.0
40.3
Table 3. ASRs (%) against five Transformer-based target models on the ImageNet-Compatible dataset. All the attack methods are combined
with MI-TI. The best results are shown in bold and the second best results are underlined.
[40], EfficientNet-B0 (EFB0) [45], Inception ResNetv2
(IRv2) [44], Inception-v3 (Incv3) [43], and Inception-v4
(Incv4) [44]; and 2) Transformer-based models: ViT [9],
LeViT [13], ConViT [10], Twins [5], and Pooling-based Vi-
sion Transformer (PiT) [18]. All the models are pre-trained
on the ImageNet dataset [6] . Additionally, we include CLIP
[37], trained on 400 million text-image pairs, to evaluate
the transferability across different modalities.
When us-
ing Transformer-based models, the input image is resized
to 224×224 to meet the model input requirement.
Baseline Attack Methods
We compare RaPA with vari-
ous existing transfer-based methods, including DI [56], RDI
[61], SI [29], Admix [51],SIA [52], BSR [49], ODI [1],
and CFM [2]. We also include two existing self-ensemble
methods, namely, MUP (whose implementation only han-
dles CNN layers) [58] and SE-ViT (which is specifically
designed for vision Transformers) [34].
These methods
are primarily used in combination with TI-FGSM [8] and
MI-FGSM [7] during the optimization process. It is worth
noting that some of these methods, namely, SI, BSR, Ad-
mix, CFM, MUP and SE-ViT, are implemented together
with RDI, which have been reported to obtain higher trans-
fer ASRs [2]. As previously mentioned, the attack meth-
ods are configured with an identical number of inferences
per iteration, denoted as S. Specifically, we pick S scaled
copies in SI and in the inner loop of Admix, and S trans-
formed images for BSR . For other baselines, we perform
S forward-backward passes and use the average gradient on
xadv to update the adversarial example in each iteration. In
the main experiment, we will set S = 5; comparison of
other choices is studied in Section 4.3.
RaPA Setting
While the DropConnect probabilities can
be chosen differently for the weight and bias parameters
in a linear layer (or the transformation parameters in the
normalization layer) and also across different layers, we
choose the same probability for all selected parameters, that
is, pw = pb = p, which greatly simplifies the implementa-
tion. Through our ablation study, we find that RaPA per-
forms well across a range of probabilities. For our experi-
ments, we will select the following DropConnect probabil-
ities: 0.05 for ResNet-50, 0.02 for Inception-v3, 0.04 for
DenseNet-121, 0.01 for Vision Transformer, and 0.03 for
CLIP. By default, RaPA applies DropConnect to all linear
and normalization layers in the surrogate model.
4.2. Main Result
We first study the performance of RaPA on the ImageNet-
Compatible dataset. We employ ResNet-50, Inception-v3,
DenseNet-121, and ViT as surrogate models, and evaluate
the obtained adversarial examples on 16 target models.
Table 3 reports the experimental results when adversar-
ial examples are generated using CNN-based models and
transferred to Transformer-based neural networks. This task
is considered more challenging in the context of transfer-
based attacks [32], as the ASRs in this case are relatively
low. Our method significantly improves the attack perfor-
mance over existing methods: it increases the average ASR
from 33.3% to 45.0% with ResNet-50 as surrogate model,
and from 22.8% to 40.3% with DenseNet-121.
Table 4 reports the ASRs of various attack methods on
ten CNN-based target models.
RaPA achieves the best
average ASR. Particularly, with Inception-v3 as surrogate
model, the ASRs are increased by 14.6% and 20.7% for the
challenging target models VGG16 and MBv2, respectively.
When transferring from Transformer-based model ViT to
CNN-based models, RaPA again attains the best average
ASR 51.2%. We also report the results of self-ensemble
methods MUP [58] and SE-ViT [34]. RaPA clearly outper-
forms these two methods by a large margin.
Additional experimental results can be found in Ap-
pendix D.4. We also visualize the heatmaps of some adver-
sarial examples in Appendix C for qualitative comparison.
4.3. Ablation Study
In this section, we conduct an ablation study to investigate
the impacts of 1) different types of layers where DropCon-
nect is applied, 2) different DropConnect probabilities and
3) more iterations and inferences.
6543

---

<!-- page 7 -->
Source : Incv3
Target model
Attack
RN18
RN50
VGG16
Incv3
EFB0
DN121
MBv2
IRv2
Incv4
Xcep
Avg.
DI
2.2
3.9
3.4
99.1
3.6
5.0
1.2
7.7
8.9
7.0
14.2
RDI
5.8
5.5
3.9
99.0
8.0
8.5
3.8
18.6
18.8
11.1
18.3
SI
6.7
6.7
4.3
98.8
9.7
9.7
4.4
23.3
22.1
13.6
19.9
MUP
13.9
13.6
9.6
98.4
17.7
22.4
8.1
42.2
42.2
26.4
29.5
BSR
15.8
13.9
11.9
98.7
20.5
24.3
9.6
45.7
45.5
30.3
31.6
DWP
17.5
17.2
13.5
99
19.2
29.8
9.7
54.4
52.9
37.5
35.1
Admix
18.5
16.7
13.8
98.1
23.8
27.5
15.9
46.3
47.1
38.9
34.7
SIA
17.4
21.5
16.7
98.8
27.2
32.2
13.1
56.1
59.2
42.9
38.5
ODI
14.4
22.3
22.0
99.4
26.0
39.5
13.9
51.8
60.7
44.7
39.5
CFM
37.4
37.9
27.3
97.9
46.1
53.0
27.8
76.9
76.1
68.2
54.9
RaPA
51.3±0.6
53.5±1.0
41.9±0.6
97.4±0.0
60.8±0.3
68.4±1.4
48.5±0.0
86.7±0.3
87.5±0.4
84.0±0.0
68.0
Source : ViT
Target model
Attack
RN18
RN50
VGG16
Incv3
EFB0
DN121
MBv2
IRv2
Incv4
Xcep
Avg.
DI
0.5
1.0
0.8
2.1
2.1
1.1
1.3
1.8
1.6
1.4
1.4
RDI
1.7
2.4
1.3
4.1
6.3
4.1
2.7
5.8
4.9
4.6
3.8
SI
2.9
3.9
1.1
8.0
9.2
5.9
2.7
7.9
6.2
6.8
5.5
BSR
5.0
8.2
3.9
11.2
15.0
12.8
5.3
15.5
13.4
11.2
10.2
DWP
13.5
11.7
7.1
15.8
22.2
16.7
10.2
17.4
16.3
16.3
14.7
SE-ViT
8.9
12.0
6.9
21
25.3
20.5
8.3
23.8
22.5
20.3
17.0
Admix
16.4
20.4
13.2
31.8
38.3
31.5
17.8
35.3
31.9
29.8
26.6
SIA
3.6
4.8
3.0
8.1
12.2
8.4
3.5
8.9
10.1
8.3
7.1
ODI
12.4
20.0
10.6
28.3
28.9
30.9
10.4
35.5
34.4
27.9
23.9
CFM
26.1
33.4
18.0
45.2
56.8
47.3
23.2
54.5
49.9
46.2
40.1
RaPA
37.2±0.7
42.9±0.7
28.6±2.0
57.5±1.4
68.2±0.4
56.8±0.2
36.4±0.7
62.9±0.4
62.6±0.1
58.3±1.0
51.2
Table 4. ASRs (%) against ten target models on the ImageNet-Compatible dataset. All the attack methods are combined with MI-TI. The
best results are shown in bold and the second best results are underlined.
100
300
500
700
900
1100
1300
1500
Iteration
20
30
40
50
60
70
80
Attack Success Rates (%)
(a) S = 1
100
300
500
700
900
1100
1300
1500
Iteration
20
30
40
50
60
70
80
Attack Success Rates (%)
(b) S = 5
100
300
500
700
900
1100
1300
1500
Iteration
20
30
40
50
60
70
80
Attack Success Rates (%)
(c) S = 10
RaPA
CFM
Admix
RDI
DI
Figure 2. Average ASRs along optimization iterations. Here S denotes the number of inferences per iteration.
Different Types of Layers
We use ResNet-50 and ViT
as surrogate models to analyze the impacts of layer types.
For ResNet-50, we apply RaPA to different combinations
of Batch Normalization (BN) layer, Fully Connected (FC)
layer, and Convolutional (Conv) layer. Similarly, we con-
sider Layer Normalization (LN) and FC layers (including
linear transformation layer in the attention layer) for ViT.
Table 5 presents the experimental results. Notably, sim-
ply applying RaPA to all layers achieves equal or higher
ASRs, compared with other baselines. The combination of
BN (or LN) and FC layers performs the best, which vali-
dates our implementation in Section 3. For ResNet-50, ap-
plying DropConnect to Conv layers performs worse than
BN layers. We conjecture that Conv layers have sparser
weights and may be less affected by over-reliance issue.
Applying DropConnect only to FC layers yields particularly
low ASRs, as ResNet contains only a single FC layer.
DropConnect Probability
We investigate the impact of
varying DropConnect probabilities p using ResNet-50 as a
surrogate model. RaPA is run with p ranging from 0.01 to
0.09, and we report the average ASRs over 16 models in
Appendix D.3. The mean ASR is 66.3% with a standard
deviation of 5.9%, peaking at 72.4% when p = 0.05. No-
tably, with p ∈[0.03, 0.07], RaPA consistently outperforms
baselines by over 2%, underscoring the stability of the pro-
posed method across different choices.
More Iterations and Inferences
We study performance
under different total iterations and numbers of inferences
per iteration, denoted as T and S, respectively. We use
ResNet-50 as the surrogate model and evaluate how well
adversarial examples transfer to the 16 target models.
6544

---

<!-- page 8 -->
Source Model
BN
FC
Conv
ASRs (%)
RN50
✓
72.1
✓
41.1
✓
65.1
✓
✓
72.4
✓
✓
69.1
✓
✓
64.7
✓
✓
✓
69.2
Source Model
LN
FC
-
ASRs (%)
ViT
✓
58.6
✓
63.9
✓
✓
65.2
Table 5. Applying DropConnect to different types of layers. The
reported ASRs(%) are averages over 16 target models.
RaPA
CFM
Admix
RDI
0
20
40
60
80
100
Attack Success Rates (%)
54.5
+15.9
+3.8
59.2
+5.9+1.9
39.9
+10.9
+2.7
37.5
+3.7+0.4
S = 1, T = 300
S = 5, T = 500
S = 10, T = 1500
Figure 3. Average ASRs with different iterations (T) and different
numbers of inferences per iteration (S).
The results are reported in Figure 2. Although existing
methods may also benefit from additional optimization it-
erations, RaPA and Admix have the best gains when T in-
creases, while RaPA achieves a much higher ASR than Ad-
mix. With S increasing, RaPA can outperform CFM even
at an early stage of the optimization process. We also de-
pict Figure 3 to ease the comparison of the gains of different
methods when both T and S increase. As we observe, RaPA
benefits the most from an additional compute budget.
4.4. Attack Performance Against Defenses
We evaluate RaPA against several defenses: adversarially
trained ResNet-50 (advRN) [39], Ensemble-Adversarial-
Inception-ResNet-v2 (ensIR) [21], High-level representa-
tion Guided Denoiser (HGD) [28], Bit Depth Reduction
(Bit) [15], JPEG compression [15], R&P [55], and Diffpure
[35]. We utilize ResNet-50 as the surrogate model. For Bit,
JPEG, and R&P, the target model is ResNet-18 and for Diff-
pure, the target model is ResNet-50. As shown in Table 6,
RaPA outperforms all other baselines. Notably, against the
strong defenses ensIR and HGD, RaPA exceeds the second-
best ASRs by 29.4% and 10.5%, respectively.
RN50
Defense methods
Method
advRN
ensIR
JPEG
Bit
R&P
HGD
Diffpure
DI
10.6
0.0
26.7
50.5
41.1
0.1
0.0
RDI
39.6
0.8
58.8
75.8
71.3
0.8
0.0
SI
61.8
9.4
75.1
80.7
78.8
2.1
0.4
Admix
68.9
5.7
76.9
81.4
78.6
2.4
0.7
BSR
60.4
3.0
76.8
85.2
83.4
2.6
0.1
ODI
58.6
5.1
71.7
73.8
76.1
0.3
0.3
CFM
84.1
13.8
88.2
91.5
89.3
15.2
0.5
RaPA
88.2
43.2
91.2
92.2
92.7
25.7
4.0
Table 6. ASRs (%) against six defense methods.
Attack
ViT
LeViT
ConViT
Twins
PiT
CLIP
Avg.
RaPA
33.8
75.4
27.6
59.5
57.3
15.6
45.0
DSM
8.1
49.5
7.8
31.6
23.3
3.1
20.6
DSM-RaPA
50.8
84.2
42.2
74.6
72.6
25.1
58.3
SASD-WS
30.1
74.1
25.6
52.1
52.9
18.1
42.2
SASD-RaPA
42.9
79.5
35.3
63.0
63.1
25.9
51.6
Table 7.
ASRs (%) of adversarial examples generated by
ResNet50 against Transformer-based targets, comparing RaPA
with training-based methods (DSM and SASD-WS) and their
combinations. Note that when RaPA is combined with SASD-WS,
we did not apply Weight Scaling.
4.5. Training-enhanced Frameworks
We compare RaPA with two training-dependent ap-
proaches, DSM[57] and SASD-WS[54], which involve ad-
ditional optimization to enhance surrogate models for bet-
ter adversarial transferability.
Specifically, DSM trains
a surrogate model with dark knowledge extracted from
a teacher model and enriched by mixing augmentation.
SASD-WS improves transferability via sharpness-aware
self-distillation and weight scaling, refining the loss land-
scape and model generalization. Table 7 shows that under
a fully training-free setting, RaPA already surpasses these
training-dependent methods. Furthermore, when integrated
with such training-based frameworks, RaPA continues to
deliver consistent gains. For instance, combining with DSM
increases the average ASR from 20.6% to 58.3%, highlight-
ing its compatibility with existing training-enhanced frame-
works.
5. Concluding Remarks
In this paper, we reveal the over-reliance issue in existing
transfer-based attacks, where adversarial examples depend
excessively on a small subset of model parameters.
To
alleviate this, we propose RaPA, which randomly prunes
surrogate parameters during optimization. We show that
the expected effect of random pruning equals adding an
importance-equalization regularizer, thereby reducing pa-
rameter over-reliance and improving transferability.
Ex-
tensive experiments on CNN and Transformer architectures
confirm the effectiveness and stability of RaPA.
6545

---

<!-- page 9 -->
6. Acknowledgement
This work was supported by the Strategic Priority Research
Program of the Chinese Academy of Sciences (Grant No.
XDB0680101); the National Key Research and Develop-
ment Program of China (Grant No. 2023YFA1011602); the
CAS Project for Young Scientists in Basic Research (Grant
No. YSBR-034); the Xiaomi Young Talents Program; and
the Innovation Project of Institute of Computing Technol-
ogy, Chinese Academy of Sciences (Grant No. E561130).
References
[1] Junyoung Byun, Seungju Cho, Myung-Joon Kwon, Hee-
Seon Kim, and Changick Kim. Improving the transferabil-
ity of targeted adversarial examples through object-based di-
verse input. In IEEE/CVF Conference on Computer Vision
and Pattern Recognition, 2022. 1, 3, 6, 12
[2] Junyoung Byun, Myung-Joon Kwon, Seungju Cho, Yoonji
Kim, and Changick Kim. Introducing competition to boost
the transferability of targeted adversarial examples through
clean feature mixup. In IEEE/CVF Conference on Computer
Vision and Pattern Recognition, 2023. 1, 3, 5, 6, 12
[3] Huanran Chen, Yichi Zhang, Yinpeng Dong, and Junyi Zhu.
Rethinking model ensemble in transfer-based adversarial at-
tacks. In International Conference on Learning Representa-
tions, 2024. 5, 12
[4] Franc¸ois Chollet. Xception: Deep learning with depthwise
separable convolutions. In IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition, 2017. 5
[5] Xiangxiang Chu, Zhi Tian, Yuqing Wang, Bo Zhang, Haib-
ing Ren, Xiaolin Wei, Huaxia Xia, and Chunhua Shen.
Twins: Revisiting the design of spatial attention in vision
transformers. In Advances in Neural Information Processing
Systems, 2021. 6
[6] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li,
and Li Fei-Fei. Imagenet: A large-scale hierarchical image
database. In IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2009. 6
[7] Yinpeng Dong, Fangzhou Liao, Tianyu Pang, Hang Su, Jun
Zhu, Xiaolin Hu, and Jianguo Li. Boosting adversarial at-
tacks with momentum. In IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition, 2018. 1, 3, 5, 6, 12
[8] Yinpeng Dong, Tianyu Pang, Hang Su, and Jun Zhu.
Evading defenses to transferable adversarial examples by
translation-invariant attacks. In IEEE/CVF Conference on
Computer Vision and Pattern Recognition, 2019. 3, 6, 12
[9] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov,
Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner,
Mostafa Dehghani, Matthias Minderer, G Heigold, S Gelly,
et al. An image is worth 16x16 words: Transformers for
image recognition at scale. In International Conference on
Learning Representations, 2020. 1, 6
[10] St´ephane d’Ascoli, Hugo Touvron, Matthew L Leavitt, Ari S
Morcos, Giulio Biroli, and Levent Sagun. Convit: Improving
vision transformers with soft convolutional inductive biases.
In International Conference on Machine Learning, 2021. 6
[11] Gongfan Fang, Xinyin Ma, Mingli Song, Michael Bi Mi, and
Xinchao Wang. Depgraph: Towards any structural pruning.
In IEEE/CVF Conference on Computer Vision and Pattern
Recognition, 2023. 13
[12] Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy.
Explaining and harnessing adversarial examples.
In
IEEE/CVF Conference on Computer Vision and Pattern
Recognition, 2015. 1, 2, 12
[13] Benjamin Graham, Alaaeldin El-Nouby, Hugo Touvron,
Pierre Stock, Armand Joulin, Herv´e J´egou, and Matthijs
Douze. Levit: a vision transformer in convnet’s clothing for
faster inference. In International Conference on Computer
Vision, 2021. 6
[14] Jindong Gu, Xiaojun Jia, Pau de Jorge, Wenqian Yu, Xin-
wei Liu, Avery Ma, Yuan Xun, Anjun Hu, Ashkan Khakzar,
Zhijiang Li, et al. A survey on transferability of adversar-
ial examples across deep neural networks. Transactions on
Machine Learning Research, 2024. 1
[15] Chuan Guo, Mayank Rana, Moustapha Cisse, and Laurens
Van Der Maaten. Countering adversarial images using input
transformations. In International Conference on Learning
Representations, 2018. 8
[16] Ali Hatamizadeh and Jan Kautz. Mambavision: A hybrid
mamba-transformer vision backbone.
In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, 2025. 15
[17] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
Deep residual learning for image recognition. In IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
2016. 1, 2, 5
[18] Byeongho Heo, Sangdoo Yun, Dongyoon Han, Sanghyuk
Chun, Junsuk Choe, and Seong Joon Oh. Rethinking spatial
dimensions of vision transformers. In International Confer-
ence on Computer Vision, 2021. 6
[19] Gao Huang, Zhuang Liu, Laurens Van Der Maaten, and Kil-
ian Q Weinberger.
Densely connected convolutional net-
works. In IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2017. 1, 2, 5
[20] Hao Huang, Ziyan Chen, Huanran Chen, Yongtao Wang, and
Kevin Zhang.
T-sea: Transfer-based self-ensemble attack
on object detection. In IEEE/CVF Conference on Computer
Vision and Pattern Recognition, 2023. 1, 3, 5, 12
[21] Alexey Kurakin, Ian Goodfellow, Samy Bengio, Yinpeng
Dong, Fangzhou Liao, Ming Liang, Tianyu Pang, Jun Zhu,
Xiaolin Hu, Cihang Xie, et al. Adversarial attacks and de-
fences competition. In The NIPS’17 Competition: Building
Intelligent Systems, 2018. 8
[22] Alexey Kurakin, Ian Goodfellow, Samy Bengio, Yinpeng
Dong, Fangzhou Liao, Ming Liang, Tianyu Pang, Jun Zhu,
Xiaolin Hu, Cihang Xie, et al. Adversarial attacks and de-
fences competition. In The NIPS’17 Competition: Building
Intelligent Systems, pages 195–231. Springer, 2018. 5
[23] Alexey Kurakin, Ian J Goodfellow, and Samy Bengio. Ad-
versarial examples in the physical world. In Artificial In-
telligence Safety and Security, pages 99–112. Chapman and
Hall/CRC, 2018. 2, 12
6546

---

<!-- page 10 -->
[24] Yann LeCun, John Denker, and Sara Solla. Optimal brain
damage. Advances in neural information processing systems,
2, 1989. 3
[25] Maosen Li, Cheng Deng, Tengjiao Li, Junchi Yan, Xinbo
Gao, and Heng Huang. Towards transferable targeted attack.
In IEEE/CVF Conference on Computer Vision and Pattern
Recognition, 2020. 12
[26] Yingwei Li, Song Bai, Yuyin Zhou, Cihang Xie, Zhishuai
Zhang, and Alan Yuille. Learning transferable adversarial
examples via ghost networks. In AAAI Conference on Artifi-
cial Intelligence, 2020. 1, 2, 3, 5, 12, 15
[27] Kaisheng Liang, Xuelong Dai, Yanjie Li, Dong Wang, and
Bin Xiao. Improving transferable targeted attacks with fea-
ture tuning mixup. In Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition, pages
25802–25811, 2025. 1, 3, 12
[28] Fangzhou Liao, Ming Liang, Yinpeng Dong, Tianyu Pang,
Xiaolin Hu, and Jun Zhu. Defense against adversarial at-
tacks using high-level representation guided denoiser.
In
IEEE/CVF Conference on Computer Vision and Pattern
Recognition, 2018. 8
[29] Jiadong Lin, Chuanbiao Song, Kun He, Liwei Wang, and
John E. Hopcroft. Nesterov Accelerated Gradient and Scale
Invariance for Adversarial Attacks. In International Confer-
ence on Learning Representations, 2020. 3, 6, 12
[30] Chuan Liu, Huanran Chen, Yichi Zhang, Yinpeng Dong, and
Jun Zhu.
Scaling laws for black box adversarial attacks.
arXiv preprint arXiv:2411.16782, 2024. 1, 5, 12
[31] Yanpei Liu, Xinyun Chen, Chang Liu, and Dawn Xiaodong
Song.
Delving into transferable adversarial examples and
black-box attacks. In International Conference on Learning
Representations, 2017. 1, 5, 12
[32] Kaleel Mahmood, Rigel Mahmood, and Marten Van Dijk.
On the robustness of vision transformers to adversarial ex-
amples. In International Conference on Computer Vision,
2021. 6
[33] Pavlo Molchanov, Arun Mallya, Stephen Tyree, Iuri Frosio,
and Jan Kautz. Importance estimation for neural network
pruning. In IEEE/CVF conference on computer vision and
pattern recognition, 2019. 3, 13
[34] Muzammal Naseer, Kanchana Ranasinghe, Salman Khan,
Fahad Shahbaz Khan, and Fatih Porikli. On improving ad-
versarial transferability of vision transformers. In Interna-
tional Conference on Learning Representations, 2022. 1, 3,
5, 6, 12
[35] Weili Nie, Brandon Guo, Yujia Huang, Chaowei Xiao, Arash
Vahdat, and Anima Anandkumar. Diffusion models for ad-
versarial purification. In International Conference on Ma-
chine Learning (ICML), 2022. 8
[36] Maxime Oquab, Timoth´ee Darcet, Theo Moutakanni, Huy V.
Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez,
Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Rus-
sell Howes, Po-Yao Huang, Hu Xu, Vasu Sharma, Shang-
Wen Li, Wojciech Galuba, Mike Rabbat, Mido Assran, Nico-
las Ballas, Gabriel Synnaeve, Ishan Misra, Herve Jegou,
Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bo-
janowski. Dinov2: Learning robust visual features without
supervision, 2023. 15
[37] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya
Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,
Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learn-
ing transferable visual models from natural language super-
vision. In International Conference on Machine Learning,
2021. 6
[38] Benjamin Recht, Rebecca Roelofs, Ludwig Schmidt, and
Vaishaal Shankar. Do imagenet classifiers generalize to im-
agenet?
In International conference on machine learning,
pages 5389–5400. PMLR, 2019. 15
[39] Hadi Salman, Andrew Ilyas, Logan Engstrom, Ashish
Kapoor, and Aleksander Madry. Do adversarially robust im-
agenet models transfer better? In Advances in Neural Infor-
mation Processing Systems, 2020. 8
[40] Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zh-
moginov, and Liang-Chieh Chen.
Mobilenetv2: Inverted
residuals and linear bottlenecks. In IEEE/CVF Conference
on Computer Vision and Pattern Recognition, 2018. 6
[41] Karen Simonyan and Andrew Zisserman. Very deep convo-
lutional networks for large-scale image recognition. In In-
ternational Conference on Learning Representations, 2015.
5
[42] Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan
Bruna, Dumitru Erhan, Ian Goodfellow, and Rob Fergus. In-
triguing properties of neural networks. In International Con-
ference on Learning Representations, 2014. 1, 2
[43] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon
Shlens, and Zbigniew Wojna. Rethinking the inception ar-
chitecture for computer vision. In IEEE/CVF Conference on
Computer Vision and Pattern Recognition, 2016. 6
[44] Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, and
Alexander Alemi. Inception-v4, inception-resnet and the im-
pact of residual connections on learning. In AAAI Conference
on Artificial Intelligence, 2017. 6
[45] Mingxing Tan and Quoc Le. Efficientnet: Rethinking model
scaling for convolutional neural networks. In International
Conference on Machine Learning, 2019. 6
[46] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszko-
reit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia
Polosukhin. Attention is all you need. In Advances in Neural
Information Processing Systems, 2017. 4
[47] Li Wan, Matthew Zeiler, Sixin Zhang, Yann Le Cun, and
Rob Fergus. Regularization of neural networks using drop-
connect. In International Conference on Machine Learning,
2013. 4
[48] Hung-Jui Wang, Yu-Yu Wu, and Shang-Tse Chen. Enhanc-
ing targeted attack transferability via diversified weight prun-
ing. In Proceedings of the IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition Workshops, 2024. 3
[49] Kunyu Wang, Xuanran He, Wenxuan Wang, and Xiaosen
Wang. Boosting adversarial transferability by block shuffle
and rotation. In IEEE/CVF Conference on Computer Vision
and Pattern Recognition, 2024. 1, 3, 6, 12
[50] Xiaosen Wang and Kun He. Enhancing the transferability of
adversarial attacks through variance tuning. In IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
2021. 12
6547

---

<!-- page 11 -->
[51] Xiaosen Wang, Xuanran He, Jingdong Wang, and Kun He.
Admix: Enhancing the transferability of adversarial attacks.
In International Conference on Computer Vision, 2021. 1, 3,
6, 12
[52] Xiaosen Wang, Zeliang Zhang, and Jianping Zhang. Struc-
ture invariant transformation for better adversarial transfer-
ability. In Proceedings of the IEEE/CVF International Con-
ference on Computer Vision, 2023. 1, 3, 6
[53] Sanghyun Woo, Shoubhik Debnath, Ronghang Hu, Xinlei
Chen, Zhuang Liu, In So Kweon, and Saining Xie. Con-
vnext v2: Co-designing and scaling convnets with masked
autoencoders. arXiv preprint arXiv:2301.00808, 2023. 15
[54] Han Wu, Guanyan Ou, Weibin Wu, and Zibin Zheng. Im-
proving transferable targeted adversarial attacks with model
self-enhancement. In Proceedings of the IEEE/CVF Confer-
ence on Computer Vision and Pattern Recognition, 2024. 2,
3, 8, 12
[55] Cihang Xie, Jianyu Wang, Zhishuai Zhang, Zhou Ren, and
Alan Yuille. Mitigating adversarial effects through random-
ization. In International Conference on Learning Represen-
tations, 2018. 8
[56] Cihang Xie, Zhishuai Zhang, Yuyin Zhou, Song Bai, Jianyu
Wang, Zhou Ren, and Alan L Yuille.
Improving trans-
ferability of adversarial examples with input diversity.
In
IEEE/CVF Conference on Computer Vision and Pattern
Recognition, 2019. 1, 3, 6, 12
[57] Dingcheng Yang, Zihao Xiao, and Wenjian Yu. Boosting
the adversarial transferability of surrogate models with dark
knowledge. In 2023 IEEE 35th International Conference on
Tools with Artificial Intelligence, 2023. 3, 8, 12
[58] Dingcheng Yang, Wenjian Yu, Zihao Xiao, and Jiaqi Luo.
Generating adversarial examples with better transferability
via masking unimportant parameters of surrogate model. In
International Joint Conference on Neural Networks, 2023. 1,
2, 3, 5, 6, 12
[59] Zhuolin Yang, Linyi Li, Xiaojun Xu, Shiliang Zuo, Qian
Chen, Benjamin Rubinstein, Ce Zhang, and Bo Li.
Trs:
Transferability reduced ensemble via encouraging gradient
diversity and model smoothness. In Advances in Neural In-
formation Processing Systems, 2021. 1
[60] Zhengyu Zhao, Zhuoran Liu, and Martha Larson. On suc-
cess and simplicity: A second look at transferable targeted
attacks. In Advances in Neural Information Processing Sys-
tems, 2021. 1, 5, 12
[61] Junhua Zou, Zhisong Pan, Junyang Qiu, Xin Liu, Ting Rui,
and Wei Li. Improving the transferability of adversarial ex-
amples with resized-diverse-inputs, diversity-ensemble and
region fitting. In European Conference on Computer Vision,
2020. 1, 3, 6, 12
6548