## Overall Architecture

Deconver is a U-shaped medical image segmentation network built around an encoder–decoder structure with skip connections. Its main difference from standard U-Net variants is that it replaces attention-based feature mixing with a learnable deconvolution-based mechanism. The model takes a 2D image or 3D volume as input and produces a segmentation logit map of the same spatial size. In the 3D setting described in the paper, the input has shape (C_{in}\times H\times W\times D), and the output has shape (C_{out}\times H\times W\times D). The architecture diagram on page 3 shows the full pipeline: a stem first projects the input into feature space, the encoder gradually compresses spatial resolution while increasing channels, the decoder restores resolution, and a final head generates class logits. 

### Encoder

The encoder contains (L) stages. It begins with a stem layer implemented as a (3\times 3\times 3) convolution that maps the raw input to (C_0) channels. After that, each stage applies a Deconver block, which is the core processing unit of the model. Between stages, strided convolutions with stride 2 reduce spatial dimensions by half and increase the number of feature channels. The channel count at stage (\ell) follows
[
C_\ell = \min(C_0 \cdot 2^\ell, 512).
]
This design lets the network learn increasingly abstract features as it moves deeper into the encoder. Larger receptive fields at coarser resolutions help the model capture broader anatomical context that is useful for segmentation. 

### Decoder

The decoder mirrors the encoder. It uses transposed convolutions with stride 2 to upsample the feature maps back to higher spatial resolutions. At every level, the upsampled decoder features are concatenated with the encoder features from the matching resolution through skip connections. These skip connections are essential because they preserve fine spatial information that would otherwise be lost during downsampling. After the last decoder stage, a pointwise (1\times 1\times 1) convolution produces the final segmentation logits, which can then be passed through sigmoid or softmax depending on whether the task is binary or multi-class. 

---

## Deconvolution Background Used by the Architecture

Before defining the network block, the paper introduces the deconvolution problem that motivates the design. The idea is to recover a latent source image (S) from an observed image (X) using a filter tensor (V), with the model
[
X \approx S * V.
]
Here, the operator is written in the paper using CNN-style cross-correlation indexing. In the nonnegative setting, the observed image, latent source, and filter are all constrained to be nonnegative. The goal is to estimate (S) by minimizing the reconstruction error
[
|X - S * V|_F^2
]
subject to (S \ge 0). This matters because Deconver does not use deconvolution as an external preprocessing step. Instead, it inserts a learnable version of this inverse process directly inside the network to improve feature representations. 

### Multiplicative Update Rule

The paper derives a multiplicative update rule for refining the source estimate. Starting from an initial nonnegative source (S^{(0)}), the update corrects it elementwise using a ratio between a data-consistency term and a normalization term. This update preserves nonnegativity and serves as the mathematical basis of the NDC layer used later in the network. In other words, the NDC layer is an optimization-inspired operation: it computes an initial source estimate and then refines it with one deconvolution-style update step. 

---

## Deconver Block

The Deconver block is the basic processing unit used throughout the encoder and decoder. It replaces the attention-heavy transformer block with a lighter structure based on deconvolution. As shown in the block diagram on page 4, the block contains two sequential submodules:

1. Deconv Mixer
2. MLP

Each submodule is preceded by instance normalization and followed by a residual connection. For an input feature map (X), the block computes
[
Z = \text{DeconvMixer}(\text{InstanceNorm}(X)) + X,
]
[
Y = \text{MLP}(\text{InstanceNorm}(Z)) + Z.
]
This residual design helps stabilize optimization while allowing the block to refine spatial and channel-wise features separately. 

### Why Instance Normalization

The paper uses instance normalization instead of layer normalization because medical image segmentation often relies on small batch sizes, especially for 3D volumes or high-resolution images. Instance normalization is more suitable in that setting and is therefore better aligned with the practical training conditions of the model. 

### MLP Submodule

The MLP consists of two pointwise convolutions separated by a GELU activation:
[
\text{MLP}(X) = \text{PointwiseConv}(\text{GELU}(\text{PointwiseConv}(X))).
]
Its role is to mix information across channels while preserving spatial structure. The first pointwise convolution expands the channel dimension by a factor (\alpha), and the second projects it back to the original size. This gives the block nonlinear channel interaction after the spatial refinement performed by the Deconv Mixer. 

---

## Deconv Mixer

The Deconv Mixer is the module that replaces self-attention inside the Deconver block. Its function is to model spatial dependencies and restore fine details without the computational cost of transformer attention.

### Processing Flow

Given an input feature map (X), the Deconv Mixer performs three operations:
[
X_1 = \text{PointwiseConv}(X),
]
[
X_2 = \text{NDC}(\text{ReLU}(X_1)),
]
[
\text{DeconvMixer}(X) = \text{PointwiseConv}(X_2).
]

This sequence can be interpreted as follows:

#### 1. Pointwise Projection

A pointwise convolution first remaps the input channels at every spatial location. This is a lightweight linear transformation that prepares the features for deconvolution-based processing. 

#### 2. ReLU for Nonnegativity

A ReLU activation is then applied to ensure the intermediate feature map is nonnegative. This is necessary because the following NDC layer assumes nonnegative inputs. 

#### 3. NDC-Based Spatial Refinement

The nonnegative feature map is passed into the NDC layer, which performs deconvolution-inspired feature refinement. This stage is responsible for capturing spatial dependencies and recovering high-frequency information that may have been weakened in previous layers. 

#### 4. Output Projection

A second pointwise convolution maps the NDC output back into the feature space expected by the rest of the network. All intermediate and output tensors preserve the same spatial size as the input. 

---

## Nonnegative Deconvolution Layer

The Nonnegative Deconvolution layer, or NDC layer, is the main innovation of Deconver. It embeds nonnegative deconvolution as a differentiable neural network layer.

### Grouped Processing

The NDC layer receives a nonnegative input tensor (X \in \mathbb{R}_{\ge 0}^{C \times H \times W \times D}) and splits it into (G) groups along the channel dimension. Each group is processed independently. For each group (g), the model learns:

* a nonnegative filter (V_g)
* a latent source representation (S_g)

This grouped design keeps the layer efficient and lets different channel groups specialize in different spatial patterns. 

### Source Channel Ratio

The layer introduces a source channel ratio
[
R = \frac{E}{C},
]
where (E) is the number of source channels. This ratio controls how much the latent source expands the channel dimension relative to the input. A larger (R) gives the layer more representational capacity, but also increases cost. 

### Learnable Initialization of the Source

For each group, the initial source estimate (S_g^{(0)}) is not fixed manually. It is generated from the input group using a pointwise convolution followed by ReLU. This makes the initialization adaptive and learnable while still ensuring nonnegativity. 

### Learnable Nonnegative Filter

Each group also has a learnable filter (V_g). The filter is initialized with Kaiming uniform initialization and then clamped to nonnegative values using ReLU before it is used in the deconvolution update. Its adjoint (V_g^{-}) is not learned separately; it is obtained from the same parameters by transpose-and-flip operations. This reduces parameter count and keeps the update rule structurally consistent. 

### Single-Step Refinement

The NDC layer applies one iteration of the multiplicative nonnegative deconvolution update:
[
S_g^{(1)} = S_g^{(0)} \odot
\frac{X_g * V_g^{-} + \epsilon}
{(S_g^{(0)} * V_g) * V_g^{-} + \epsilon}.
]
The constant (\epsilon = 10^{-8}) avoids division by zero. The numerator boosts regions where the current source underestimates the input, while the denominator normalizes the correction to avoid overshooting. After all groups are processed, their outputs are concatenated along the channel dimension. Spatial resolution stays unchanged, while the channel dimension is expanded by the factor (R). 

### Why Only One Iteration

Although classical deconvolution methods often use many iterations, the paper reports that one iteration is enough to achieve a strong trade-off between performance and efficiency. This is an important architectural decision because it keeps the layer lightweight enough to be inserted repeatedly inside the encoder and decoder. 

### Differentiability and End-to-End Training

The entire NDC layer is differentiable and compatible with backpropagation. That means the source initialization mapping, the nonnegative filters, and the surrounding pointwise convolutions are all trained jointly with the rest of the segmentation network. This lets the model learn how to use deconvolution as a feature transformation rather than as a fixed restoration module. 

---

## End-to-End Data Flow Through the Network

The full processing chain of Deconver can be described in a simple sequence.

### Step 1: Input Projection

The stem converts the input image or volume into an initial feature representation using a convolution. 

### Step 2: Multi-Scale Encoding

At each encoder stage, a Deconver block refines the features. Inside that block, the Deconv Mixer performs nonnegative deconvolution-based spatial enhancement, and the MLP performs channel mixing. After refinement, downsampling reduces spatial size and increases channels. 

### Step 3: Bottleneck Representation

At the deepest part of the network, the features have the richest semantic abstraction and the largest effective receptive field. This is where global contextual structure is most strongly encoded. 

### Step 4: Multi-Scale Decoding

The decoder upsamples the features back toward the original resolution. At each level, it concatenates encoder features through skip connections and applies Deconver blocks again to refine the fused representation. 

### Step 5: Segmentation Head

A final pointwise convolution maps the last decoder feature map to class logits for the output segmentation mask. 

---

## Key Design Choices

### U-Shaped Backbone

The architecture keeps the proven encoder–decoder geometry of U-Net because it is effective for combining global semantic understanding with precise spatial localization. 

### Deconvolution Instead of Attention

Rather than using self-attention to capture long-range structure, Deconver uses the NDC-based Deconv Mixer. The idea is that deconvolution can recover fine structures and suppress artifacts more efficiently than attention-heavy modules. 

### Residual Block Design

Each Deconver block uses normalization and residual connections around both the Deconv Mixer and the MLP, which helps optimization and allows repeated stacking across the encoder and decoder. 

### Grouped NDC for Efficiency

The grouped structure of the NDC layer greatly reduces parameter count and computation while preserving the ability to learn diverse spatial patterns across channels. 

### One-Step Optimization-Inspired Update

The architecture borrows an update rule from nonnegative deconvolution but compresses it into a single trainable layer step, making it suitable for deep learning while retaining the interpretability of an inverse-problem formulation. 

---

## Reference

The architecture description is based on the paper:

Ashtari, P., Noei, S., Nateghi Haredasht, F., Chen, J. H., Jurman, G., Pizurica, A., and Van Huffel, S.  
**Deconver: A Deconvolutional Network for Medical Image Segmentation**.  
arXiv preprint arXiv:2504.00302, 2025.  
https://arxiv.org/abs/2504.00302
