# Methodology

## Problem Formulation

The task addressed in this study was the automated detection and segmentation of quantum-dot-like particles in grayscale microscopy-style images. The problem was formulated as a binary semantic segmentation task, where each input image was mapped to a foreground-background mask indicating the spatial support of particle-like structures. Instead of directly predicting particle coordinates or counts, the model first generated a segmentation mask, after which connected-component analysis could be used to estimate the number of detected dots.

This formulation was selected for three reasons. First, segmentation provides dense spatial supervision and is therefore easier to optimize than direct coordinate regression when precise point annotations are unavailable. Second, segmentation masks are visually interpretable, which is important for qualitative verification. Third, mask-based predictions can be post-processed into counts, overlays, and candidate centroids, making the model suitable both for quantitative evaluation and for downstream visual inspection.

## Synthetic Data Generation

Because no manually annotated real microscopy dataset was available, all training and evaluation data were generated synthetically. The synthetic data generator was implemented directly in Python, NumPy, and PyTorch, without relying on external microscopy simulation frameworks in the final model pipeline.

Each image was generated as a 128 × 128 single-channel grayscale array. A constant low-intensity background was first created. A random number of dots was then sampled from a predefined interval depending on the experimental condition. For each dot, a random center position was drawn uniformly over the image plane, a radius was sampled from a specified range, and an intensity value was sampled uniformly between 0.6 and 1.0. A Gaussian blob was rendered for each dot and added to the image. The target binary mask was constructed by thresholding each blob at half of its peak intensity and then taking the union across all dots.

To mimic imaging noise, additive Gaussian noise was applied after rendering. Finally, intensities were clipped to nonnegative values and normalized to the range [0, 1]. This procedure produced paired synthetic images and binary masks suitable for supervised segmentation.

The baseline distribution used for training and matched validation contained 5 to 20 dots per image, dot radii between 3.0 and 7.0 pixels, and Gaussian noise with standard deviation 0.04. Additional evaluation-only distributions were created to test generalization under controlled shifts in scene density, particle size, and noise level.

## Model Architecture

The segmentation network was a compact U-Net-style convolutional neural network implemented in plain PyTorch. The architecture consisted of an encoder path, a bottleneck, and a decoder path with skip connections linking corresponding encoder and decoder levels.

Each encoder and decoder stage used a double-convolution block comprising two 3 × 3 convolutional layers, each followed by batch normalization and ReLU activation. Downsampling was performed using max pooling, and upsampling was performed using transposed convolutions. The final output layer was a 1 × 1 convolution producing two channels corresponding to background and foreground.

The encoder channel widths were set to [32, 64, 128], providing a relatively lightweight network that remained expressive enough for the synthetic segmentation task while still being practical for local experimentation and rapid iteration.

## Training Procedure

The model was trained using the Adam optimizer with a learning rate of 0.001. Because the foreground occupied only a small fraction of the image area, a class-weighted cross-entropy loss was used, with weights [1.0, 10.0] for background and foreground respectively. This weighting reduced the incentive for the model to predict mostly background and improved learning on sparse foreground regions.

The baseline training configuration used 512 synthetic training samples and 128 validation samples, with a batch size of 8 and a total of 30 training epochs. A global random seed of 1337 was used to make the experimental procedure reproducible. Separate seeds were used for training and validation set generation to reduce overlap between the two splits.

Training was performed on the best available local device, with automatic selection between CUDA, Apple MPS, and CPU backends depending on hardware availability.

## Evaluation Metrics

Two primary quantitative metrics were used.

### Pixel Accuracy

Pixel accuracy was defined as the fraction of correctly classified pixels over the entire image. While this metric provides a simple measure of segmentation correctness, it is heavily influenced by the background class and can therefore overstate performance in sparse-object settings.

### Intersection over Union (IoU)

Intersection over Union was used as the main segmentation metric. IoU was computed as the ratio between the intersection and union of predicted foreground pixels and true foreground pixels. Because it directly measures foreground overlap quality, IoU is much more informative than pixel accuracy for sparse particle segmentation.

In addition to segmentation metrics, connected-component analysis was applied to predicted masks for inference purposes, allowing the model output to be converted into estimated particle counts and centroid coordinates. However, count accuracy itself was not used as the primary optimization objective during training.

## Experimental Design

A baseline experiment was first conducted using matched training and validation distributions. After baseline training was completed, the saved model was evaluated under several controlled distribution shifts without retraining. These additional experiments were designed to probe model robustness rather than maximize performance.

The following evaluation regimes were tested:

1. **Baseline regime:** 5 to 20 dots per image, radius 3.0 to 7.0 pixels, noise standard deviation 0.04.
2. **Sparse regime:** 1 to 5 dots per image.
3. **Dense regime:** 50 to 100 dots per image.
4. **Higher-noise regime:** 5 to 20 dots per image with noise standard deviation increased to 0.08.
5. **Small-dot regime:** 5 to 20 dots per image with dot radii reduced to 1.5 to 3.0 pixels.

This evaluation strategy allowed the model to be assessed not only on the distribution it was trained on, but also on several plausible failure modes. Such stress testing is important because high performance on a matched synthetic validation set alone does not demonstrate robustness.

# Results and Discussion

## Summary of Quantitative Results

The model achieved strong performance on the matched synthetic validation distribution and remained partially robust under several controlled shifts. Table 1 summarizes the key quantitative results.

**Table 1. Segmentation performance across evaluation regimes**

- Baseline (5 to 20 dots): pixel accuracy = 0.9988, IoU = 0.9444
- Sparse (1 to 5 dots): pixel accuracy = 0.9997, IoU = 0.9512
- Dense (50 to 100 dots): pixel accuracy = 0.9851, IoU = 0.8840
- Higher noise (5 to 20 dots, noise std 0.08): pixel accuracy = 0.9978, IoU = 0.9013
- Small dots (5 to 20 dots, radius 1.5 to 3.0): pixel accuracy = 0.9993, IoU = 0.8484

## Baseline Performance

On the matched validation distribution, the model achieved a pixel accuracy of 0.9988 and an IoU of 0.9444. The high IoU indicates that the network learned to localize the synthetic particle masks effectively. This result supports the conclusion that a relatively compact U-Net is sufficient for the synthetic segmentation problem under the chosen image formation assumptions.

Training dynamics also suggested that the task was comparatively easy under the matched synthetic setting. IoU increased rapidly within the first several epochs and exceeded 0.93 by epoch 6. By epoch 30, the model reached 0.9444 IoU, indicating convergence to a strong segmentation solution on the validation set.

## Generalization to Sparse Scenes

When the number of dots was reduced to 1 to 5 per image, the model achieved an IoU of 0.9512, slightly exceeding baseline performance. This is a reasonable outcome, since sparse scenes involve less overlap between particles and therefore create a cleaner segmentation task. In effect, the model was trained on a moderately crowded regime and therefore had little difficulty when the scene complexity was reduced.

This result suggests that the learned representation generalized well to simpler sparse cases, and that the baseline training regime did not over-specialize to crowded scenes.

## Generalization to Dense Scenes

Performance declined under the dense-scene stress test, where the number of dots increased to 50 to 100 per image. In this setting, the model achieved an IoU of 0.8840. Although this remains a respectable result, it is materially worse than the matched baseline.

The likely reason is increased overlap and crowding among nearby Gaussian blobs. As particles become more densely packed, boundaries between neighboring structures become less distinct, making it harder for a segmentation network to separate the union of dot regions accurately. This behavior is expected and indicates that density is an important axis of difficulty for the current model.

From an application standpoint, the dense-scene result shows that the model is not restricted to the exact training dot-count regime, but also that it should not be assumed to remain optimal as particle density increases substantially.

## Robustness to Higher Noise

When image noise was increased from a standard deviation of 0.04 to 0.08, the model achieved an IoU of 0.9013. This reduction from baseline indicates that the network is somewhat sensitive to degraded signal quality, but the degradation was not catastrophic.

This result suggests that the model learned features that are reasonably stable to moderate noise increases. However, the drop in IoU also implies that noise impacts mask boundary quality and may introduce either false positives or fragmented foreground regions. Therefore, although the model is fairly robust to noise, additional noise augmentation during training would likely improve generalization further.

## Sensitivity to Particle Size

The most challenging stress test was the small-dot regime, in which the dot radius range was reduced to 1.5 to 3.0 pixels. In this case, IoU dropped to 0.8484. This was the weakest performance observed among the tested conditions.

This finding is significant. It indicates that the model is strongly influenced by the particle scale distribution seen during training. The baseline model was optimized on particles with radii between 3.0 and 7.0 pixels, and therefore smaller structures occupy fewer pixels and present a more difficult segmentation target. Small objects are also inherently more vulnerable to both noise and discretization effects.

Importantly, pixel accuracy remained extremely high even in this setting. This again demonstrates that pixel accuracy is not a reliable standalone measure for sparse-object segmentation, since a model can miss small foreground structures while still correctly classifying most background pixels.

## Interpretation of the Results

Taken together, the results support three main conclusions.

First, the chosen plain PyTorch U-Net is sufficient to achieve high segmentation quality on matched synthetic data. The baseline IoU of 0.9444 demonstrates that the model can capture the relevant spatial patterns in the synthetic dataset.

Second, the model generalizes reasonably well to some distribution shifts, particularly lower object density and moderately increased noise. This suggests that the learned features are not entirely brittle and that the model does not merely memorize a narrow subset of training examples.

Third, performance degrades most clearly when the underlying object morphology changes, especially when the particles become substantially smaller. This indicates that object scale is an important determinant of model robustness and should be explicitly addressed in any future training strategy.

## Limitations

Several limitations must be acknowledged.

The most important limitation is that all reported results are based on synthetic data generated by the same procedural framework used for training. Consequently, the validation and stress-test distributions are still synthetic and may not capture the full variability of real microscopy images. As a result, the reported metrics likely overestimate real-world performance.

A second limitation is that counting performance was not directly optimized. Dot counts were inferred by connected-component analysis applied to segmentation masks. While this is practical, it is only an indirect route to counting and may fail when particles overlap or masks fragment.

A third limitation is that the data generator is simplified. Real microscopy images may contain structured background, non-Gaussian blur, intensity inhomogeneity, imaging artifacts, or particle morphologies that are not well represented by the current generator.

Finally, the study did not include an ablation analysis. For example, no controlled comparison was performed between different U-Net widths, loss functions, or training distributions. Such experiments would be necessary to determine which design choices contributed most strongly to the observed performance.

## Implications and Future Work

Despite these limitations, the results demonstrate that a compact segmentation network trained purely on synthetic data can achieve strong performance on a controlled synthetic quantum-dot segmentation task. This makes the approach useful as a proof of concept and a foundation for further work.

Future extensions should include:

- expanding the training distribution to include a wider range of particle sizes and densities,
- increasing realism in the synthetic image formation process,
- evaluating count accuracy directly rather than relying only on segmentation metrics,
- testing on manually annotated real microscopy data,
- and comparing segmentation-based counting with alternative approaches such as heatmap regression or direct object detection.

In summary, the present results show that the proposed model is effective on matched synthetic data and reasonably robust to some controlled distribution shifts, but they also make clear that synthetic success should not be confused with validated real-world performance.
