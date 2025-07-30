# Harmonic Clustering Challenges in Real-Time Pulsar Search (SPOTLIGHT)

This repository contains code developed to analyze candidate outputs from AstroAccelerate, a real-time GPU-based pulsar search tool used in the SPOTLIGHT project — a commensal transient and pulsar survey conducted with the GMRT.

Unlike traditional software such as PRESTO, which typically returns the fundamental frequency of a pulsar, AstroAccelerate often outputs harmonics — higher integer multiples of the fundamental frequency. This is a natural consequence of Fourier-domain acceleration searches, where narrow pulses can have stronger signal-to-noise ratios at their harmonics than at the base frequency.

While this approach enables real-time performance, it introduces complications in the clustering and sifting stages of the pipeline. These stages are designed to group multiple detections of the same pulsar and filter out duplicates or noise. However, if the detected candidates are primarily harmonics rather than fundamentals, the clustering algorithm may fail to associate them correctly, treating each harmonic as a separate source. This can result in missed detections or misclassification of valid pulsar signals.

The purpose of this code is to investigate and illustrate how the presence of harmonics affects the clustering process. It provides insights into:

- How harmonics replace fundamentals in the candidate list?
- Why clustering can fail in such scenarios?
- How future algorithms might be improved to correctly identify fundamental periods even when harmonics dominate?

This analysis is crucial for improving the robustness and reliability of real-time pulsar detection pipelines in high-throughput environments like SPOTLIGHT.
