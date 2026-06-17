# CS240 Final Project: Dynamic Programming and other alogorithm application for Content-Aware Image Resizing

This repository contains my CS240 final project for **Algorithm Design and Analysis**.

The project studies **content-aware image resizing** using the classical **seam carving** algorithm. The main goal is to show how a modern image-processing problem can be formulated and solved using classical algorithmic ideas, especially **dynamic programming** and **shortest paths on grid graphs**.

---

## 1. Project Topic

**Topic:** Content-Aware Image Resizing
**Core Algorithm:** Dynamic Programming for Seam Carving
**Course Concepts:** Dynamic Programming, Shortest Paths, Greedy Algorithms, Runtime Analysis
**Application:** Image retargeting and resizing for different display sizes

A standard image resize operation changes the whole image uniformly, which may distort important objects. Seam carving instead removes low-energy paths of pixels, called **seams**, so that less important regions are removed first while more visually important regions are preserved.

---

## 2. Problem Formulation

Given an input image and a target width or height, the task is to produce a resized image while preserving important visual content.

For a vertical seam, we select one pixel from each row:

\[
S = {(i, j_i) : i = 1,2,\ldots,H}
\]

where adjacent seam pixels must be connected:

\[
|j_i - j_{i-1}| \le 1
\]

Given an energy matrix (E), the cost of a seam is

\[
\sum_{(i,j)\in S} E[i,j]
\]

The goal is to find a valid seam with minimum total energy, remove it from the image, and repeat the process until the target size is reached.

---

## 3. Algorithms to Implement

This project will implement and compare the following methods.

### 3.1 Standard Resize Baseline

A simple baseline using library interpolation. This method resizes the whole image uniformly and does not consider image content.

### 3.2 Greedy Seam Selection

A local greedy method that chooses a low-energy path step by step. This method is simple and fast, but it may fail to find the globally minimum-energy seam.

### 3.3 Dynamic Programming Seam Carving

The main method of this project.

Let

\[
DP[i][j]
\]

be the minimum cumulative energy of a vertical seam ending at pixel ((i,j)). The recurrence is

\[
DP[i][j]
=

E[i][j]
+
\min{DP[i-1][j-1], DP[i-1][j], DP[i-1][j+1]}
\]

After filling the DP table, the minimum-energy seam is recovered by backtracking from the minimum entry in the last row.

For an image of size (H \times W), finding one seam takes

\[
O(HW)
\]

time. Removing (r) seams takes approximately

\[
O(rHW)
\]

time.

---

## Planned Experiments

The project will evaluate the implemented methods on a small collection of test images.

The planned experiments include:

1. qualitative visual comparison between standard resizing, greedy seam removal, and DP seam carving;
2. runtime comparison between greedy and dynamic programming seam selection;
3. scalability analysis as image size increases;
4. comparison of results on different image types, such as landscapes, buildings, and images with clear foreground objects.

The main goal is not to achieve state-of-the-art image retargeting performance, but to clearly demonstrate the algorithmic behavior of dynamic programming in this application.

---

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Create sample images from `scikit-image`:

```bash
python experiments/make_sample_data.py
```

Run one image experiment:

```bash
python experiments/run_single_image.py --input data/input/astronaut.png --target-width 360 --methods all
```

Run batch experiments:

```bash
python experiments/run_batch.py --target-width-ratio 0.75 --methods standard dp greedy
```

Run runtime analysis:

```bash
python experiments/runtime_analysis.py --input data/input/astronaut.png
```

Run tests:

```bash
pytest tests
```

## Algorithms

### Standard Resize

A simple baseline using library interpolation.

### Greedy Seam Selection

A local greedy method that chooses a low-energy path step by step.

### Dynamic Programming Seam Carving

Let \(DP[i][j]\) be the minimum cumulative energy of a vertical seam ending at pixel \((i,j)\). The recurrence is

\[
DP[i][j]
=
E[i][j]
+
\min\{DP[i-1][j-1], DP[i-1][j], DP[i-1][j+1]\}
\]

For an image of size \(H \times W\), finding one seam takes \(O(HW)\) time. Removing \(r\) seams takes approximately \(O(rHW)\) time.

## Repository Structure

```text
cs240-seam-carving/
├── data/input/                 # input images
├── data/output/                # resized images
├── src/                        # core implementation
├── experiments/                # runnable experiment scripts
├── results/figures/            # comparison figures
├── results/tables/             # csv results
└── tests/                      # basic tests
```

## References

1. Owais Aijaz, Syed Muhammad Ali, and Yousuf Uyghur.
   *Analysis of Different Algorithmic Design Techniques for Seam Carving*. arXiv preprint arXiv:2410.21207, 2024.

2. Feihong Shen, Chao Li, Yifeng Geng, Yongjian Deng, and Hao Chen.
   *Prune and Repaint: Content-Aware Image Retargeting for Any Ratio*. NeurIPS, 2024.

3. Tim Elsner, Julia Berger, Tong Wu, Victor Czech, Lin Gao, and Leif Kobbelt.
   *Retargeting Visual Data with Deformation Fields*. ECCV, 2024.

---

## 9. Current Status

* [x] Topic selected
* [x] Proposal drafted
* [x] Repository setup
* [ ] Energy function implementation
* [ ] DP seam search implementation
* [ ] Greedy baseline implementation
* [ ] Standard resize baseline
* [ ] Experiments
* [ ] Final report
* [ ] Presentation slides
