# Temporal Clustering of LEGO Set Design: Detecting Structural Drift Across Decades

## Table of Contents

1. [Introduction](#1-introduction)
2. [Problem Statement](#2-problem-statement)
3. [Data Source](#3-data-source)
4. [Methodology](#4-methodology)
   - [4.1 Preprocessing](#41-preprocessing)
   - [4.2 Feature Engineering](#42-feature-engineering)
   - [4.3 Dimensionality Reduction for Visualization](#43-dimensionality-reduction-for-visualization)
   - [4.4 Temporal Clustering](#44-temporal-clustering)
   - [4.5 Robustness Check via DBSCAN](#45-robustness-check-via-dbscan)
   - [4.6 Drift Detection](#46-drift-detection)
   - [4.7 Reproducibility](#47-reproducibility)
5. [Results](#5-results)
   - [5.1 Optimal Number of Clusters](#51-optimal-number-of-clusters)
   - [5.2 Temporal Clustering](#52-temporal-clustering)
   - [5.3 DBSCAN Robustness and External Validation](#53-dbscan-robustness-and-external-validation)
   - [5.4 Limitations and Future Work](#54-limitations-and-future-work)
6. [Conclusion](#6-conclusion)
7. [Appendix](#7-appendix)
8. [References](#references)

---

## 1. Introduction

LEGO has a fascinating history. Originating in Denmark in 1932, it began as a small wooden toy manufacturer. Since then it has grown into one of the most iconic toys ever produced. The LEGO brick as we know it today was introduced in 1958, and since then the brand has released thousands of distinct sets spanning dozens of licensed and original themes. Everything from the earliest Town and Castle series released in the 1970s, to the endlessly popular Star Wars licensed sets released beginning in the late 1990s, to the adult-targeted Icons and Architecture lines released for those of us who haven't fully grown up yet — each set comes with hundreds of individual parts, colors, minifigures and other characteristics that make it both distinct and part of a greater collection.

Over the years, LEGO sets have evolved in piece count, colors, set complexity, and other key changes that make LEGO what they are. Things like the first LEGO sets with instructions being introduced in 1964, LEGO Technic introducing an entirely new style of piece added in 1977, the first Star Wars licensed set released in 1999, and robotics getting added with LEGO Mindstorms in 2006 have all been critical additions which have shaped the brand and the composition of those sets. This project examines that composition and how it has changed over the years.

## 2. Problem Statement

> *Can unsupervised machine learning detect meaningful structural shifts in LEGO set design over time, and do those detected shifts correspond to known historical events in the company's history?*

This is the question that this project answers using temporal clustering [[Chi et al., 2007]](#references) and distributional drift detection [[Gama et al., 2014]](#references). Over the past several decades, LEGO has expanded from a company focusing primarily on traditional building sets to a global entertainment brand encompassing licensed franchises, collector-focused products, and highly specialized themes. These changes are incredibly well documented from a business perspective, but it can be very unclear how LEGO's product designs have evolved over time when viewed solely through the characteristics of the sets.

The objective of this project is to analyze the evolution of LEGO products from the 1950s through the present day using unsupervised machine learning techniques. Specifically, LEGO sets will be grouped into clusters based on structural and design-related features, and the analysis examines how these clusters change over time. When looking at the shifts in the composition of clusters across different years, the expectation is that certain substantial changes will appear aligning with major changes to LEGO's product portfolio.

Let $S = \{s_1, s_2, \dots, s_N\}$ denote the full catalog of official LEGO sets — those released for standard sale, not including promotional materials, special editions, or subsets sold as larger groups. Each set $s_i$ is characterized by a feature vector $\mathbf{x}_i \in \mathbb{R}^d$ derived from its structural composition. Each set carries a release year $y_i \in \{1955, \dots, 2026\}$. We partition $S$ into cohorts made up of one or more years, $C_t = \{s_i : y_i = t\}$, for each time period $t$.

There are two ideas at the core of this problem:

1. **Temporal Clustering:** For each cohort $C_t$, this project fits a clustering model $K_t$ that partitions sets into $k$ groups based on structural similarity. The cluster centroids $\{\mu_1^{(t)}, \dots, \mu_k^{(t)}\}$ are tracked for each period $t$.
2. **Drift Detection:** Measuring the degree of distributional change between consecutive cohort clusterings $K_t$ and $K_{t+1}$ using a quantitative dissimilarity metric. This identifies specific years, $t^*$, where this dissimilarity exceeds a threshold. These years are noted as *drift points*.

A formal notation of the goal of this project is to estimate:

$$
\Delta_t = \text{drift}(K_t, K_{t+1}), \quad t \in \{1955, \dots, 2026\}
$$

where $\text{drift}(K_t, K_{t+1})$ is the drift shown by clustering dissimilarity, measured using centroid displacement and the Adjusted Rand Index. A detected drift point is defined as:

$$
t^* = \{t : \Delta_t > \mu_{\Delta} + c\,\sigma_{\Delta}\}
$$

where $c$ is a tunable threshold and $\mu_{\Delta}$ and $\sigma_{\Delta}$ are the mean and standard deviation of $\Delta_t$ across all years.

The hypothesis is that drift points $t^*$ will cluster near known historical milestones in LEGO's product history — for example, the introduction of the Star Wars license in 1999, the near-bankruptcy of the company in 2003–2004, the growth of licensed themes throughout the 2010s, and the growth of adult-targeted sets in the 2020s. If these events can be detected using structural, label-free features, this would constitute a meaningful unsupervised result.

## 3. Data Source

Data has been sourced from the Rebrickable database [[Rebrickable, 2024]](#references), which provides both API and CSV exports for all LEGO set–related data. For this project, the CSV export of a large relational database was used, including the following tables: `sets`, `themes`, `parts`, `part_categories`, `colors`, `inventories`, `inventory_parts`, `inventory_minifigs`, and `minifigs`. The complete entity relationship diagram including all attributes can be found in [Appendix A](#71-rebrickable-entity-relationship-diagram). These were loaded into a local SQLite database via Python's `sqlite3` module and queried using `pandas`.

The complete dataset is 149.3 MB, containing over 21,500 individual LEGO sets.

## 4. Methodology

### 4.1 Preprocessing

The data has some flaws which were handled in preprocessing. The primary one to fix is the large number of sets containing one piece. LEGO assigns a set number to all items it sells, including merchandise that is not bricks at all (see the `num_parts` distribution below).

> **Figure 1.** Histogram showing the number of sets containing a specific number of parts. Notice the large skew towards the left, with a disproportionate number of sets sitting at 1. The red vertical line represents the cutoff line for this analysis. Sets with fewer parts are considered "not standard." While some "standard" sets may be eliminated, this is determined to be the best way to get a clean dataset.
> `images/num_parts_hist.png`

Sets with a very small number ($n < 20$) of parts were excluded, as well as non-standard releases (e.g., promotional packs, individual minifigures, individual parts sold as a set, merchandise with set numbers). After preprocessing, just over 12,000 standard LEGO sets remained. Additionally, years with a small number of sets released in the catalog were grouped using rolling windows to ensure a large enough cohort size for stable clustering. Grouping was accomplished by setting a minimum cohort size (100) along with a maximum number of years per cohort (12). This ensured sufficient cohort size while not allowing too many years to be clustered together. Years with more than the minimum cohort size were evaluated alone. A detailed breakdown of the number of sets included in each cohort can be found in [Appendix B](#72-breakdown-of-number-of-sets-per-time-window).

### 4.2 Feature Engineering

Each set $s_i$ is represented by a $d$-dimensional feature vector spanning four categories:

- **Scale:** total part count, unique part count, minifigure count, spare part ratio
- **Diversity:** Shannon entropy of part categories ($H_{\text{cat}}$), Shannon entropy of colors ($H_{\text{col}}$), number of distinct colors, number of distinct part categories
- **Composition:** proportion of Technic parts, proportion of decorative/printed parts, proportion of transparent parts (via `colors.is_trans`), rare part ratio (parts appearing in fewer than 20 sets)
- **Minifigure characteristics:** minifigure-to-part ratio, a binary indicator for minifigure presence

Before modeling, this candidate set was screened for multicollinearity using Variance Inflation Factors (Table 1). Two features stood out immediately: `total_part_count` and `num_parts` had VIF values above 7000, reflecting their near-total redundancy, and `n_distinct_categories` exceeded the common threshold of 10. Both `total_part_count` and `n_distinct_categories` were removed, and VIF was recomputed on the remaining twelve features (Table 2); all values fell below 5, indicating the reduced set carries largely independent information.

**Table 1. Variance Inflation Factors (Original Feature Set)**

| # | Feature | VIF |
|---|---|---:|
| 0 | num_parts | 8299.992733 |
| 1 | total_part_count | 8145.789045 |
| 2 | unique_part_count | 13.160328 |
| 3 | minifig_count | 5.305488 |
| 4 | spare_part_ratio | 1.297114 |
| 5 | h_cat | 5.350565 |
| 6 | h_col | 3.749628 |
| 7 | n_distinct_colors | 7.882505 |
| 8 | n_distinct_categories | 15.857922 |
| 9 | prop_technic | 1.231668 |
| 10 | prop_printed | 1.929013 |
| 11 | prop_transparent | 1.186414 |
| 12 | rare_part_ratio | 2.373186 |
| 13 | minifig_to_part_ratio | 1.317496 |
| 14 | has_minifigs | 2.428044 |

**Table 2. Variance Inflation Factors (Reduced Feature Set)**

| # | Feature | VIF |
|---|---|---:|
| 0 | num_parts | 2.904824 |
| 1 | unique_part_count | 4.509105 |
| 2 | minifig_count | 1.952869 |
| 3 | spare_part_ratio | 1.245666 |
| 4 | h_cat | 2.159906 |
| 5 | h_col | 1.521950 |
| 6 | prop_technic | 1.224180 |
| 7 | prop_printed | 1.924807 |
| 8 | prop_transparent | 1.185894 |
| 9 | rare_part_ratio | 2.363765 |
| 10 | minifig_to_part_ratio | 1.286617 |
| 11 | has_minifigs | 2.123624 |

The retained twelve features still span the same three functional roles: scale and part diversity (`num_parts`, `unique_part_count`), minifigure presence and intensity (`minifig_count`, `has_minifigs`, `minifig_to_part_ratio`), and material or stylistic composition (`h_cat`, `h_col`, `prop_technic`, `prop_printed`, `prop_transparent`, `rare_part_ratio`, `spare_part_ratio`). Rather than relying on piece count alone, this combination captures how a set is built. Each feature is normalized prior to clustering. Note that this VIF-reduced set is used only for the UMAP visualization described in [Section 4.3](#43-dimensionality-reduction-for-visualization); the temporal clustering described in [Section 4.4](#44-temporal-clustering) uses the full feature set, since collinearity among features is less consequential for distance-based clustering methods like $k$-means than it is for interpreting a low-dimensional projection.

### 4.3 Dimensionality Reduction for Visualization

To visualize the structure of the feature space described above, Uniform Manifold Approximation and Projection (UMAP) [[McInnes et al., 2018]](#references) was applied to the normalized feature matrix. UMAP was chosen over alternatives such as PCA or $t$-SNE for its ability to preserve both local neighborhood structure and, to a reasonable extent, global relationships between clusters, while remaining computationally efficient on the dataset size. A neighborhood size of $n_{\text{neighbors}} = 15$ and minimum distance of $0.1$ were used, projecting the data into two components for the primary visualization and three components as a check on whether additional structure emerges beyond what is visible in two dimensions.

> **Figure 2.** Two-dimensional UMAP embedding of LEGO sets, colored by time window.
> `Code/outputs/UMAP_embedding.png`

The resulting two-dimensional embedding reveals three to four well-separated clusters, along with a small number of isolated outlier points at the periphery. This separation indicates that the feature set captures genuinely distinct regimes of LEGO sets, likely corresponding to differences in theme, set type, or piece composition, that do not blend continuously into one another.

Notably, time window color is thoroughly mixed within each cluster rather than segregated: early and late sets of a given type occupy the same region of the embedding. This suggests that time of release is not the dominant axis separating set types, since a strong time effect would appear as solid-colored blobs rather than mixed ones. A weaker secondary time trend is still visible, however: later sets tend to concentrate toward the outer edges of each cluster, while earlier sets sit closer to the center, implying that set characteristics drift gradually over time within a given type even though type itself is the primary driver of cluster membership.

The embedding also exhibits several thin tendrils connecting or extending from the main clusters. These likely correspond to sets that are transitional between two categories, or to sparsely represented set variants such as reissues. Identifying which sets populate these bridging regions is a natural next step for validating the semantic meaning of the clusters.

To confirm that two dimensions are sufficient to capture the dominant structure, the embedding was repeated with three components and inspected from multiple viewing angles (see [Appendix D](#74-additional-umap-visualizations)). The three-dimensional projection reproduces the same three-cluster structure and the same within-cluster mixing of time windows observed in two dimensions, with no additional cluster or separation emerging along the third axis. This indicates that the two-dimensional embedding is not discarding meaningful structure, and that two components are adequate for downstream visualization and interpretation.

### 4.4 Temporal Clustering

The optimal number of clusters $k$ is selected using the elbow method and silhouette score analysis on the second-to-last cohort rather than the most recent one. The final cohort in the dataset does not correspond to a complete year, since it captures only the partial set of releases available at the time of data collection, whereas the second-to-last cohort corresponds to 2025, the most recent complete year of releases. Selecting $k$ from a full year avoids anchoring the clustering solution to an incomplete and potentially unrepresentative sample. Cluster fitting itself proceeds backward in time, beginning from this reference cohort and moving through earlier cohorts, so that each earlier cohort's structure is interpreted relative to present-day design conventions rather than relative to the earliest sets in the catalog. Drift, by contrast, is calculated moving forward through time: $\Delta_t$ measures the change from cohort $t$ to cohort $t+1$, so that detected drift points $t^*$ correspond to the year in which a shift occurred, consistent with the historical framing in [Section 2](#2-problem-statement).

To address any concern that standard $k$-means clustering applied independently per cohort may produce inconsistent labeling (the "cluster correspondence problem"), centroids were matched across consecutive years using the Hungarian algorithm to minimize total centroid displacement [[Kuhn, 1955]](#references).

### 4.5 Robustness Check via DBSCAN

As a robustness check on the $k$-means-based temporal clustering, DBSCAN [[Ester et al., 1996]](#references) was additionally fit to each cohort $\mathcal{C}_t$. Unlike $k$-means, DBSCAN does not require a fixed number of clusters to be specified in advance and can identify clusters of arbitrary shape, while explicitly labeling low-density points as noise rather than forcing them into the nearest cluster. This makes it a useful check on two assumptions underlying the primary $k$-means approach: first, that the fixed value of $k$ selected from the 2025 reference cohort remains an appropriate choice across earlier cohorts with potentially different underlying structure, and second, that LEGO sets naturally form roughly spherical, similarly sized groups in feature space — an assumption implicit in $k$-means but not required by DBSCAN. Rather than treating DBSCAN as a competing model to be scored against $k$-means, its purpose here is diagnostic: if DBSCAN, run independently and without the constraint of a fixed $k$, recovers a broadly similar number and shape of clusters within each cohort, this lends confidence that the cluster structure detected by $k$-means reflects genuine groupings in the data rather than an artifact of its shape assumptions. If DBSCAN instead reveals substantially different structure, particularly within cohorts flagged as drift points, this would signal that conclusions drawn from the $k$-means clustering warrant additional caution rather than invalidating them outright.

### 4.6 Drift Detection

The dissimilarity between consecutive clustering solutions was measured using two metrics:

1. **Adjusted Rand Index (ARI):** For sets present in both cohorts $\mathcal{C}_t$ and $\mathcal{C}_{t+1}$ (overlapping years), ARI measures the agreement between the two clustering assignments. A sharp drop in ARI signals structural reorganization [[Mishra & Stamp, 2025]](#references).
2. **Centroid displacement:** The average Euclidean distance between matched centroid pairs $\|\mu_j^{(t+1)} - \mu_j^{(t)}\|_2$, averaged over all $k$ clusters, captures smooth directional drift even when set membership does not overlap between cohorts.

### 4.7 Reproducibility

All code has been written in Python and made available alongside the SQLite database construction scripts. Random seeds were fixed at 42 (the meaning of life, the universe, and everything) to ensure reproducible random cluster initializations.

## 5. Results

### 5.1 Optimal Number of Clusters

One of the key aspects of comparison in this project is determining the number of clusters to use in the temporal clustering. A minimum number of clusters larger than 2 was defined, and cluster comparison was run on the second-to-last epoch. Silhouette scores [[Rousseeuw, 1987]](#references) and the elbow method were compared to determine the best cluster count to use throughout.

> **Figure 3.** Elbow method (inertia, blue) and silhouette score (green) across $k = 4$–$15$. Inertia shows a steep drop through $k=6$ before flattening, with a secondary bend around $k=8$. Silhouette score peaks at $k=4$ (0.51) and declines steadily thereafter, confirming $k=4$ as the optimal number of clusters.
> `Code/outputs/combined_plt.png`

### 5.2 Temporal Clustering

> **Figure 4.** Mean centroid drift across consecutive clustering runs, with the gray dotted line marking the mean drift ($\mu = 11.71$) and the red dashed line marking the drift-point threshold ($\mu + 1\sigma = 18.23$). Nine transitions (red points) exceed this threshold and are flagged as significant drift events ($t^*$), concentrated in the early-1980s to early-1990s period and recurring more frequently from the early 2000s through 2025.
> `Code/outputs/centroid_drift_threshold.png`

The mean and standard deviation of the drift values were computed as $\mu_\delta = 11.709$ and $\sigma_\delta = 6.519$, yielding a drift-point threshold of $\mu_\delta + 1\sigma_\delta = 18.228$ (dashed red line, Figure 4). Applying this threshold flagged 9 of the 43 transitions as significant drift points $t^*$, summarized in Table 4 and shown as red markers in Figure 4. Full drift metrics for every transition, including mean, max, and total centroid displacement, are reported in Table 3 ([Appendix E](#75-temporal-drift-results)).

Several of the flagged drift points align closely with the historical milestones hypothesized in [Section 2](#2-problem-statement). Most notably, the 2003 and 2004 windows both register drift values well above threshold (22.328 and 20.273, respectively), consistent with the well-documented near-bankruptcy and subsequent restructuring the company underwent during this period. Similarly, the cluster of drift points in the early 2020s (2022, 2023, and 2025) corresponds to the period of rapid growth in adult-targeted product lines discussed in [Section 2](#2-problem-statement), and the 2007–2008 drift points plausibly reflect the broader product-line consolidation that followed LEGO's earlier financial difficulties as the company diversified its licensed offerings. Not every hypothesized milestone produced a detectable drift point, however: the introduction of the Star Wars license in 1999 does not appear among the flagged transitions, suggesting that this change, while commercially significant, may not have produced a large enough shift in the structural features used here (part composition, minifigure ratios, color and category entropy) to register as statistical drift, or that its effects were absorbed gradually across several neighboring cohorts rather than concentrated in a single transition.

Two additional drift points, 1982–1984 and 1992–1993, do not correspond to any milestone identified a priori in [Section 2](#2-problem-statement). Both occur in relatively early, lower-volume windows (137 and 179 sets respectively, per Table 5), where smaller cohort sizes can produce noisier centroid estimates and inflate apparent drift independent of any genuine structural shift. This suggests that some flagged points, particularly in the earlier decades of the catalog, may partly reflect sampling variability rather than a true change in set design philosophy — a limitation worth weighing alongside the more clearly historically-grounded drift points identified in the 2000s and 2020s. Overall, the drift detection approach successfully recovers several of the major inflection points in LEGO's product history without using any label information, supporting the central hypothesis that structural, label-free features can surface meaningful shifts in product design over time.

**Table 4. Flagged drift events exceeding the threshold $\mu_\delta + 1\sigma_\delta = 18.228$, ranked by drift magnitude.**

| Run | Drift | Year Window |
|---|---:|---|
| 40 | 23.187 | 2023 |
| 25 | 22.969 | 2008 |
| 24 | 22.781 | 2007 |
| 20 | 22.328 | 2003 |
| 39 | 21.716 | 2022 |
| 42 | 21.036 | 2025 |
| 10 | 20.489 | 1992–1993 |
| 21 | 20.273 | 2004 |
| 5 | 19.203 | 1982–1984 |

### 5.3 DBSCAN Robustness and External Validation

To assess whether the $k$-means-based clustering structure reflects genuine groupings in the data rather than an artifact of its spherical-cluster assumption, DBSCAN was independently fit to each cohort using the $\epsilon$ value determined in [Appendix F](#76-epsilon-determination-for-dbscan). Agreement between the two methods was measured using Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI) on the overlapping set membership between cohorts, along with the fraction of points DBSCAN labeled as noise. Full results are reported in Table 6 ([Appendix G](#77-dbscan-robustness-results)).

Agreement between the two clustering methods varies substantially across cohorts. Several windows show strong correspondence: window 4 (1980–1981) achieves an ARI of 0.981 and NMI of 0.874, and windows 0, 14, and 17 all exceed ARI $= 0.7$, indicating that in these periods DBSCAN, run without any constraint on cluster count, recovers essentially the same grouping structure as $k$-means. At the other extreme, several windows show near-zero agreement, most notably window 42 (2025, ARI $= 0.003$, NMI $= 0.004$), along with windows 15, 18, 19, and 32, all below ARI $= 0.06$.

Comparing agreement scores at the drift points identified in [Section 5.2](#52-temporal-clustering) against all other windows shows a modest but consistent pattern: the average ARI across the nine flagged drift windows is 0.266, compared to 0.303 across the remaining windows. This is broadly consistent with the interpretation that periods of genuine structural reorganization are harder for both clustering methods to agree on, since the underlying data itself is less stable during these transitions. However, this difference is small relative to the spread within each group, and window 42 alone, an extreme outlier at ARI $= 0.003$, accounts for a meaningful share of the gap; excluding it, the average ARI at the remaining eight drift windows rises to 0.296, nearly identical to the non-drift average. This suggests the drift-agreement relationship should be treated as a mild, suggestive trend rather than strong confirmatory evidence, and that window 42 in particular, the most recent complete cohort and the one used as the $k$ reference cohort in [Section 4.4](#44-temporal-clustering), may warrant separate investigation as a special case.

Noise fractions, meanwhile, do not show an obvious relationship to drift status: the highest noise fraction in the dataset (0.210, window 14) occurs at a non-drift cohort, while several drift-flagged windows (e.g., window 24 at 0.048, window 42 at 0.008) show low noise. This indicates that DBSCAN's tendency to label points as noise is driven more by local density variation within a given cohort's feature space than by the magnitude of drift from the previous period.

Overall, while agreement between $k$-means and DBSCAN is far from uniform across the 44 cohorts, the presence of several very high-agreement windows indicates that the underlying cluster structure detected by $k$-means is not purely an artifact of its assumptions, and the weak tendency toward lower agreement at drift points offers modest additional support for those transitions representing genuine structural change rather than clustering noise.

### 5.4 Limitations and Future Work

Several limitations of this analysis are worth noting. First, cohort size varies considerably across the dataset's history (Table 5), and smaller early-era cohorts produce noisier centroid estimates; this likely explains why two of the nine flagged drift points (1982–1984 and 1992–1993) do not correspond to any hypothesized historical milestone, as discussed in [Section 5.2](#52-temporal-clustering). Second, the DBSCAN robustness check in [Section 5.3](#53-dbscan-robustness-and-external-validation) used a single $\epsilon$ value fit across the entire dataset ([Appendix F](#76-epsilon-determination-for-dbscan)); since feature density plausibly varies across eras as LEGO's catalog diversified, a fixed $\epsilon$ may be better suited to some cohorts than others, and a per-cohort $\epsilon$ selection could be explored in future work. Third, theme labels were used only for external validation and not as clustering features; this means the analysis is well-suited to detecting broad structural shifts but may miss more subtle drift occurring *within* a single theme over time. Finally, the drift-point threshold $\mu_\delta + \sigma_\delta$ is a simple heuristic rather than a formally validated statistical test; future work could apply a more rigorous change-point detection framework, or bootstrap confidence intervals around $\Delta_t$, to better distinguish genuine structural drift from cohort-size-driven noise, particularly in the smaller early-era windows.

## 6. Conclusion

This project set out to determine whether unsupervised machine learning could detect meaningful structural shifts in LEGO set design over time, and whether those shifts align with known historical events. Using temporal $k$-means clustering with Hungarian-algorithm centroid matching and a threshold-based drift detection scheme, nine drift points were identified across 43 cohort transitions spanning 1955 through 2026. Several of these, most notably the 2003–2004 transitions and the 2022–2025 cluster of drift points, align closely with LEGO's well-documented near-bankruptcy and its more recent expansion into adult-targeted product lines, while others (1982–1984, 1992–1993) appear more likely to reflect small-cohort noise than genuine structural change. A DBSCAN robustness check found broadly consistent cluster structure with $k$-means in several cohorts, alongside a weak tendency toward lower agreement at flagged drift points, offering modest additional support for the drift detection results. Not every hypothesized milestone, including the 1999 introduction of the Star Wars license, produced a detectable drift point, suggesting that some historically significant changes may not manifest as large discontinuities in the structural features used here. Taken together, these results provide partial but genuine support for the hypothesis that label-free structural features can surface meaningful shifts in LEGO's product design over time, while also highlighting the limitations of a purely threshold-based approach to drift detection on unevenly sized cohorts.

LEGO has long been a passion of mine from a very young age. While my appreciation for the brand and the product began as a child, it has remained a hobby well into adulthood. My growth reflects LEGO's growth. When I was younger, I thought that LEGO Star Wars was the greatest thing on earth. While I still love those sets, my passions have shifted towards the large-scale car models like the Formula 1 cars, Porsche 911, Land Rover Defender, as well as the new LEGO Lord of the Rings collection.

This project gives me a great opportunity to combine a personal passion for LEGO with the machine learning techniques studied in this course. By applying clustering and drift detection methods to decades of LEGO set data, I hope to uncover patterns in the evolution of the brand and show how the composition of the product design which has made this iconic brand so successful reveals shifts not obvious to the human eye.

## 7. Appendix

### 7.1 Rebrickable Entity Relationship Diagram

> **Figure A1.** Database schema for the Rebrickable [[Rebrickable, 2024]](#references) LEGO dataset. The relational structure links LEGO sets to inventories, parts, colors, themes, and minifigures, providing the foundation for feature engineering and longitudinal analysis of changes in LEGO product design over time.
> `images/ERD.png`

### 7.2 Breakdown of Number of Sets per Time Window

**Table 5. Number of Sets per Time Window**

| Window | num sets | min | max | | Window | num sets | min | max |
|---|---:|---|---|---|---|---:|---|---|
| 0 | 56 | 1955 | 1966 | | 22 | 227 | 2005 | 2005 |
| 1 | 131 | 1967 | 1973 | | 23 | 216 | 2006 | 2006 |
| 2 | 110 | 1974 | 1976 | | 24 | 209 | 2007 | 2007 |
| 3 | 149 | 1977 | 1979 | | 25 | 238 | 2008 | 2008 |
| 4 | 101 | 1980 | 1981 | | 26 | 272 | 2009 | 2009 |
| 5 | 137 | 1982 | 1984 | | 27 | 252 | 2010 | 2010 |
| 6 | 104 | 1985 | 1985 | | 28 | 311 | 2011 | 2011 |
| 7 | 219 | 1986 | 1987 | | 29 | 358 | 2012 | 2012 |
| 8 | 134 | 1988 | 1989 | | 30 | 362 | 2013 | 2013 |
| 9 | 178 | 1990 | 1991 | | 31 | 423 | 2014 | 2014 |
| 10 | 179 | 1992 | 1993 | | 32 | 471 | 2015 | 2015 |
| 11 | 105 | 1994 | 1994 | | 33 | 496 | 2016 | 2016 |
| 12 | 130 | 1995 | 1995 | | 34 | 475 | 2017 | 2017 |
| 13 | 139 | 1996 | 1996 | | 35 | 487 | 2018 | 2018 |
| 14 | 181 | 1997 | 1997 | | 36 | 479 | 2019 | 2019 |
| 15 | 276 | 1998 | 1998 | | 37 | 496 | 2020 | 2020 |
| 16 | 230 | 1999 | 1999 | | 38 | 521 | 2021 | 2021 |
| 17 | 265 | 2000 | 2000 | | 39 | 508 | 2022 | 2022 |
| 18 | 232 | 2001 | 2001 | | 40 | 532 | 2023 | 2023 |
| 19 | 250 | 2002 | 2002 | | 41 | 610 | 2024 | 2024 |
| 20 | 269 | 2003 | 2003 | | 42 | 640 | 2025 | 2025 |
| 21 | 250 | 2004 | 2004 | | 43 | 452 | 2026 | 2026 |

*Number of sets per time window. Each window aggregates one or more consecutive release years to balance sample sizes across the dataset's history; **Window** is the window index, **num sets** is the count of LEGO sets falling in that window, and **min**/**max** give the first and last release year included in the window.*

### 7.3 Cluster Determination Charts

This appendix presents supplementary cluster determination graphs that support the discussion in [Section 5.1](#51-optimal-number-of-clusters).

> **Figure A2.** Elbow method plot of within-cluster sum of squares (inertia) vs. number of clusters ($k$). Inertia decreases sharply from $k=4$ to $k=6$, then flattens out beyond $k=8$, marking the elbow point — the $k$ value beyond which additional clusters yield diminishing returns.
> `Code/outputs/elbow_plt.png`

> **Figure A3.** Silhouette score across $k=4$–$15$. Score peaks sharply at $k=4$ ($\approx 0.51$), then drops and stays consistently lower for all higher $k$ values, confirming $k=4$ as the best choice for cluster separation and cohesion.
> `Code/outputs/silhouette_plt.png`

### 7.4 Additional UMAP Visualizations

This appendix presents supplementary UMAP visualizations that support the discussion in [Section 4.3](#43-dimensionality-reduction-for-visualization) (dimensionality reduction for visualization).

> **Figure A4.** Two-dimensional UMAP embedding faceted by time window. In each panel, points from that window are highlighted in color and all other points are shown in gray for reference.
> `Code/outputs/UMAP_grid_by_window.png`

Figure A4 shows the same two-dimensional embedding as Figure 2, but faceted by individual time window rather than using a single continuous color scale. In each panel, points belonging to that window are highlighted in color while all other points are shown in gray for reference. This view makes the temporal drift within clusters easier to inspect than a single overlaid plot. In the earliest windows (0 through roughly 5), highlighted points are concentrated almost entirely in a small region of the upper cluster, indicating that sets released in this period are compositionally similar to one another and represent a narrow slice of the overall feature space. Moving through the middle windows, the highlighted region expands and begins to populate the lower cluster as well, suggesting the introduction of new set types over time rather than a simple continuation of earlier patterns. By the later windows (roughly 30 onward), highlighted points spread across nearly the full extent of both major clusters, indicating that recent sets span a much broader range of the feature space than earlier ones. This progressive broadening is consistent with the interpretation in [Section 4.3](#43-dimensionality-reduction-for-visualization) that set diversity, in terms of scale, composition, and minifigure content, has increased over time, even though time window itself is not the primary axis separating the clusters.

> **Figure A5.** Three-dimensional UMAP embedding of LEGO sets viewed from multiple angles, colored by time window.
> `Code/outputs/UMAP_embedding_3d_multiangle.png`

Figure A5 shows the three-dimensional UMAP embedding from multiple viewing angles, used to confirm that no additional cluster structure is hidden along a third dimension beyond what is visible in the two-dimensional embedding.

### 7.5 Temporal Drift Results

**Table 3. Drift metrics between consecutive clustering runs, mapped to their corresponding year windows.**

| From Run | To Run | Mean Drift | Max Drift | Total Drift | Year Window |
|---:|---:|---:|---:|---:|---|
| 0 | 1 | 5.135 | 12.952 | 20.540 | 1967–1973 |
| 1 | 2 | 6.172 | 12.958 | 24.688 | 1974–1976 |
| 2 | 3 | 5.410 | 9.854 | 21.641 | 1977–1979 |
| 3 | 4 | 8.302 | 21.209 | 33.208 | 1980–1981 |
| 4 | 5 | 19.203 | 52.971 | 76.811 | 1982–1984 |
| 5 | 6 | 17.808 | 37.006 | 71.233 | 1985 |
| 6 | 7 | 9.207 | 15.759 | 36.826 | 1986–1987 |
| 7 | 8 | 15.606 | 28.349 | 62.425 | 1988–1989 |
| 8 | 9 | 14.257 | 38.044 | 57.030 | 1990–1991 |
| 9 | 10 | 20.489 | 49.974 | 81.955 | 1992–1993 |
| 10 | 11 | 10.417 | 30.963 | 41.669 | 1994 |
| 11 | 12 | 11.582 | 34.168 | 46.330 | 1995 |
| 12 | 13 | 11.245 | 31.156 | 44.980 | 1996 |
| 13 | 14 | 11.889 | 21.045 | 47.557 | 1997 |
| 14 | 15 | 3.587 | 10.689 | 14.348 | 1998 |
| 15 | 16 | 4.681 | 13.839 | 18.724 | 1999 |
| 16 | 17 | 7.912 | 27.249 | 31.650 | 2000 |
| 17 | 18 | 4.699 | 13.399 | 18.795 | 2001 |
| 18 | 19 | 1.859 | 2.795 | 7.435 | 2002 |
| 19 | 20 | 22.328 | 80.952 | 89.312 | 2003 |
| 20 | 21 | 20.273 | 77.618 | 81.093 | 2004 |
| 21 | 22 | 10.282 | 37.534 | 41.127 | 2005 |
| 22 | 23 | 3.627 | 11.682 | 14.508 | 2006 |
| 23 | 24 | 22.781 | 48.977 | 91.124 | 2007 |
| 24 | 25 | 22.969 | 83.050 | 91.877 | 2008 |
| 25 | 26 | 8.381 | 20.134 | 33.524 | 2009 |
| 26 | 27 | 15.965 | 52.724 | 63.860 | 2010 |
| 27 | 28 | 16.586 | 58.092 | 66.342 | 2011 |
| 28 | 29 | 8.847 | 27.972 | 35.387 | 2012 |
| 29 | 30 | 3.658 | 11.171 | 14.633 | 2013 |
| 30 | 31 | 3.496 | 6.160 | 13.983 | 2014 |
| 31 | 32 | 5.154 | 14.538 | 20.615 | 2015 |
| 32 | 33 | 7.632 | 24.946 | 30.529 | 2016 |
| 33 | 34 | 3.247 | 5.269 | 12.989 | 2017 |
| 34 | 35 | 9.139 | 31.289 | 36.556 | 2018 |
| 35 | 36 | 11.484 | 40.367 | 45.937 | 2019 |
| 36 | 37 | 15.454 | 40.260 | 61.817 | 2020 |
| 37 | 38 | 7.667 | 18.707 | 30.668 | 2021 |
| 38 | 39 | 21.716 | 78.391 | 86.863 | 2022 |
| 39 | 40 | 23.187 | 78.440 | 92.749 | 2023 |
| 40 | 41 | 18.179 | 65.438 | 72.717 | 2024 |
| 41 | 42 | 21.036 | 65.434 | 84.143 | 2025 |
| 42 | 43 | 10.940 | 38.842 | 43.759 | 2026 |

The mean and standard deviation of the drift values were computed as $\mu_\delta = 11.709$ and $\sigma_\delta = 6.519$, respectively, yielding a threshold of $\mu_\delta + 1\sigma_\delta = 18.228$. Using this threshold, 9 out of 43 transitions were flagged as significant drift events ($t^*$) — see Table 4 in [Section 5.2](#52-temporal-clustering).

### 7.6 Epsilon Determination for DBSCAN

This appendix presents supplementary graphs and discussion of the selection of the $\epsilon$ value used to tune the DBSCAN model used for a robustness check.

> **Figure A6.** K-distance plot (24-NN distances, sorted ascending) used for DBSCAN eps selection. The curve remains relatively flat and gradually increasing up to approximately point 600, followed by a sharp upward inflection — the "knee" — indicating the optimal eps value of 5.2014.
> `Code/outputs/k_distance_plt.png`

The k-distance plot is a standard heuristic for selecting the `eps` parameter in DBSCAN, where $k$ corresponds to `min_samples` (here, the 24th nearest neighbor distance for each point). By sorting these distances in ascending order and plotting them, the resulting curve reveals the density structure of the dataset: points in dense regions have small $k$-NN distances and cluster together in the flat portion of the curve, while points in sparse regions or outliers have much larger distances and appear in the steeply rising tail. In this plot, the distances stay below roughly 5 units for the first ~600 points, indicating that the majority of the dataset lies in reasonably dense, well-connected regions.

The key feature to identify is the "knee" or "elbow" of the curve — the point where the distance values transition from slow, gradual growth to a sharp, near-vertical increase. This inflection marks the natural boundary between core/border points (which DBSCAN should treat as part of clusters) and noise points or extreme outliers (which lie much farther from their neighbors). Here, that knee occurs around point 600–610, corresponding to a 24-NN distance of approximately 5.2, which was selected as the `eps` value. Choosing `eps` at this inflection balances two failure modes: setting `eps` too low (which fragments the data into excessive small clusters and/or classifies too many points as noise), or too high (which merges distinct clusters together into a single large one). The steep tail beyond the knee — jumping from ~5 to over 25 in the last few dozen points — reflects a small number of increasingly isolated points that are appropriately treated as noise under the chosen `eps`.

### 7.7 DBSCAN Robustness Results

**Table 6. Clustering evaluation metrics (ARI, NMI, and Noise Fraction) per index.**

| Index | ARI | NMI | Noise | | Index | ARI | NMI | Noise |
|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 0 | 0.759 | 0.732 | 0.125 | | 22 | 0.243 | 0.298 | 0.097 |
| 1 | 0.294 | 0.390 | 0.046 | | 23 | 0.298 | 0.438 | 0.093 |
| 2 | 0.246 | 0.276 | 0.045 | | 24 | 0.219 | 0.196 | 0.048 |
| 3 | 0.141 | 0.185 | 0.034 | | 25 | 0.200 | 0.191 | 0.067 |
| 4 | 0.981 | 0.874 | 0.099 | | 26 | 0.306 | 0.332 | 0.077 |
| 5 | 0.433 | 0.517 | 0.109 | | 27 | 0.721 | 0.549 | 0.083 |
| 6 | 0.616 | 0.420 | 0.144 | | 28 | 0.329 | 0.312 | 0.077 |
| 7 | 0.188 | 0.274 | 0.041 | | 29 | 0.211 | 0.219 | 0.045 |
| 8 | 0.335 | 0.187 | 0.164 | | 30 | 0.264 | 0.245 | 0.030 |
| 9 | 0.220 | 0.140 | 0.112 | | 31 | 0.061 | 0.094 | 0.021 |
| 10 | 0.246 | 0.190 | 0.061 | | 32 | 0.037 | 0.033 | 0.017 |
| 11 | 0.643 | 0.441 | 0.114 | | 33 | 0.320 | 0.243 | 0.056 |
| 12 | 0.135 | 0.182 | 0.038 | | 34 | 0.130 | 0.112 | 0.025 |
| 13 | 0.084 | 0.151 | 0.050 | | 35 | 0.101 | 0.055 | 0.027 |
| 14 | 0.700 | 0.542 | 0.210 | | 36 | 0.129 | 0.092 | 0.050 |
| 15 | 0.030 | 0.085 | 0.011 | | 37 | 0.343 | 0.230 | 0.083 |
| 16 | 0.269 | 0.264 | 0.083 | | 38 | 0.317 | 0.216 | 0.035 |
| 17 | 0.725 | 0.650 | 0.109 | | 39 | 0.350 | 0.185 | 0.045 |
| 18 | 0.043 | 0.043 | 0.043 | | 40 | 0.249 | 0.152 | 0.028 |
| 19 | 0.031 | 0.062 | 0.056 | | 41 | 0.189 | 0.236 | 0.031 |
| 20 | 0.299 | 0.378 | 0.097 | | 42 | 0.003 | 0.004 | 0.008 |
| 21 | 0.397 | 0.449 | 0.152 | | 43 | 0.159 | 0.125 | 0.033 |

*The index column in the above table corresponds to the window value in [Appendix B](#72-breakdown-of-number-of-sets-per-time-window).*

## References

- Chi, Y., Song, X., Zhou, D., Hino, K., & Tseng, B. L. (2007). Evolutionary spectral clustering by incorporating temporal smoothness. In *Proceedings of the 13th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 153–162). https://doi.org/10.1145/1281192.1281212
- Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. In *Proceedings of the Second International Conference on Knowledge Discovery and Data Mining (KDD-96)* (pp. 226–231). AAAI Press.
- Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M., & Bouchachia, A. (2014). A survey on concept drift adaptation. *ACM Computing Surveys*, 46(4), 1–37. https://doi.org/10.1145/2523813
- Kuhn, H. W. (1955). The Hungarian method for the assignment problem. *Naval Research Logistics Quarterly*, 2(1–2), 83–97. https://doi.org/10.1002/nav.3800020109
- McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform manifold approximation and projection for dimension reduction. *arXiv preprint arXiv:1802.03426*. https://arxiv.org/abs/1802.03426
- Mishra, A., & Stamp, M. (2025). Cluster analysis and concept drift detection in malware. *arXiv preprint arXiv:2502.14135*. https://arxiv.org/abs/2502.14135
- Rebrickable. (2024). LEGO sets database downloads. https://rebrickable.com/downloads/
- Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics*, 20, 53–65. https://doi.org/10.1016/0377-0427(87)90125-7