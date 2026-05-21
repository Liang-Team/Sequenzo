## Page 1

&lt;img&gt;International Journal of Social Research Methodology cover image&lt;/img&gt;
Routledge
Taylor & Francis Group

# International Journal of Social Research Methodology

ISSN: 1364-5579 (Print) 1464-5300 (Online) Journal homepage: www.tandfonline.com/journals/tsrm20

# Assessing the impact of timing errors in sequence analysis

## Gilbert Ritschard & Tim F. Liao

To cite this article: Gilbert Ritschard & Tim F. Liao (08 May 2026): Assessing the impact of timing errors in sequence analysis, International Journal of Social Research Methodology, DOI: 10.1080/13645579.2026.2666297

To link to this article: https://doi.org/10.1080/13645579.2026.2666297

© 2026 The Author(s). Published by Informa UK Limited, trading as Taylor & Francis Group.

View supplementary material

Published online: 08 May 2026.

Submit your article to this journal

Article views: 141

View related articles

View Crossmark data

Full Terms & Conditions of access and use can be found at https://www.tandfonline.com/action/journalInformation?journalCode=tsrm20

---


## Page 2

INTERNATIONAL JOURNAL OF SOCIAL RESEARCH METHODOLOGY
https://doi.org/10.1080/13645579.2026.2666297

&lt;img&gt;Routledge Taylor & Francis Group&lt;/img&gt;

RESEARCH ARTICLE
OPEN ACCESS &lt;img&gt;Check for updates&lt;/img&gt;

# Assessing the impact of timing errors in sequence analysis

Gilbert Ritschard &lt;img&gt;ORCID&lt;/img&gt; and Tim F. Liao &lt;img&gt;ORCID&lt;/img&gt;

<sup>a</sup>Institute of Demography and Socioeconomics, University of Geneva, Geneva, Switzerland; <sup>b</sup>Department of Sociology, University of Illinois Urbana-Champaign, IL, USA

**ABSTRACT**
The standard practice in sequence analysis is to regard sequences as exact and measured without error. Transition timing is a common source of sequence measurement error in social research. In this paper, we propose a method for conducting Monte Carlo simulations of transition timing errors to assess the robustness of sequence analysis outcomes against such timing errors. The method consists in repeating multiple times the analyses with updated sequence datasets. We consider three models for performing the simulations, depending on the assumptions of whether a spell can be suppressed or whether time to subsequent transitions must be kept. Using a sample of 2,000 family life sequences and reanalysing a published example of sequence cluster analysis, we demonstrate, by means of different summarising principles, how transition timing uncertainty may affect sequence analytic results and thus conclusions reached.

**ARTICLE HISTORY**
Received 28 November 2025
Accepted 22 April 2026

**KEYWORDS**
Sequence analysis; timing error; telescoping effect; uncertainty; Monte Carlo simulation; standard error

## 1. Introduction

Sequence analysis (SA) in social research aims to characterize sets of sequences of states (e.g. working statuses, cohabitation types, activities) at successive time points. SA has become popular in many social sciences such as life course and time use analyses concerned with ordering, timing, and duration of events. SA typically starts with computing distances between sequences, and further analyses these pairwise distances for building typologies, computing multidimensional scaling (MDS) scores to sort and represent sequences numerically, identifying medoids or other representative sequences, and comparing groups (Abbott & Tsay, 2000; Liao et al., 2022; Piccarreta & Studer, 2019). In the twenty-first century, SA has gained a tremendous momentum in not just life course research but other social sciences, showing its importance in a broad range of social research (Liao et al., 2022), and it has seen about over 20 dissimilarity measures developed and applied, including some optimal matching (OM) variants (Studer & Ritschard, 2016).

In this paper, we propose a method for assessing the robustness of dissimilarity-based sequence analysis (SA) outcomes in the face of timing errors.

CONTACT Gilbert Ritschard &lt;img&gt;Envelope icon&lt;/img&gt; gilbert.ritschard@unige.ch &lt;img&gt;Building icon&lt;/img&gt; Institute of Demography and Socioeconomics, University of Geneva, Bd du Pont d'Arve 40, Geneva CH-1205, Switzerland
&lt;img&gt;Download icon&lt;/img&gt; Supplemental data for this article can be accessed online at https://doi.org/10.1080/13645579.2026.2666297
© 2026 The Author(s). Published by Informa UK Limited, trading as Taylor & Francis Group.
This is an Open Access article distributed under the terms of the Creative Commons Attribution License (http://creativecommons.org/licenses/by/4.0/), which permits unrestricted use, distribution, and reproduction in any medium, provided the original work is properly cited. The terms on which this article has been published allow the posting of the Accepted Manuscript in a repository by the author(s) or with their consent.

---


## Page 3

&lt;page_number&gt;2&lt;/page_number&gt; G. RITSCHARD AND T. F. LIAO

In all SA applications of OM or other distance measures, the measurement of state sequences is always treated as exact or as given by the data. This may not be a problem in genetic research because DNA, RNA, and protein sequences can oftentimes reasonably be considered as accurately measured. In the social sciences, however, considering sequence or trajectory information as accurate is a strong assumption because this information is often based on registrations or surveys that are prone to human recording (or recall) errors, especially when based on social surveys that include a life history calendar, a common source for collecting life course trajectory data (e.g. Drasch & Matthes, 2013). Similar errors can indeed affect the accuracy of any social research relying on self-reporting.

*The general issue of timing errors in social survey research.* Timing errors affect all social surveys beyond data for sequence analysis. Seam and telescoping effects represent two common manifestations of timing-related measurement error in social surveys, each arising from different cognitive and survey-design processes. Seam effects – often observed in panel surveys – occur when reports of change cluster artificially at the boundary (‘seam’) between waves, largely due to recall difficulties and the structure of dependent interviewing (see, e.g., Engstrom & Sinibaldi, 2024; Jäckle & Eckman, 2020). Telescoping, by contrast, refers to recall-based timing errors in which events are reported as having occurred more recently (forward telescoping) or further in the past (backward telescoping) than they actually did. Studies across different fields – including program participation (Celahy et al., 2024), food consumption (Abate et al., 2022), and online panel recall tasks (Rettig & Struminskaya, 2023) – consistently find that respondents struggle to accurately place events in time, leading to systematic distortions in measurement. Together, seam effects in panel contexts and telescoping in recall-based surveys illustrate how temporal misplacement is a pervasive challenge in collecting accurate longitudinal data.

*Extensity of telescoping effects in empirical research.* Telescoping, which may also occur in event history calendars (a common source of sequence data), can substantially distort retrospective reports of when behaviors first occurred, and this pattern is particularly pronounced in studies relying on self-reported smoking onset. Bright and Soulakova’s (2014) research on regular smoking onset age provides clear empirical evidence of extensive telescoping, showing that adult respondents tend to report a later age of onset than what earlier survey waves or historical data indicate. The study demonstrates both forward and backward telescoping effects: About 31.6% of respondents forwardly shifted the smoking onset age, while 31.8% of them backwardly shifted such age, with a mean discrepancy of about 2.7 years in both directions. Their research established the importance of dealing with and correcting such timing errors in social survey research.

When sequences are subject to timing errors, SA outcomes become uncertain. For example, consider two ideal-type sequences: ‘working full time’ represented by FFFFF and ‘being unemployed’ represented by UUUUU. Suppose we want to assign observed sequences to their closest ideal-type by the majority rule. Sequence FFFUU would be assigned to the full time ideal-type but if, because of a timing error, the sequence is reported as FFUUU, it would be assigned to the other ideal-type.

---


## Page 4

<header>INTERNATIONAL JOURNAL OF SOCIAL RESEARCH METHODOLOGY</header> &lt;page_number&gt;3&lt;/page_number&gt;

Our aim, then, is to address this uncertainty in SA in the social sciences by proposing a Monte Carlo (MC) simulation method for exploring and demonstrating the sensitivity of SA outcomes to potential timing errors.

Different kinds of uncertainty have been considered in the SA literature, such as sampling uncertainty (e.g. in cluster validation, Studer, 2021, in group comparison, Liao and Fasang, 2020; Studer et al., 2011), uncertainty related to the choice of the dissimilarity measure (Robette & Bry, 2012; Studer & Ritschard, 2016), uncertainty of the clustering method (Preud’homme et al., 2021), uncertainty related to missing data (Emery et al., 2024), and the uncertain nature of outliers (Piccarreta & Struffolino, 2024). Sequence uncertainty in states has also been recognized as an issue in bioinformatics (Becker et al., 2023; Herman et al., 2015; O’Rawe et al., 2015). In these bioinformatic works, uncertainty typically concerns the measurement of states, i.e. the token reported at each position. In contrast to all of the above, we specifically focus on timing measurement errors and propose a general method for assessing their potential impact on any type of SA outcomes including dissimilarities, MDS scores, clusters, group differences, and representative sequences.

The proposed method consists in generating simulated sequence datasets by applying timing errors in the observed sequences. The advantage of modeling explicitly timing errors over simulating sequences with, for example, Markov-based models (Gabadinho & Ritschard, 2016; Huang et al., 2025), is that it allows measuring the effect of timing errors while controlling for all other characteristics of the sequences. To be clear, our objective is to propose a method for exploring the consequences of timing errors, a basic type of common recall or reporting error, and we do not intend to come up with a comprehensive solution for all kinds of measurement errors in sequence data.

Although our MC-process can be seen as a method for validating SA outcomes, it differs from other methods that have been proposed for validating cluster results (Studer, 2021), for example. First, our approach is not limited to cluster results but encompasses any SA outcomes. Second, and more importantly, it differs conceptually. Cluster validation is not concerned with the impact of possible timing errors but of the sampling of the sequences used for the cluster analysis, with the aim of confirming that the identified partition does not result from chance in selecting the sequences.

We structure our paper as follows. Next, we introduce our MC simulation method for simulating altered sequence datasets. In Section 3, we first explain how we compute distance standard errors and address tuning possibilities of the MC process by commenting on how parameter choices may impact standard errors. Through an analysis of 2,000 family life sequences in Section 4 and the data of migration trajectories from a published study in Section 5, we propose various alternatives for evaluating the sensitivity of SA outcomes to potential timing errors. Section 6 discusses limitations and possible extensions of the proposed MC process. In the final section, we provide some summaries, further discuss the advantages and limitations of our approach, and conclude by considering additional necessary future research beyond the scope of the current paper.

The MC process has been implemented in the MCseqReplic R-package distributed on the CRAN. The package relies on TraMineR (Gabadinho et al., 2011) for the computation of distances between sequences. The applications in Sections 4 and 5 were run in R (R

---


## Page 5

&lt;page_number&gt;4&lt;/page_number&gt; &lt;img&gt;icon&lt;/img&gt; G. RITSCHARD AND T. F. LIAO

Core Team, 2024) using WeightedCluster (Studer, 2013) for clustering, aricode (Chiquet et al., 2023) for cluster similarity indices, and TraMineR and TraMineRextras for visualization and group comparison.

## 2. A Monte Carlo simulation and its implementation

A typical MC simulation consists of studying the behavior of random objects or relationships between random objects – sequences in our case – by randomly sampling given distributions (Harrison, 2010; Kroese & Rubinstein, 2011; Kroese et al., 2014). Here, we assume that the observed sequences are the result of a random reporting process. The coding of trajectories – life courses or daily time use, for example – as sequences is subject to essentially two different types of approximation or reporting errors:

(1) Timing error or approximation: for example, transition time gets inaccurately reported such as during a survey.
(2) Wrong or uncertain state: for example, when changes occur within a unit period (e.g. starting a job after a two-month jobless spell when using yearly data) one must arbitrarily select a state to report among the different states experienced during the period.

In this paper, we focus on timing errors and only briefly outline how state uncertainty could be addressed in the conclusion.

To evaluate the uncertainty of SA results with regard to possible timing errors, we adopt an MC approach where we simulate time reporting errors in sequences. We generate R number of MC-simulated sets of sequences, which we call MC-sets, run the SA analysis for each MC-set, and examine how results – dissimilarities and cluster solutions, for example – vary across simulated MC-sets.

In the following section, we propose different models to simulate timing changes in sequences.

### 2.1. Models of transition time alterations

We adopt the following principle: For each sequence, we generate replications by randomly altering the timing of the transitions – state changes – in a sequence. Timing changes are selected independently for all transitions in the sequence and successively applied from left to right. However, with such independent changes, the change in one transition can erase adjacent spells (or episodes). For example, in the sequence PPAMM, with P for ‘living with parents’, A for ‘living alone’, and M for ‘Married’, advancing the second transition (from A to M) erases the spell in A and the sequence becomes PPMMM. This raises the question of whether such suppressions of spells should be allowed or not. In addition, when we expect respondents to fail to remember exact timings but have good memory of the length of time between two events, we may want to preserve the time to the subsequent transitions after a timing change. To handle these issues, we consider three different models (or assumptions) to apply the randomly drawn changes in timing:

---


## Page 6

INTERNATIONAL JOURNAL OF SOCIAL RESEARCH METHODOLOGY &lt;page_number&gt;5&lt;/page_number&gt;

(1) Model ‘indep’, which applies independent timing changes regardless of any possible suppression of spells and changes in the time to subsequent transitions.
(2) Model ‘keep.dss’, which constrains randomly drawn timing changes of transitions to preserve the original sequence of distinct successive states (DSS).
(3) Model ‘relative’, which preserves the time to the subsequent transitions after each timing change. Like ‘indep’, this model can possibly erase spells.

Model ‘indep’ and ‘keep.dss’ assume that a timing error at t does not affect the timing of subsequent state changes, while model ‘relative’ changes the timing of all subsequent transitions after each timing change.

These three simple models for generating errors do not have the pretension to model the behavior of survey respondents but can relatively easily be operationalized. In the absence of more reliable information, using such simple general models to simulate random behaviors within an MC process is sufficient to get an approximate idea of how sequence data could be affected by timing errors. Of course, we could imagine more realistic models (see the extensions discussed in Section 6), but such models would necessarily be more complex and therefore more difficult to implement.

Model ‘indep’ is the most general and will produce the greatest variability between generated timing errors. The two other models restrict the timing changes that can be generated.

We detail the three alternative approaches below and consider for illustration the mini example of seven sequences of length 7 shown in Table 1.

For all three models, the timing change j of each transition is randomly selected in the range [−J, +J] where J is a user-specified integer defining the maximum timing error in units of time. For example, setting J = 1, the timing change j can take the values of −1, 0, or +1. The error j applied to a given transition is randomly drawn using any user-defined discrete probability distribution $P_J = (p_{-J}, \ldots, p_0, \ldots, p_J)$. For example, specifying $P_J = (.25, .5, .25)$, which implicitly sets J = 1, a no-change would get a probability of $p_0 = 0.5$ and would be twice as probable than advancing one position. The length of the $P_J = (p_{-J}, \ldots, p_0, \ldots, p_J)$ vector should be based on substantive considerations, and it must be an odd number. The changes are applied successively from the start (the left-most transition) to the end of the sequence. Delaying the timing of a transition from state A to state B requires increasing the duration of the spell in the state of origin A and reducing the duration of subsequent spells. Advancing a transition requires reducing the duration of

Table 1. Mini example of seven sequences of length 7 in successive state form (STS), and duration-stamped spell form (SPS). Tokens F, P, and U stand for working full time, working part time, and unemployed, respectively.

<table>
  <thead>
    <tr>
      <th></th>
      <th>STS Sequence</th>
      <th>SPS Sequence</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>s1</td>
      <td>PPFUUPP</td>
      <td>P/2-F/1-U/2-P/2</td>
    </tr>
    <tr>
      <td>s2</td>
      <td>FUUUPPP</td>
      <td>F/1-U/3-P/3</td>
    </tr>
    <tr>
      <td>s3</td>
      <td>UUUUUUU</td>
      <td>U/7</td>
    </tr>
    <tr>
      <td>s4</td>
      <td>FUFPFPF</td>
      <td>F/1-U/1-F/1-P/1-F/1-P/1-F/1</td>
    </tr>
    <tr>
      <td>s5</td>
      <td>PPFUUPP</td>
      <td>P/2-F/1-U/2-P/2</td>
    </tr>
    <tr>
      <td>s6</td>
      <td>FFPPPPP</td>
      <td>F/2-P/5</td>
    </tr>
    <tr>
      <td>s7</td>
      <td>FFPUUUU</td>
      <td>F/2-P/1-U/4</td>
    </tr>
  </tbody>
</table>

---


## Page 7

&lt;page_number&gt;6&lt;/page_number&gt; G. RITSCHARD AND T. F. LIAO

the spell of origin with a compensation of subsequent spells. The three models differ in the compensating mechanism.

## 2.2. Model ‘indep’: possible erasures of spells

This first model compensates for the change of duration of the spell of origin by a change in the spell that immediately follows. As defined earlier, $j$ is the timing change or error in $[-J, +J]$. A replication is generated by adding $j$ to the duration $d^-$ of the spell before the transition and subtracting $j$ from the duration $d^+$ of the spell after the transition.

For example, applying $j = -1$ to the first transition (advancing transition by one unit of time) in FFPUUUU (sequence s7 in Table 1), the sequence becomes FPPUUUU. Applying $j = 1$ (delaying the ending of the first spell by one unit of time) reduces the duration of the spell of ‘P’ to zero, that is, erases the spell of part time, resulting in FFFUUUU.

When $J > 1$, the timing change of one transition may require reducing the duration of more than one spell. For example, delaying the ending of the first spell by 2 units of time ($j = 2$) in sequence s7, requires reducing the duration of the second spell from 1 to 0 and the duration of the third spell from 4 to 3, resulting in FFFFUUU. When delaying the transition ($j > 0$), we augment the duration of the spell preceding the transition and reduce as necessary the duration of the following spells. When advancing the transition ($j < 0$), we augment the duration of the spell after the transition and reduce as necessary the duration of the preceding spells.

Timing changes are applied successively to all transitions from left to right. Spells whose duration was set to zero after a previous change are treated as any other spell and will retrieve a non-zero duration if we delay the transition that originates in that spell. Once the timing of all transitions has been changed, spells with zero duration are dropped from the sequence. Consequently, the resulting simulated sequence can have a shorter DSS.

Although model ‘indep’ applies the successive timing changes independently, the length of the sequences must be preserved. Therefore, a transition cannot be advanced before the start of a sequence and cannot be delayed after the end of a sequence.

## 2.3. Model ‘keep.dss’: preserving the DSS

This second model works similarly to the previous one, except that it imposes restrictions to avoid suppressing spells. These restrictions constrain the simulated sequences to share the DSS with the original sequence.

As seen above, when applying a change of $j$ without restrictions, one of the resulting durations $d^- + j$ and $d^+ - j$, where $d^-$ is the duration of the spell of origin and $d^+$ the duration of the subsequent spell, can be zero or negative, which would imply spell deletions and modifying the DSS. To avoid this, model ‘keep.dss’ imposes the new durations $d^- + j$ and $d^+ - j$ to be strictly positive. Formally, this leads to defining the applied change $j^*$ as

---


## Page 8

INTERNATIONAL JOURNAL OF SOCIAL RESEARCH METHODOLOGY &lt;page_number&gt;7&lt;/page_number&gt;

&lt;img&gt;Original sequence&lt;/img&gt;

&lt;img&gt;indep&lt;/img&gt;
(1,1)
(1,-1)
(-1,1)
(-1,-1)

&lt;img&gt;keep.dss&lt;/img&gt;

&lt;img&gt;relative&lt;/img&gt;

**Figure 1.** Applying selected timing changes to sequence FFPUUUU using each of the three models. Left labels indicate wanted timing changes of the two successive transitions, e.g., (−1, −1) means that the two transitions should be advanced unless constraints apply.

$$
j^* = \begin{cases}
j, & \text{if } d^- + j > 0 \text{ and } d^+ - j > 0 \\
-(d^- - 1), & \text{if } d^- + j \leq 0 \\
d^+ - 1, & \text{if } d^+ - 1 \leq 0
\end{cases}
$$

Let us again illustrate with sequence s7 of Table 1, i.e. FFPUUUU. A change $j = -1$ (advancing) applies to the first transition without any complication, generating FPPUUUU. However, drawing $j = 1$ (delaying) for this same transition, we face an issue with the second spell, which lasts only one unit of time. Using the above restrictions, the timing change to apply becomes $j^* = 1 - 1 = 0$, that is, a no-change. After this no-change, the restrictions) prevent advancing the second transition in sequence s7 by one unit of time ($j = -1$), which would erase the spell of parttime ('P'). Advancing the second transition would be possible after advancing the first transition, in which case the simulated sequence would be FPUUUUU.

### 2.4. Model 'relative': preserving time to subsequent transitions

This model proceeds similarly to model 'indep' but instead of compensating the duration change of the spell before a transition on the spells that immediately follow the transition, the compensation is done on the last spell and as necessary on spells preceding the last spell. For example, to apply $j = 3$ (delaying) to the first transition of sequence s1 (PPFUUPP), the duration of the first spell is increased from 2 to 5, and this increase is compensated by reducing the duration of the last spell (of part time) from 2 to 0 and the duration of the spell preceding the last one (spell of unemployment, 'U') from 2 to 1. After this change, the sequence becomes PPPPPFU.

Timing changes are again applied successively from left to right and spells with zero duration are dropped once all transitions have been treated. As with model 'indep', the resulting simulated sequence can have a shorter DSS.

Figure 1 shows the sequences generated from sequence s7 by each of the three models for four drawn sequences of changes of the two transitions, namely (1,1), (1,−1), (−1,1), and (−1,−1), where 1 means delaying by one unit of time and −1 advancing by one unit of time. Models 'indep' and 'keep.dss' give the same results for (−1,1) and (−1,−1), but all other solutions differ across the models.

---


## Page 9

&lt;page_number&gt;8&lt;/page_number&gt; G. RITSCHARD AND T. F. LIAO

## 3. Tuning the MC process

The MC process for generating the simulated sets of sequences is subject to choices regarding the timing error model and the R number of replications.

The choice of the model depends on theoretical considerations about how timing errors may have occurred during data collection but also on how the model chosen may impact the range of results. For this latter aspect, because variations in dissimilarity-based SA outcomes are dictated by variations in dissimilarities, in the following discussion, we focus on the impact of the choices on the deviations of the MC-simulated dissimilarities from the observed dissimilarity, or more specifically, on the impact on the MC standard error, MCse, of the observed dissimilarity. Let us first explain how we compute such MCse of observed dissimilarities.

Using one of the timing error models, we generate R MC-sets and compute the pairwise dissimilarity matrix of each of these MC-sets. For each observed dissimilarity $d^{obs} = d(x, y)$, we consider the distribution of the R corresponding simulated dissimilarities $d_i = d(\tilde{x}, \tilde{y})$. This empirical distribution of MC-simulated dissimilarities is summarised by the mean, MCmean, and standard deviation, $MCsd = \sqrt{MCvar}$, of the MC-simulated distances. The variance $MCvar = (1/R) \sum_i (d_i - MCmean)^2$ considers deviations from the mean, while we need deviations from the observation $d^{obs}$. We consider for this the standard error $MCse(d^{obs})$, which we define as the square root of the mean squared error $(1/R) \sum_i (d_i - d^{obs})^2$. With this definition, MCse can be expressed as the square root of the sum of the variance of the simulated distances and the squared bias, i.e. $MCse(d^{obs}) = \sqrt{MCvar(d^{obs}) + [MCbias(d^{obs})]^2}$, where $MCbias(d^{obs})$ is the difference between the observed distance $d^{obs}$ and the MCmean.

## 3.1. Choosing the timing error model

Choosing an appropriate timing error model can be a complex decision based on the type of data (e.g. administrative records, prospective and retrospective surveys, etc.), the chosen time unit for reporting (such as day, week, month, year, etc.), and a priori knowledge. Choices about the timing error model include:

*   Model type: ‘indep’, ‘keep.dss’, ‘relative’;
*   $P_J$: the probability distribution of the timing error.

These choices depend on theoretical considerations about how timing errors may have occurred during data collection. Model ‘indep’ is the most conservative, generating the most diverse sequences. It should be the first choice unless we have high confidence in the DSS reported, in which case ‘keep.dss’ should be chosen, or when there are indications that respondents remember better the time since reference events than the exact calendar times, in which case the choice could be ‘relative’.

The range of possible errors and the probability distribution $P_J$ should be set according to what we can reasonably expect for the data at hand. A uniform distribution or a distribution with decreasing probabilities for larger timing errors

---


## Page 10

INTERNATIONAL JOURNAL OF SOCIAL RESEARCH METHODOLOGY &lt;page_number&gt;9&lt;/page_number&gt;

can be natural for possible distributions. When a priori knowledge of the range and frequency of the timing errors is available, the probability distribution $P_J$ should reflect it as much as possible. For example, if we have quarterly data and expect respondents to correctly report the quarter but possibly make a one year forward or backward error, and if, in addition, we assume a 50% probability of no error, a consistent $P_J$ could be (0.22, 0.01, 0.01, 0.01, 0.5, 0.01, 0.01, 0.01, 0.22). As another example, we may want to exploit the results of the study on smoking onset age by Bright and Soulakova (2014) who found that about 63% of the respondents shifted forwardly or backwardly the onset age by about 2.7 years on average. We show in Appendix H in the supplementary materials how a distribution $P_J$ consistent with such a priori information can be set using constrained Poisson distributions.

Alongside the contextual relevance, the choice will also depend on how the timing model chosen may impact the range of results and possibly the performance of the MC process. The model chosen affects the overall level of the standard error of the dissimilarities between MC-replicated sequences. The restrictions necessary to maintain the DSS when transition times change limit the possibilities to alter sequences and, consequently, reduce the variability of simulated distances. Therefore, model ‘keep.dss’ will generate lower standard errors than the other two models. For example, with $J = 1$ and a uniform distribution, the median ‘keep.dss’ standard error of the LCS distances between the example sequences in Table 1 is 1.5, while it is about 2 with model ‘indep’, and 1.8 with model ‘relative.’ The difference increases with $J$, and, for $J = 5$, the median standard error becomes 1.8 with ‘keep.dss’ against 4.3 with ‘indep’ and 3.9 with ‘relative’. Whatever the model, the previous examples also show that standard errors increase with $J$. (Detailed results, using OM distances and models ‘keep.dss’ and ‘indep’ with $J = 1$, 2, and 3, are provided in Appendix C in the supplementary materials.) These example standard errors were obtained using a uniform distribution of the timing errors. Using instead a distribution that gives a higher probability of small time changes than large changes would result in smaller MCse. Although increasing $J$ and imposing restrictions on timing changes may slightly affect computation time, we have not been able to detect any significant effect in our experiments with artificial or real data.

### 3.2. Choosing R, the number of replicated sequences

In the MC process, the sequence dataset is MC-replicated R times. Which value R should be used? We can expect that the larger the R, the more reliable the MCse will be. However, since computation time and required storage increase with R, we may want to keep R as small as possible while ensuring MC standard errors close to those that would be obtained with larger R.

Figure 2 shows how MCse, the standard error estimated, varies with the R number of replications. The figure reports results for the distances $d(s1,s2)$ and $d(s2,s4)$ and distances between pairs of extended sequences derived from $s1$, $s2$, and $s4$. For each pair of sequences considered, we have run four series of MC processes with $J = 2$, uniform distribution, model ‘indep’, and no other restrictions.

---


## Page 11

&lt;page_number&gt;10&lt;/page_number&gt; G. RITSCHARD AND T. F. LIAO

&lt;img&gt;LCS d(s1,s2) plot&lt;/img&gt;
&lt;img&gt;LCS d(s2,s4) plot&lt;/img&gt;
&lt;img&gt;LCS d(s1,s2), long sequences, constant tails plot&lt;/img&gt;
&lt;img&gt;LCS d(s2,s4), long sequences, constant tails plot&lt;/img&gt;
&lt;img&gt;LCS d(s1,s2), long, repeated sequences plot&lt;/img&gt;
&lt;img&gt;LCS d(s2,s4), long, repeated sequences plot&lt;/img&gt;

**Figure 2.** MC standard error (MCse) of LCS distances. Four runs of the MC process with J = 2, model “indep”, and varying R values. Left, MCse of distances between sequences derived from s1 and s2, and, right, from sequences s2 and s4. Top: sequences of length 7 of the mini example (Table 1); middle: sequences of length 77 obtained by completing sequences s1, s2, and s4 with a spell of length 70 respectively in state F, U, and P; bottom: sequences of length 77 obtained by juxtaposing 11 times the original sequence. The MCse starts to stabilize after R = 100 and variations become imperceptible after R = 500.

The plots show that while there are variations in the outcome when R is less than about 100, results stabilize after 100 whatever the length of the sequences and their complexity.

## 4. Uncertainty of SA outcome

So far, we have seen how, equipped with the MC-process of timing errors in sequences, we can generate R MC-sets of updated sequence data and the R associated MC-updated dissimilarity matrices. We have also seen that we need at least R = 100 MC-replications to

---


## Page 12

<header>INTERNATIONAL JOURNAL OF SOCIAL RESEARCH METHODOLOGY</header> &lt;page_number&gt;11&lt;/page_number&gt;

stabilize the overall variability of the MC-updated dissimilarities. These together form the basic input for assessing the impact of timing uncertainty in dissimilarity-based SA. The methodology we propose to assess this impact consists of running the analysis of interest on each MC-set and examining the distribution of either characteristic statistics of the outcome, such as cluster quality indexes, or of measures of the similarity – e.g. correlations – of MC-results with results for the observed sequences. It also is sometimes useful to graphically compare results for observed sequences with those of a selection of extreme or the full range of MC-sets.

Using the *biofam* dataset from the TraMineR package as an illustration, we show below how this general method applies. The illustrative *biofam* dataset is an excerpt of 2,000 16-year-long family life sequences from the 2001–2002 biographic survey of the Swiss Household Panel (Voorpostel et al., 2024). Sequences are coded with eight different states (see Figure 3). For this applied example, we generate $R = 100$ replications of the dataset by applying ‘indep’ timing changes with a distribution $P_I = (.1, .25, .3, .25, .1)$. Among the three timing error models, ‘indep’ is the one that generates the larger outcome variations.

### 4.1. Distance uncertainty

A first aspect we may want to analyse is how distances vary with timing errors. We have already seen how to compute the MCmean, MCsd, and MCse to summarise the empirical distribution of MC-simulated dissimilarities across MC-sets. Using these statistics, we can compute z-ratios by dividing observed distances by their respective standard errors, i.e. $z = d(x, y)/MCse(x, y)$. For $R$ large enough, we can identify distances with z-ratios less than 2 as potentially non-significant.¹ To avoid issues with zero or close to zero MCse’s, we set the $z$ value to zero when the distance is zero and to 99 when the ratio exceeds that value. In addition to MCse and z-ratios, we suggest looking at how MC-replicated dissimilarities correlate with observed dissimilarities. This requires computing the – Spearman or Pearson – correlation for each of the $R$ MC-distance matrix. The higher these correlations, the higher the confidence in the observed dissimilarities would be.

For the ‘biofam’ data, the summary statistics of distributions of the $n(n-1)/2 = 1,999,000$ MCse and related values reported in Table 2 show that the distribution of the mean of the MC-simulated distances (MCmean) is very close to the distribution of the observed distances. Both range from 0 to 19.1. The MCse ranges from 0 to 2.91. Most ratios of observed dissimilarity over its MCse are greater than 2, indicating significant dissimilarities between pairs of sequences. Among the 1,999,000 ratios, 9.3% are less than 2, of which about 12% correspond to zero distances. Thus, about 8% of non-zero observed dissimilarities between sequences do not significantly differ from zero, i.e. about 8% of the pairs of non-identical sequences could be considered non-identical because of timing errors only.

**Table 2.** Summary of distribution of the MC-standard-errors (MCse) of the 1,999,000 pairwise dissimilarities and associated values.

<table>
  <thead>
    <tr>
      <th></th>
      <th>diss</th>
      <th>MCmean</th>
      <th>MCsd</th>
      <th>MCse</th>
      <th>diss/MCse</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Min</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <td>Q1</td>
      <td>5.30</td>
      <td>5.57</td>
      <td>0.97</td>
      <td>0.99</td>
      <td>3.71</td>
    </tr>
    <tr>
      <td>Median</td>
      <td>8.04</td>
      <td>8.14</td>
      <td>1.11</td>
      <td>1.15</td>
      <td>6.25</td>
    </tr>
    <tr>
      <td>Q3</td>
      <td>10.50</td>
      <td>10.55</td>
      <td>1.46</td>
      <td>1.54</td>
      <td>9.79</td>
    </tr>
    <tr>
      <td>Max</td>
      <td>19.12</td>
      <td>19.09</td>
      <td>2.91</td>
      <td>3.71</td>
      <td>99.00</td>
    </tr>
  </tbody>
</table>

---


## Page 13

&lt;page_number&gt;12&lt;/page_number&gt; G. RITSCHARD AND T. F. LIAO

**Table 3.** Distribution of R Spearman correlation values between observed and MC-simulated dissimilarities.

<table>
  <thead>
    <tr>
      <th></th>
      <th>Spearman Correlation between Dissimilarities</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Min.</td>
      <td>0.928</td>
    </tr>
    <tr>
      <td>Q1</td>
      <td>0.931</td>
    </tr>
    <tr>
      <td>Median</td>
      <td>0.932</td>
    </tr>
    <tr>
      <td>Mean</td>
      <td>0.932</td>
    </tr>
    <tr>
      <td>Q3</td>
      <td>0.934</td>
    </tr>
    <tr>
      <td>Max.</td>
      <td>0.937</td>
    </tr>
  </tbody>
</table>

The first column of **Table 3** summarises the distribution of the R Spearman correlations between observed and MC-simulated distances. The correlation remains high for all MC-sets and looks very stable, ranging from 0.928 to 0.937. Values of Pearson correlations (not shown) are very similar, ranging from 0.930 to 0.938. All this indicates that timing errors only moderately impact dissimilarities. However, let us look at how these timing-based variations in the distances may affect SA outcomes. Below, we examine effects on clustering results, the most frequently considered SA outcome. Effects on MDS scores and group comparisons are addressed as further illustrations in appendix G in the supplementary materials.

**4.2. Clustering**

Here, we consider partitioning around medoids (Kaufman & Rousseeuw, 2005, PAM), but we could proceed similarly with any other dissimilarity-based clustering method. We first consider the impact of timing uncertainty on the choice of the number of groups. To this end, we compute with WeightedCluster (Studer, 2013) a series of cluster quality indices (CQI) for a partitioning in a range of 2 to 10 groups of each MC-set and determine the optimum k number of groups suggested by each CQI for each MC-set. The distribution of the optimum k by CQI is displayed in **Table 4**. It appears that the optimum number of groups for PAM does not change much across the 100 MC-replicated sets. Among our 100 MC-sets, we get three different values with Calinski-Harabasz based on non-squared distances (CH), two with the point-biserial correlation (PBC), Hubert’s Gamma (HG), Somers’ d (HGSD) and Hubert-Levin C (HC), and a single value with the average silhouette width (ASW), Calinski-Harabashz using squared distances (CHsq), and the share of reproduced pseudo-variance (R2 and R2sq). From the ASW, the optimum number of groups is 5 for all MC-sets, a solution also supported by PBC, CH, and CHsq.

**Table 4.** Distribution of the optimal k number of groups by cluster quality index. PAM clustering in the range k = 2, ..., 10.

<table>
  <thead>
    <tr>
      <th>k</th>
      <th>ASW</th>
      <th>PBC</th>
      <th>HG</th>
      <th>HGSD</th>
      <th>CH</th>
      <th>CHsq</th>
      <th>R2</th>
      <th>R2sq</th>
      <th>HC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>45</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>5</td>
      <td>100</td>
      <td>99</td>
      <td>0</td>
      <td>0</td>
      <td>54</td>
      <td>100</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>52</td>
      <td>52</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>60</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>48</td>
      <td>48</td>
      <td>0</td>
      <td>0</td>
      <td>100</td>
      <td>100</td>
      <td>40</td>
    </tr>
  </tbody>
</table>

---


## Page 14

<header>INTERNATIONAL JOURNAL OF SOCIAL RESEARCH METHODOLOGY</header> &lt;page_number&gt;13&lt;/page_number&gt;

**Table 5.** Comparison of 100 MC-cluster solutions with solution for observed sequences. RI: Rand index, ARI: adjusted Rand index, MI: mutual information, AMI: adjusted mutual information, V: Cramer's v.

<table>
  <thead>
    <tr>
      <th></th>
      <th>Min</th>
      <th>Q1</th>
      <th>Median</th>
      <th>Q3</th>
      <th>Max</th>
      <th>Mean</th>
      <th>sd</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>RI</td>
      <td>0.89</td>
      <td>0.90</td>
      <td>0.90</td>
      <td>0.91</td>
      <td>0.92</td>
      <td>0.90</td>
      <td>0.01</td>
    </tr>
    <tr>
      <td>ARI</td>
      <td>0.69</td>
      <td>0.71</td>
      <td>0.72</td>
      <td>0.73</td>
      <td>0.76</td>
      <td>0.72</td>
      <td>0.02</td>
    </tr>
    <tr>
      <td>MI</td>
      <td>1.03</td>
      <td>1.05</td>
      <td>1.06</td>
      <td>1.08</td>
      <td>1.13</td>
      <td>1.07</td>
      <td>0.02</td>
    </tr>
    <tr>
      <td>AMI</td>
      <td>0.66</td>
      <td>0.68</td>
      <td>0.69</td>
      <td>0.69</td>
      <td>0.73</td>
      <td>0.69</td>
      <td>0.01</td>
    </tr>
    <tr>
      <td>V</td>
      <td>0.84</td>
      <td>0.86</td>
      <td>0.86</td>
      <td>0.87</td>
      <td>0.88</td>
      <td>0.86</td>
      <td>0.01</td>
    </tr>
  </tbody>
</table>

Retaining this last $k$ value of five, we now compare the solutions of PAM clustering into five groups by computing cluster similarity indices between the partition of the observed sequences and the solution obtained for each MC-set (see Vinh et al., 2010; Sundqvist et al., 2022, for a review of cluster similarity indexes). Table 5 summarises the distribution of a series of similarity indexes, namely the Rand index (RI), adjusted Rand index (ARI), mutual information (MI), adjusted mutual information (AMI), and Cramer's v (V). The indexes do not vary much across MC-sets. Their relatively high values indicate a strong similarity between the partition of the observed sequences and that of the MC-sets. Figure 3 illustrates with chronograms the partitioning into five

&lt;img&gt;Figure 3. Biofam data: chronograms of five-cluster solutions of the observed sequences (bottom) and of MC-sets X3 and X4 whose clustering resembles the less to the partition of the observed sequences according respectively to the ARI and AMI indexes.&lt;/img&gt;

<legend>
Parent
Left
Married
Left+Marr
Child
Left+Child
Left+Marr+Child
Divorced
</legend>

---


## Page 15

&lt;page_number&gt;14&lt;/page_number&gt; G. RITSCHARD AND T. F. LIAO

groups of the observed sequences (bottom panel) and of MC-sets X3 and X4 whose partitioning differs the most from the partition of the observed sequences according to ARI and AMI, respectively. The good news is that differences between cluster solutions are hardly visible, which assesses that the cluster analysis is quite robust against timing errors. Nevertheless, the cluster sizes slightly vary across sets, which means that the cluster memberships of a few individual sequences change because of the timing errors.

## 5. Uncertainty in cluster analysis of a published example

In this section, we replicate the cluster analysis in a published article by introducing uncertainty in the timing of reported transitions. The example under consideration is a study by Liao and Gan (2020), where the authors analysed migration trajectories of Filipina migrant domestic workers in Hong Kong. Their analysis is based on a life history calendar instrument in a survey conducted in 2017. In the article, they reported a cluster analysis of the 350 women with complete migration information from age 15 to under age 40.

In the cluster analysis, Liao and Gan (2020) used the dissimilarity measure SVRspell (with subsequence length weights kweights = 1 and spell duration weight tpow = 0) that provided great sensitivity to ordering or sequencing of states. They reported the results of a 3-cluster solution also because it is substantively most meaningful. To replicate their analysis, we cannot use their SVRspell parameter specification of tpow = 0, i.e. a spell duration weight of 0, because that parameter choice does not allow errors in timing. However, by specifying SVRspell with kweights = 1 and tpow = 3, we could produce clustering results sufficiently similar to those in the original published example.

In our replication of this cluster analysis, we select model ‘keep.dss’ and set $P_J = (.025, .05, .075, .2, .3, .2, .075, .05, .025)$, and $R = 300$. A few words about these parameter choices are in order: We chose model ‘keep.dss’ because it is extremely unlikely that the respondents reported countries in which they stayed less than a month before coming to Hong Kong and because the sequencing of the previous stays is a primary concern in this study. We set $P_J$ as a probability vector giving lower probabilities to large errors and giving higher probabilities to no or small errors, because this looks more realistic than a uniform distribution, and we decided, with a vector $P_J$ of length 9, to accommodate errors in a range of four time-units before or after the reported $t$ because ethnographic observations revealed that some women might choose to round up (or down) to five months in the reported transitions. We chose $R = 300$ to achieve a full range of results in the obtained CQI estimates, for showing as much variation as feasible in our visualization. For our MC simulation of timing uncertainty of the published example, we focused on three commonly used statistics for studying cluster quality – Average Silhouette Width (ASW), Hubert’s Gamma (HG), and Point Biserial Correlation (PBC) – which can provide insightful results.

Figure 4 displays the ASW, HG, and PBC CQI results for a cluster analysis of two to 20 clusters, with the MC simulated CQI values shown as dashed lines with their corresponding mean values shown as solid color lines, while the observed results are shown as solid black lines. Both the observed HG and PBC CQIs (solid black lines showing a knee point before a sharper decline) appear to suggest a three-cluster

---


## Page 16

<header>INTERNATIONAL JOURNAL OF SOCIAL RESEARCH METHODOLOGY</header> &lt;page_number&gt;15&lt;/page_number&gt;

&lt;img&gt;A Comparison of Observed & MC Clustering CQI Values (R=300)&lt;/img&gt;

Figure 4. ASW, HG, and PBC quality measures for the 2 to 20 cluster solutions of the Filipino migration data with R = 300 simulations.

solution without settling for the minimum number of two or a substantively not meaningful much high number of clusters. The three-cluster ASW value is also in an acceptable higher 0.6 range. The clustering results based on the observed data are consistent with the published example, which also favored the substantively meaningful three-cluster solution. However, the results based on the MC simulations suggest a different conclusion (see the fourth observation below).

The MC simulations provide some insights into how timing uncertainty may impact the computed CQI values, as revealed by Figure 4. Four observations are in order. First, the CQI results based on the observed data are mostly located within the simulated results, which can be higher or lower than the observed patterns. Second, while the degree of uncertainty in terms of spread can remain high for both ASW and PBC for the entire range of 2 to 20 clusters, it stabilizes to a narrow range for HG with the increase in the number of clusters. Third, whereas the concentration of the ASW simulated values for the 3-cluster solution is higher than that based on the observed data, the counterpart concentrations of the HG and PBC simulated results are both lower than those based on the observed data; at the same time, all three concentrated CQI values are within an acceptable range. Fourth and

---


## Page 17

&lt;page_number&gt;16&lt;/page_number&gt; G. RITSCHARD AND T. F. LIAO

finally, will the third observation change our clustering conclusion? Figure 4 also presents the ASW, HG, and PBC MCmean values. Although the observed CQIs (black lines) support a 3-cluster solution, the HG MCmean for the six-cluster solution is 0.953, higher than those for lower-numbered clusters, with a corresponding ASW MCmean of 0.605 and a corresponding PBC MCmean of 0.664, both at an acceptable level. These collectively provide some support for a 6-cluster solution, and of course, in the final analysis, we must consider the substantive sensibleness of such a solution as well. The results in this section suggest that by considering uncertainty in timing reporting, we may gain additional insights for sequence cluster analysis, and possibly a different conclusion. This reanalysis also suggests that an assessment of uncertainty in timing errors can be employed in a sensitivity analysis in applied SA research.

## 6. Limits and further extensions

Alongside demanding computation and storage capacity requirements for running R times the SA, an important, more conceptual, limitation of the MC-process proposed is that it applies the same timing error model to all transitions of all sequences, which may be unrealistic. Without as much as defining a specific timing error model for each individual, which would hardly be feasible for datasets with thousands of sequences, several extensions could help make timing error models more realistic. We see three extensions that could be implemented by playing with the distribution $P_j$ of the timing error around the transition points of concern.

First, the probability distribution $P_j$ could be made dependent on covariates. For instance, assuming that older people are more prone to reporting timing errors, setting up a mechanism by which probabilities of large errors increase with a respondent’s age should allow to take this assumption into account.

Second, in retrospective surveys, we can expect that the older an event is, the higher the probability of reporting a timing error may become. This suggests making the distribution $P_j$ depend on the position of the concerned transition in the sequence with typically higher probability for large errors at early positions than near the end of the sequence.

Third, timing errors may be more frequent for some types of events than others. For example, we can reasonably expect smaller timing errors regarding first childbirths than for the start of spells of living with a partner. We could take this into account by making $P_j$ depend on the type of life course transition.

The timing error models used to generate the MC-sets are an original contribution of the paper. Other models would also be possible. For example, instead of focusing on transitions, we could focus on spell durations and, retaining the DSS of the sequence, randomly generate spell durations using a Poisson distribution. For each spell, we could use the observed spell duration as a parameter but would have to find how to constrain the sum of spell durations drawn to equal the sequence length. Other methods such as those used by Studer (2021) for generating sequences satisfying a null hypothesis for validating typologies or by Huang et al. (2025) for generating synthetic sequences sharing certain characteristics with observed sequences could probably also prove useful. Nevertheless, our timing error models have the advantage of focusing on the timing

---


## Page 18

INTERNATIONAL JOURNAL OF SOCIAL RESEARCH METHODOLOGY &lt;page_number&gt;17&lt;/page_number&gt;

aspect that we are interested in, while it could be more difficult to isolate timing effects from these last two alternative approaches.

Despite their simplicity, the three timing error models introduced in this paper suffice, in their original formulation, to assess the sensitivity of SA outcomes to timing errors. Limiting the range of possible timing errors reduces the resulting variance of the dissimilarities and, therefore, the variation of SA outcomes. The approach is conservative: by establishing that the timing errors modeled only lightly affect SA results, we can expect even smaller effects of more restricted timing errors.

## 7. Conclusion

To date, the standard practice in SA has always been to regard sequences as exact and measured without error. Transition timing is a common source of sequence measurement error and uncertainty, especially in data collected from surveys, in particular, any surveys employing a retrospective approach. In this paper, we proposed a method for conducting Monte Carlo simulations of transition timing errors for assessing their impact on SA outcomes. We considered three models for performing such simulations – ‘indep’, ‘keep.dss’, and ‘relative’ – depending on the assumptions of whether a spell can be suppressed or whether time to subsequent transitions must be kept. In the MC method proposed, we also allowed for the flexibility in a range of conditions, including the probability distribution of timing errors around the concerned transition, and the number of simulations. We further discussed issues of optimization in performing uncertainty simulations and found that when the R value or the number of simulations is at least 100, results will stabilize.

Through a large sample of real sequences for carrying out a cluster analysis of sequences (and MDS and group analysis in the supplementary material) and reanalysing a published example of sequence cluster analysis, we demonstrated how we can investigate the effect of possible timing errors on sequence analytic results and thus conclusions reached. In particular, from these illustrations with real data, it appears that transition timing uncertainty may impact on the optimal number of clusters, the assignment of individual sequences to clusters, and the comparison of specific individual sequences. Luckily, the examples also showed that global results such as group comparison and clusters of trajectories seem to be relatively robust against transition timing changes when the number of clusters is given. Here, it is worth mentioning that the measured impact of timing errors depends on the chosen metric for measuring dissimilarities between sequences (see supplementary material) and, in case of clustering, for example, of the chosen clustering algorithm as well.

The principle proposed is not limited to the SA outcomes used for illustration, i.e. distances clustering, MDS scores, and group analysis, but can easily apply to any other SA outcomes such as the identification of representative sequences and outliers. For example, when studying outliers with the methods proposed by Piccarreta and Struffolino (2024), we could check whether sequences identified as outliers remain at a distance above the selected threshold in the MC-sets or if their neighborhood remains empty in the MC-sets.

The MC-simulation-based method proposed requires intensive computation. Using optimization procedures described in the supplementary materials, the time needed for a full sensitivity analysis remains reasonable. For example, using parallelization for computing

---


## Page 19

&lt;page_number&gt;18&lt;/page_number&gt; G. RITSCHARD AND T. F. LIAO

MDS scores and the CQIs for the range 2 to 10 of clustering solutions, the computation for all the analyses with biofam (dissimilarities and cluster in the core of the paper, and MDS scores computation and group analysis in the supplementary materials) takes about 15 min on a laptop with a 3.00 GHz processor, 8 cores, and 32 GB RAM.

In our proposed method, we only considered transition timing errors, and we realize that there may be other sources of error in sequence data, such as incorrectly reported states and incorrectly reported ordering of states. However, such errors, while potentially possible, are far less common than transition timing errors. A possible way of simulating state reporting error could be to randomly turn a few elements in the sequences into missing values and resort to multiple imputation methods for sequence data such as those proposed by the seqimpute R package (Emery et al., 2024, 2025) to simulate state changes. Works on sequence uncertainty in bioinformatics such as (Becker et al., 2023) could also be an inspiring source. Future research may want to consider methods for modeling these additional sources of potential sequence error.

Although our methodological proposal applies to sequence data only, the uncertainty principle of measurement error can also apply to any other analyses that rely on the computation of dissimilarity matrices, such as network analysis. However, assessing effects of uncertainty for these other types of analysis is beyond the scope of the current paper, even though it can be worthy of investigation.

How can an applied SA researcher take advantage of the information from the proposed MC-simulation approach to analysing timing errors? Although we believe an analysis based on the observed data can and should still be reported in the main analysis, new results using R-simulated datasets can be reported in a further sensitivity analysis, such as presenting new clustering results and using them in a subsequent regression analysis.

Finally, because timing errors can affect the quality of data collection in both prospective and retrospective surveys, we suggest strengthening survey instrument design and interviewer training for both types of surveys. Prospective surveys still collect life history data retrospectively, just at shorter intervals (between waves) and more frequently as defined by the periodicity of such surveys. Therefore, when designing survey instruments and conducting interviewer training, we suggest finding substantively meaningful time cues, increasing their frequency for achieving shorter intervals between time cues, and emphasizing the importance of time cues to interviewers so that timing errors can be minimized.

**Note**

1. See Appendix D on the distribution of z-ratios under the zero-distance assumption in the supplementary material.

**Author contributions**

CRediT: **Gilbert Ritschard**: Conceptualization, Formal analysis, Investigation, Methodology, Software, Visualization, Writing – original draft, Writing – review & editing; **Tim F. Liao**: Conceptualization, Formal analysis, Investigation, Methodology, Visualization, Writing – original draft, Writing – review & editing.

---


## Page 20

INTERNATIONAL JOURNAL OF SOCIAL RESEARCH METHODOLOGY &lt;page_number&gt;19&lt;/page_number&gt;

**Disclosure statement**

No potential conflict of interest was reported by the author(s).

**Notes on contributors**

**Gilbert Ritschard** is Professor Emeritus of Statistics at the Geneva School of Social Sciences, University of Geneva. His research interests are in methods for life course analysis and, in particular, sequence analysis and related approaches. He was President of the Sequence Analysis Association at its creation, 2018–2020. With his team, he developed TraMineR, the most comprehensive sequence analysis tool kit for social research. He published on sequence analysis in, among others, Sociological Methods and Research, Sociological Methodology, Journal of the Royal Statistical Society: Series A, and Journal of Statistical Software.

**Tim F. Liao** is Professor of Sociology & Statistics and LAS Alumni Distinguished Professorial Scholar at University of Illinois Urbana-Champaign. He is the current Editor of Socius, an American Sociological Association journal, and was a recent President of the Sequence Analysis Association. He also served as a Deputy Editor of Demography and an Associate Editor of Advances in Life Course Research in recent years. He received the American Sociological Association Methodology’s Paul F. Lazarsfeld Award in 2021.

**ORCID**

Gilbert Ritschard &lt;img&gt;ORCID icon&lt;/img&gt; http://orcid.org/0000-0001-7776-0903
Tim F. Liao &lt;img&gt;ORCID icon&lt;/img&gt; http://orcid.org/0000-0002-1296-7660

**Main argument and key methodology**

The standard practice of sequence analysis (SA) in social research is to consider sequences as exact and measured without error, which is a strong assumption. We focus on timing errors and propose a method for assessing the possible impact of such errors on SA results. The method proceeds with Monte Carlo simulations using three different models for generating timing errors. Updating the set of sequences many times, the impact of timing errors is assessed by examining how the outcome varies across the MC-updated sequence sets. This is the first time the issue of timing errors in SA is addressed in the literature.

**References**

Abate, G. T., de Brauw, A., Gibson, J., Hirvonen, K., & Wolle, A. (2022). Telescoping error in recalled food consumption: Evidence from a survey experiment in Ethiopia. *The World Bank Economic Review*, 36(4), 889–908. https://doi.org/10.1093/wber/lhac015
Abbott, A., & Tsay, A. (2000). Sequence analysis and optimal matching methods in sociology, review and prospect. *Sociological Methods & Research*, 29(1), 3–33. https://doi.org/10.1177/0049124100029001001
Becker, D., Champredon, D., Chato, C., Gugan, G., & Poon, A. (2023). Sup: A probabilistic framework to propagate genome sequence uncertainty, with applications. *NAR Genomics and Bioinformatics*, 5(2). https://doi.org/10.1093/nargab/lqad038
Bright, B. C., & Soulakova, J. N. (2014). Evidence of telescoping in regular smoking onset age. *Nicotine & Tobacco Research*, 16(6), 717–724. https://doi.org/10.1093/ntr/ntt220
Celhay, P., Meyer, B. D., & Mittag, N. (2024). What leads to measurement errors? Evidence from reports of program participation in three surveys. *Journal of Econometrics*, 238(2), 105581. https://doi.org/10.1016/j.jeconom.2023.105581

---


## Page 21

&lt;page_number&gt;20&lt;/page_number&gt; &lt;img&gt;icon&lt;/img&gt; G. RITSCHARD AND T. F. LIAO

Chiquet, J., Rigaill, G., Sundqvist, M., Dervieux, V., & Bersani, F. (2023). Aricode: Efficient computations of standard clustering comparison measures. *Reference Manual, Comprehensive R Archive Network, CRAN*. https://doi.org/10.32614/CRAN.package.aricode
Drasch, K., & Matthes, B. (2013). Improving retrospective life course data by combining modularized self-reports and event history calendars: Experiences from a large scale survey. *Quality and Quantity*, 47(2), 817–838. https://doi.org/10.1007/s11135-011-9568-0
Emery, K., Guinchard, A., Taher, K., & Berchtold, A. (2025). Seqimpute: Imputation of missing data in sequence analysis. *Reference Manual, CRAN*. https://doi.org/10.32614/CRAN.package.seqimpute
Emery, K., Studer, M., & Berchtold, A. (2024). Comparison of imputation methods for univariate categorical longitudinal data. *Quality and Quantity*, 59(2), 1767–1791. https://doi.org/10.1007/s11135-024-02028-z
Engstrom, C., & Sinibaldi, J. (2024). Reducing burden in a web survey through dependent interviewing. *Journal of Survey Statistics and Methodology*, 12(1), 60–79. https://doi.org/10.1093/jssam/smad006
Gabadinho, A., & Ritschard, G. (2016). Analysing state sequences with probabilistic suffix trees: The PST R package. *Journal of Statistical Software*, 72(3), 1–39. https://doi.org/10.18637/jss.v072.i03
Gabadinho, A., Ritschard, G., Müller, N. S., & Studer, M. (2011). Analyzing and visualizing state sequences in R with TraMineR. *Journal of Statistical Software*, 40(4), 1–37. https://doi.org/10.18637/jss.v040.i04
Harrison, R. L. (2010). Introduction to Monte Carlo simulation. *AIP Conf Proc*, 1204, 17–21. https://doi.org/10.1063/1.3295638
Herman, J. L., Novák, A., Lyngsø, R., Szabó, A., Miklós, I., & Hein, J. (2015). Efficient representation of uncertainty in multiple sequence alignments using directed acyclic graphs. *BMC Bioinformatics*, 16(108). https://doi.org/10.1186/s12859-015-0516-1
Huang, Z., Wolfson, J., Fulkerson, J. A., Demmer, R., & Chen, H. N. (2025). A flexible framework for synthesizing categorical sequences with application to human activity patterns. *Journal of Computational and Graphical Statistics*, 34(4), 1–14. https://doi.org/10.1080/10618600.2025.2450461
Jäckle, A., & Eckman, S. (2020). Is that still the same? Has that changed? On the accuracy of measuring change with dependent interviewing. *Journal of Survey Statistics and Methodology*, 8(4), 706–725. https://doi.org/10.1093/jssam/smz021
Kaufman, L., & Rousseeuw, P. J. (2005). *Finding groups in data*. John Wiley & Sons.
Kroese, D. P., Brereton, T., Taimre, T., & Botev, Z. I. (2014). Why the Monte Carlo method is so important today. *WIRES Computational Statistics*, 6(6), 386–392. https://doi.org/10.1002/wics.1314
Kroese, D. P., & Rubinstein, R. Y. (2011). Monte Carlo methods. *WIRES Computational Statistics*, 4(1), 48–58. https://doi.org/10.1002/wics.194
Liao, T. F., Bolano, D., Brzinsky-Fay, C., Cornwell, B., Fasang, A. E., Helske, S., Piccarreta, R., Raab, M., Ritschard, G., Struffolino, E., & Studer, M. (2022). Sequence analysis: Its past, present, and future. *Social Science Research*, 107, 102772. https://doi.org/10.1016/j.ssresearch.2022.102772
Liao, T. F., & Fasang, A. E. (2020). Comparing groups of life-course sequences using the Bayesian information criterion and the likelihood ratio test. *Sociological Methodology*, 51(1), 44–85. https://doi.org/10.1177/0081175020959401
Liao, T. F., & Gan, R. Y. (2020). Filipino and Indonesian migrant domestic workers in Hong Kong: Their life courses in migration. *The American Behavioral Scientist*, 64(6), 740–764. https://doi.org/10.1177/0002764220910229
O’Rawe, J. A., Ferson, S., & Lyon, G. J. (2015). Accounting for uncertainty in DNA sequencing data. *Trends in Genetics*, 31(2), 61–66. https://doi.org/10.1016/j.tig.2014.12.002
Piccarreta, R., & Struffolino, E. (2024). Identifying and qualifying deviant cases in clusters of sequences: The why and the how. *European Journal of Population*, 40(1). https://doi.org/10.1007/s10680-023-09682-3

---


## Page 22

INTERNATIONAL JOURNAL OF SOCIAL RESEARCH METHODOLOGY &lt;page_number&gt;21&lt;/page_number&gt;

Piccarreta, R., & Studer, M. (2019). Holistic analysis of the life course: Methodological challenges and new perspectives. *Advances in Life Course Research*, 41, 100251. https://doi.org/10.1016/j.alcr.2018.10.004
Preud’homme, G., Duarte, K., Dalleau, K., Lacomblez, C., Bresso, E., Smaïl-Tabbone, M., Couceiro, M., Devignes, M., Kobayashi, M., Huttin, O., Ferreira, J. P., Zannad, F., Rossignol, P., & Girerd, N. (2021). Head-to-head comparison of clustering methods for heterogeneous data: A simulation-driven benchmark. *Scientific Reports*, 11(1), 11. https://doi.org/10.1038/s41598-021-83340-8
R Core Team. (2024). *R: A language and environment for statistical computing*. URL R Foundation for Statistical Computing. https://www.r-project.org
Rettig, T., & Struminskaya, B. (2023). Memory effects in online panel surveys: Investigating respondents’ ability to recall responses from a previous panel wave. *Survey Research Methods*, 17(3). https://doi.org/10.18148/srm/2023.v17i3.7991
Robette, N., & Bry, X. (2012). Harpoon or bait? A comparison of various metrics in fishing for sequence patterns. *Bulletin of Sociological Methodology/Bulletin de Méthodologie Sociologique*, 116(1), 5–24. https://doi.org/10.1177/0759106312454635
Studer, M. (2013). *WeightedCluster library manual: A practical guide to creating typologies of trajectories in the social sciences with R*. Lives working papers 24. NCCR LIVES. https://doi.org/10.12682/lives.2296-1658.2013.24
Studer, M. (2021). Validating sequence analysis typologies using parametric bootstrap. *Sociological Methodology*, 51(2), 290–318. https://doi.org/10.1177/00811750211014232
Studer, M., & Ritschard, G. (2016). What matters in differences between life trajectories: A comparative review of sequence dissimilarity measures. *Journal of the Royal Statistical Society, Series A*, 179(2), 481–511. https://doi.org/10.1111/rssa.12125
Studer, M., Ritschard, G., Gabadinho, A., & Müller, N. S. (2011). Discrepancy analysis of state sequences. *Sociological Methods & Research*, 40(3), 471–510. https://doi.org/10.1177/0049124111415372
Sundqvist, M., Chiquet, J., & Rigalli, G. (2022). Adjusting the adjusted Rand index: A multinomial story. *Computational Statistics*, 38(1), 327–347. https://doi.org/10.1007/s00180-022-01230-7
Vinh, N. X., Epps, J., & Bailey, J. (2010). Information theoretic measures for clusterings comparison: Variants, properties, normalization and correction for chance. *The Journal of Machine Learning Research*, 11, 2837–2854. URL http://jmlr.org/papers/v11/vinh10a.html
Voorpostel, M., Tillmann, R., Lebert, F., Kuhn, U., Lipps, O., Ryser, V. A., Antal, E., Dasoki, N., Janssen, C., & Wernli, B. (2024). *Swiss household panel user guide (1999-2022), wave 24*. FORS.