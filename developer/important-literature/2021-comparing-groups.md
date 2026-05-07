## Page 1

<header>Original Article</header>

&lt;img&gt;ASA American Sociological Association&lt;/img&gt;

Sociological Methodology
2021, Vol. 51(1) 44–85
© American Sociological Association 2020
DOI: 10.1177/0081175020959401
http://sm.sagepub.com
&lt;img&gt;SAGE&lt;/img&gt;

# Comparing Groups of Life-Course Sequences Using the Bayesian Information Criterion and the Likelihood-Ratio Test

**Tim Futing Liao¹** and **Anette Eva Fasang²**

**Abstract**
How can we statistically assess differences in groups of life-course trajectories? The authors address a long-standing inadequacy of social sequence analysis by proposing an adaption of the Bayesian information criterion (BIC) and the likelihood-ratio test (LRT) for assessing differences in groups of sequence data. Unlike previous methods, this adaption provides a useful measure for degrees of difference, that is, the substantive significance, and the statistical significance of differences between predefined groups of life-course trajectories. The authors present a simulation study and an empirical application on whether employment life-courses converged after reunification in the former East Germany and West Germany, using data for six birth-cohort groups ages 15 to 40 years from the German National Education Panel Study. The new methods allow the authors to show that convergence of employment life-courses around reunification was stronger for men than for women and that it was most pronounced in terms of the duration of employment states but weaker for their order and timing in the life-course. Convergence of East German and West German women's employment lives set in earlier and reflects a secular trend toward a more gender-egalitarian division of labor in West Germany that is unrelated to reunification. The simulation study and the substantive application demonstrate the usefulness of the proposed BIC and LRT methods for assessing group differences in sequence data.

**Keywords**
Bayesian information criterion, likelihood-ratio test, sequence analysis, life-course, East Germany and West Germany

**INTRODUCTION**
Sequence analysis has been widely applied in the social sciences since its introduction by Abbott and Forrest (1986) more than three decades ago. Despite rapid and continuing methodological advances in social sequence analysis (Barban et al. 2017; Blanchard, Bühlmann, and Gauthier 2014; Cornwell 2015; Piccarreta 2017; Raab et al. 2014; Studer 2013; Studer, Struffolino, and Fasang 2018), some core critiques

¹University of Illinois at Urbana-Champaign, Urbana, IL, USA
²Humboldt University of Berlin and WZB Berlin Social Science Center, Berlin, Germany

**Corresponding Author:**
Tim Futing Liao, University of Illinois at Urbana-Champaign, Department of Sociology, 3120 Lincoln Hall, 702 S. Wright Street, Urbana, IL 61801, USA.
Email: tfliao@illinois.edu

---


## Page 2

<header>Liao and Fasang</header>
&lt;page_number&gt;45&lt;/page_number&gt;

voiced against the method almost two decades ago remain valid today, including a lack of statistical criteria for assessing meaningful differences between sets of sequences (Levine 2000; Piccarreta and Studer 2018; Wu 2000). A notable exception in this regard is discrepancy analysis (Studer et al. 2011), whose merits and limitations we will discuss in detail.

In this article, we propose a statistical assessment of group differences between life-course sequences. We demonstrate the utility of the proposed methods first with a simulation study and second with an example application on employment life-courses in East Germany and West Germany before and after reunification in 1990. The proposed methods allow us to distinguish between similarity of life-courses between groups (cohorts in our example application) in terms of the timing, duration, and order (or sequencing, a term we use interchangeably) of life-course states. This distinction is important in assessing whether institutional change affects different temporal features of life-courses differentially, which can help us examine the mechanisms through which macro-structural change affects individual lives.

Methodologically, statistical assessments of a single change of states or durations, such as the transition to unemployment or the duration until reemployment, are straightforward. Mean timing and duration can be compared across samples, and their expected values are easy to use in further analysis. With sequence data of numerous sequentially linked categorical life-course states and changes between them, however, no obvious single aspect best represents an entire sequence for statistical assessments. Despite the theoretical preeminence of the trajectory concept in the life-course literature (Aisenbrey and Fasang 2010), existing methods cannot adequately statistically assess the overall difference between sets of sequences. Cluster analysis with the aim of finding distinct clusters that emerge from a sample of sequences has greatly improved (Studer 2013). But many issues remain in assessing the reliability and validity of life-course typologies derived with cluster analysis (Piccarreta and Studer 2018). Generally, classification error associated with cluster assignment cannot be captured in (bootstrap) confidence intervals, because the outcome of interest is a qualitative categorical typology and not metric (Jalovaara and Fasang 2020). Therefore, statistically sound comparisons of predefined groups of sequences outside of clustering can be particularly attractive. To date, we still lack methods for statistically assessing differences between fixed groups or samples of sequences. Yet comparing theoretically or empirically predefined groups is at the core of many social science research questions, for example, interests in differences by gender, race, or change across birth cohorts. To that end, we propose a method to adapt the Bayesian information criterion (BIC) and the likelihood-ratio test (LRT) statistic for sequence comparisons. The LRT is a common statistical test in many disciplines, and the BIC, originally introduced by Raftery (1986) to sociology, has become a popular method for comparing and selecting models in the social sciences and beyond.

To put our methodological contributions in perspective, we follow Tukey’s (1977) division of applied data analysis into exploratory and confirmatory data analysis. Exploratory data analysis often uses graphical means for discovering patterns in multi-dimensional visualizations (see Tukey and Tukey 1988a, 1988b). Most often a mere

---


## Page 3

&lt;page_number&gt;46&lt;/page_number&gt;
Sociological Methodology 51(1)

path of exploratory analysis is insufficient. For example, a simple comparison of men’s and women’s educational attainment in a distributional plot cannot ascertain if the two educational attainment distributions are fundamentally different until a significance test is applied, such as a t test of the mean difference. This type of statistical testing falls into the domain of confirmatory data analysis, which quantifies the extent to which deviations from a null model could be expected to occur by chance (Gelman 2004). In this article, we visualize life-course sequences for discovering trends and patterns in an exploratory fashion. The proposed BIC and LRT methods provide summaries of these trends and patterns. In addition, our BIC measures provide Jeffreys’s (1961) “levels of evidence,” defined as the strength of statistical evidence supporting a hypothesis of such trends and patterns. They are the foundation for Kass and Raftery’s (1995) guidelines for using BICs (further elaborated in a later section) that provide benchmarks for assessing degrees of difference between groups.

A long-standing critique of common misinterpretations of, and overreliance on, simple statistical significance testing recently intensified in the social sciences, with some scholars advocating for a broader evaluation of the “social” and substantive significance of empirical findings (Bernardi, Chakhaia, and Leopold 2017; Gross 2015; McShane et al. 2019). McShane et al. (2019) recently suggested “abandoning statistical significance” and demoting p-value thresholds from their screening role to attest meaningful findings. Instead, p values should be treated continuously along with other factors to evaluate the quality and relevance of a statistical result, including related prior evidence, plausibility of mechanisms, and data quality. Similarly, Bernardi et al. (2017) suggested an informed benchmarking of estimates against minimum and maximum plausible values of an effect. The BIC and LRT adaptation to sequence comparisons that we propose also depart from reliance on binary statistical significance. In addition to emphasizing degrees of difference (effect sizes), the methods can be augmented with visualizations that inform the substantive content of varying degrees of difference in categorical social sequence data—in our application, employment life-courses. The degrees of difference (BIC values) and visualizations thus provide information about the social significance of differences between groups of life-courses, in addition to providing a p value that should be interpreted continuously, not as a binary threshold.

Our empirical application to employment life-courses follows birth cohorts born 1944 to 1971 between ages 15 and 40 years over three decades before and after German reunification in 1990. The older cohorts experienced most of their early careers in divided Germany—a socialist communist society in the East and a democratic social market economy in the West. To illustrate the applicability of the proposed methods, we assess the degree and type of similarity between East German and West German men’s and women’s employment life-courses over time using data from the German National Education Panel Study (NEPS) (Blossfeld, Roßbach, and von Maurice 2011). The proposed BIC and LRT provide a detailed and statistically sound assessment of between-group differences in different life-course properties (timing, duration, and order) across cohorts.

---


## Page 4

<header>Liao and Fasang</header>
&lt;page_number&gt;47&lt;/page_number&gt;

We proceed as follows. We first present a new method for computing BIC and LRT statistics for analyzing sequence differences. Next, a simulation study illustrates some of the general properties of the proposed methods. After introducing the East German and West German contexts and some theoretical considerations, the new BIC and LRT statistics for comparing life-course sequences are applied to the NEPS data. The conclusion highlights the contributions of our study: the proposed BIC and LRT for the statistical assessment of life-course differences performed well in the simulation study and in the substantive application. We therefore conclude that they are promising methods to address other research questions about the similarity of sets of sequence data.

## A BIC AND LRT METHOD FOR SEQUENCE ASSESSMENT

Comparing sequences across samples is essentially a problem of assessing differences across groups. Such an assessment can be conducted by computing distance measures within and between sets of sequences belonging in specific groups. On one hand, distances within groups summarize the degree of homogeneity or standardization of the life-courses in each set of sequences, which can then be compared across groups. On the other hand, distances between sequences belonging to different groups provide a direct measure of the difference between the groups, rather than comparative group-specific summaries (Fasang 2014).

The distance-based discrepancy analysis proposed by Studer et al. (2011) is potentially a good candidate for analyzing differences between sets of sequences. In our simulation study, we compare the proposed BIC with the pseudo-R² from a discrepancy analysis. A lack of guidelines on what constitutes statistically meaningful differences between sets of sequences is one of the core criticisms of sequence analysis (e.g., Piccarreta and Studer 2018; Wu 2000). By combining sequence analysis with the more conventional statistical concepts of BIC and LRT, we aim to fill this gap in sequence methodology. The BIC has been used on Markov chains (Berchtold and Raftery 2002), including Markov chains based on sequence data (Gabadinho and Ritschard 2016), and when assessing hidden Markov models of sequence data (Helske and Helske 2017).

Sequence analysis is a collection of techniques to compare similarity between sequences. There are several ways to directly assess the difference between groups of sequences. First, the classical Levenshtein or another statistical distance can be computed between all possible pairs of sequences of the groups under comparison (Fasang 2014). Alternatively, one can calculate a distance of each sequence in a group to the group-specific medoid sequence, a popular method used for complexity reduction in recent sequence visualization and comparison (Aassve, Billari, and Piccarreta 2007; Fasang and Liao 2014; Gabadinho et al. 2011; Piccarreta 2012). The sequences of each group can then be compared with the medoids of the other groups to determine how similar they are to their own medoid and the medoids of other groups. Another, less explored, possibility is to compute distances between the sequences in a group and their collective center of gravity (Studer et al. 2011), which can then be calculated and compared within and between groups. The gravity center is given by the

---


## Page 5

&lt;page_number&gt;48&lt;/page_number&gt;
<header>Sociological Methodology 51(1)</header>

(hypothetical) “sequence” that minimizes the sum of distances from all sequences in a given group. The gravity-center sequence can be, but does not have to be, empirically observed in a given set of sequences. In an earlier research stage, we used the medoid-based approach, but because of unstable results, we settled on the gravity-center approach.

We propose to adapt the BIC and the LRT to sequence analysis by using this gravity-center approach as an alternative method for assessing differences between sets of sequences on the basis of distances computed from group-specific versus overall gravity centers. It is a straightforward application of the BIC to gravity-center differences. Depending on which temporal feature researchers are most interested in, they can choose between different distance measures that emphasize sequence similarity either in state occurrence, order, timing, or duration of states (for an overview of distance measures, see Aisenbrey and Fasang 2010; Robette and Bry 2012; Studer and Ritschard 2016).

In this article, we demonstrate the gravity-center approach for computing the BIC and the LRT with four different distance measures that have differential sensitivities to differences in state occurrence, timing, duration, and order. We use all four in our simulation study, but we use only the latter three measures to assess differences in the empirical application on East German and West German employment life-courses, because they isolate similarity in the substantively relevant features of timing, duration, and order of sequence states (Fasang 2012).

The proposed BIC and LRT statistics are often applied to parametric models on the basis of distributional assumptions. Sequence analysis, in contrast, originates in the algorithmic, nonparametric data modeling culture (Aisenbrey and Fasang 2010) of exploratory data mining approaches that do not make any prior assumptions about distributions in data (Tukey 1977). In sequence discrepancy analysis, Studer et al. (2011) proposed a permutation-based method to test the amount of (pseudo-)variance in a sequence distance matrix that is accounted for by other variables by calculating a pseudo-$R^2$. The pseudo-$R^2$ values in discrepancy analysis give a measure for the strength of an association between covariates that can mark group differences and sequence distances, with a similar aim to that of our BIC (for an application of discrepancy analysis to the NEPS data, see Struffolino, Studer, and Fasang 2016). However, they are often very low and therefore difficult to interpret, without clear established benchmarks of which values indicate strong or weak effects, as are available with the established guidelines for interpreting the BIC difference (see Table 1). Nonparametric permutation methods are appealing because they do not rely on distributional assumptions. In contrast, the statistics we propose have the advantage of being less computationally intensive and providing either levels of statistical evidence (as with BICs) or $p$ values (as with LRTs). Therefore, they do not have to deal with inconclusive intervals, and they provide an easily interpretable value of the degree of difference (effect size), for which there are established benchmarks to assess the social significance of a finding (Kass and Raftery 1995).

Let $s_i$ denote the sum of squared distance of sequence $j$ in sequence group $i$ to the gravity center of group $i$:

---


## Page 6

<header>Liao and Fasang</header>
&lt;page_number&gt;49&lt;/page_number&gt;

<table>
  <caption>Table 1. BIC Comparison Guide</caption>
  <thead>
    <tr>
      <th>Evidence</th>
      <th>BIC Difference</th>
      <th>Bayes Factor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Not worth a bare mention</td>
      <td>0 to 2</td>
      <td>1 to 3</td>
    </tr>
    <tr>
      <td>Positive</td>
      <td>2 to 6</td>
      <td>3 to 20</td>
    </tr>
    <tr>
      <td>Strong</td>
      <td>6 to 10</td>
      <td>20 to 150</td>
    </tr>
    <tr>
      <td>Very strong</td>
      <td>>10</td>
      <td>>150</td>
    </tr>
  </tbody>
</table>

Note: BIC = Bayesian information criterion.

$$
s_i = \sum_{j=1}^{n_i} q_{ij}^2,
\quad (1)
$$

where $q_{ij}$ is the distance between each sequence $j$ in the $i$th sequence group and the $i$th gravity center of the same group, and $n_i$ denotes the number of sequences in sequence group $i$. The statistic in equation (1) can be computed for all $G$ number of sequence groups.

### BIC for Sequence Comparison

We adapt the BIC to sequence comparisons by replacing errors in a linear model in the conventional application of BIC with sequence distances from group-specific or overall gravity-center sequences. Burnham and Anderson (1998, 2004) expressed least squares of normal errors as likelihoods for constructing BICs as well as the closely related Akaike information criterion.¹ However, it is unclear from the literature whether BIC computations are still appropriate for departure from normality. We will present a simulation to investigate the appropriateness after presenting our BIC computational formulas for group-based sequence distances.

We consider a series of sequence distances as a type of error, such as differences between each sequence and the group's gravity center, similar to a series of errors resulting from the difference between an observed $y$ and its predictions. Such sequence distances can be transformed into -2 log-likelihoods and BICs as follows:

$$
-2 \log\text{-likelihood}_i = n_i \log\left(\frac{s_i}{n_i}\right).
\quad (2)
$$

The BIC for the $i$th sequence group can then be expressed as

$$
\text{BIC}_i = n_i \log\left(\frac{s_i}{n_i}\right) - K \log(n_i).
\quad (3)
$$

This is the BIC form reported by Raftery (1995, equation 21) with the saturated model as the baseline, measuring deviation from the gravity center in the context of sequences. For the $i$th sequence group, there is only one degree of freedom, $K = 1$. The only degree of freedom involved is that associated with the $i$th gravity center. For comparing the $h$th and $i$th sequence groups, or more generally, $G$ number of groups, we propose a nested model approach, which is consistent with the typical application of BIC when models are nested. We define BIC$_A$ as the BIC based on the sum of

---


## Page 7

&lt;page_number&gt;50&lt;/page_number&gt;
Sociological Methodology 51(1)

squared distances using the overall gravity center and BIC$_G$ as the BIC based on the sum of squared distances using group-specific gravity centers:

$$
\text{BIC}_A = n_A \log\left(\frac{s_A}{n_A}\right) - K \log(n_A), \quad (4)
$$

and

$$
\text{BIC}_G = n_A \log\left(\sum_{i=1}^G \frac{s_i}{n_i}\right) - K \log(n_A), \quad (5)
$$

where $s_A$ is the sum of squared distances, $n_A$ is the total number of distances in all groups combined, and the parameter $K$ gives the number of degrees of freedom. $K$ equals 1 in equation (4) because only the overall gravity center is used. $K$ equals $G$ in equation (5) because $G$ gravity centers are used in the computation using the TraMineR package in R (Gabadinho et al. 2011; Studer and Ritschard 2016).² The BIC difference between equations (4) and (5), or equation (4) minus equation (5), gives the criterion for general model comparison, nested or not, as described in Raftery (1995, equation 22). However, we will apply $\delta\text{BIC} = \text{LRT}_{AG} - K \log(n_A)$ as in Raftery (1995, equations 20 and 21) for calculating the BIC difference in our simulation study and empirical application because our models are nested, where $\text{LRT}_{AG}$, to be presented under “LRT for Sequence Comparison,” is the LRT statistic between model $A$ and model $G$, the two data situations defined for equations (4) and (5), and $K$ is the degrees of freedom for the comparison of models $A$ and $G$.

To ensure that normality will not be an issue for the application of BIC differences, we additionally tested the sensitivity of the results for nonnormally distributed sequence distances (for details, see Appendix A). Sequence distances are typically not normally distributed, and this is the case in our real-life example data on East Germany and West Germany. Appendix Figure A1 shows raw and normalized distances between (1) the total samples of East German and West German respondents (w-e), (2) East German and West German men (w-e, m), and (3) East German and West German women (w-e, f) before (observed distances) and after normalization (normal distances) for three different distance measures that we will later use in our empirical analysis (Hamming, SVRspell, and OMstran). Appendix Table A1 compares BIC values for the observed and normalized distances for each comparison group for the three different distance measures. The BIC difference based on equations (4) and (5) stays virtually unchanged whether the distribution of sequence distances follows a normal distribution or departs from such a distribution (Appendix Table A1). We therefore conclude that even relatively strong deviations from normality that typically occur in sequence distance distributions (Appendix Figure A1) do not distort the calculation of BIC difference.

To interpret the sizes of BIC differences, we follow the guidelines suggested by Kass and Raftery (1995) for assessing BIC differences and related Bayes factors (Table 1): a BIC difference between 0 and 2 indicates negligible differences between two sets of sequences, a difference of 2 to 6 suggests moderate differences, a BIC difference 6 or greater can be considered strong support, and a difference of 10 or greater

---


## Page 8

<header>Liao and Fasang</header>
&lt;page_number&gt;51&lt;/page_number&gt;

indicates very strong support for statistically meaningful differences in sets of sequences. The Bayes factor, according to Kass and Raftery (p. 777), is a summary of the evidence provided by the data in favor of a given scientific hypothesis or theory that represents one model versus another, the kind of evidence Jeffreys (1961) spoke about. Indeed, Kass and Raftery's guidelines for assessing BIC and Bayes factor values are presented in levels, as shown in Table 1, and are a revised version of Jeffreys's levels of evidence for examining degrees of difference between groups. Because BICs adjust for sample sizes and numbers of parameters, negative values may result. A negative value greater than -2 suggests negligible evidence for the null model (i.e., no difference between the two sets of sequences), a value between -2 and -6 implies moderate support for the null model, and so on for values above the lower bound.³ Accordingly, a BIC difference of 3 would suggest moderate difference between two sets of life-course sequences, whereas a BIC difference of 11 supports strong differences.

A note on how these BIC differences work is in order. The BIC differences for comparing sets of sequences rely on nested models of the same data in the same way as nested regression models are compared (or how $R^2$ values are compared) in an F test. The basic rule for BIC differences and F tests is that the models must be nested. However, once computed, the resulting F ratios (conditional on d.f.) and BIC differences or Bayes factors (d.f. are already adjusted in either) can be compared in absolute terms across samples or models of totally different data. In the case of the F ratio, the F distribution and F table serve as the universal yard stick (conditional on d.f.). In a similar fashion, Kass and Raftery's (1995) universal guidelines (see Table 1) serve as the universal rule for assessing the level of support for similarities of the two nested models in one data situation versus another.

### LRT for Sequence Comparison

The BIC difference provides an appropriate statistical assessment of the level, or degree, of differences between groups or samples of sequences, but it does not provide a significance test per se. To obtain a significance test, we construct an LRT using equation (2). Let $ll_i$ stand for the -2 log-likelihood of the $i$th group. Following Liao's (2002, 2004) discussion of the LRT for generalized linear models or logit models, the LRT for testing the null hypothesis that all groups have the same gravity center is obtained as the difference between the $ll$ from the restricted model and the $ll$ from the unrestricted model. In the current application of comparing $G$ number of sequence groups, the restricted model pertains to the situation where we assume all groups are equal, with an identical gravity center. The unrestricted model describes the situation in which all groups are considered unique, with group-specific gravity centers. Following Liao (2002, equations 6.13 and 6.14), we further define $ll_R$ as the -2 log-likelihood from the restricted model and $ll_U$ as the -2 log-likelihood from the unrestricted model where

$$
ll_U = \sum_{i=1}^{G} ll_i.
$$
(6)

---


## Page 9

&lt;page_number&gt;52&lt;/page_number&gt;
Sociological Methodology 51(1)

Therefore, the LRT for testing sequence-group differences is given as

LRT = ll_R - ll_U = ll_R - Σ_{i=1}^G ll_i ~ χ^2, (7)

with G - 1 degrees of freedom that follows the chi-square distribution. Applying equation (7) to sequence data by using equations (4) and (5), we obtain

LRT = n_A log(s_A / n_A) - n_A log(Σ_{i=1}^G s_i / n_i) ~ χ^2. (8)

The LRT of equation (8) is based on the nested model principle, just like the BIC difference given by equations (4) and (5), and follows the chi-square distribution. Thus, it can be considered a significance test equivalent to the nested BIC assessment. Indeed, the LRT and BIC are functionally related, differing only in an adjusting factor (Raftery 1995, equation 21).

The BIC and LRT statistics do not automatically measure convergence or divergence of life-courses over time, but they can be used to assess these properties when comparing two groups—for example, East Germans and West Germans or men and women—across birth cohorts. When the groups under comparison do not pertain to any development over time, we would simply assess the degree of similarity between some other group specifications. Extending the group comparison to multiple birth cohorts, the BIC allows us to assess divergence or convergence of life-courses over time, because the absolute BIC values reflect the degree of difference between two groups that can increase or decline across cohorts, similar to an effect size in a regression model. The LRT additionally provides a statistical assessment: whether smaller or larger differences are statistically significant. Sequence visualization techniques provide further information about the substantive content of the type of difference between groups of sequences of categorical states.

A Simulation Study Assessing the Performance of the BIC, the LRT, and the Discrepancy Pseudo-R^2

To assess the statistical properties of the BIC difference defined earlier, the LRT statistic of equation (8), and the pseudo-R^2 statistic of Studer et al. (2011), we conducted a simulation study of 1,000 resamples of randomly generated sequences of length 100 in four types of sequence differences: state occurrence, timing, order, and duration. The latter three are theoretically important temporal life-course dimensions that we will explore in the empirical example application on East Germany and West Germany (Elder, Johnson, and Crosnoe 2003; Fasang 2012; for an assessment of the sensitivity of different distance measures to order, timing, and duration of states, see Studer and Ritschard 2016:497, Table 2). To provide a comprehensive test of the performance of the BIC and LRT adaptation in the simulation study, we also consider state occurrence, as an additional temporal dimension of sequence differences. The simulation analysis tests the sensitivity of the BIC and LRT statistics to state occurrence, timing, duration, and order at various controlled levels of differences between groups 1 and 2 for various sample sizes.

---


## Page 10

<header>Liao and Fasang</header>
&lt;page_number&gt;53&lt;/page_number&gt;

<table>
  <caption>Table 2. Designs for Evaluating Sensitivity to Occurrence, Timing, Order, and Duration of States in the Simulation Study</caption>
  <thead>
    <tr>
      <th>Tested Dimension</th>
      <th>Description</th>
      <th>Group 1</th>
      <th>Group 2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Occurrence (OMspell)</td>
      <td>Sequences randomly select states from the alphabet of ABC or DEF and assign them to groups</td>
      <td>ABC</td>
      <td>DEF</td>
    </tr>
    <tr>
      <td>Timing (Hamming)</td>
      <td>Sequences follow the order of ABC and the time point $t$ at which state B begins is controlled</td>
      <td>$t \in \{22 \ldots 27\}$</td>
      <td>$t \in \{30 \ldots 35\}$</td>
    </tr>
    <tr>
      <td>Duration (OMstran)</td>
      <td>Sequences follow the order of ABC and the duration $d$ that state B lasts is controlled</td>
      <td>$d \in \{27 \ldots 32\}$</td>
      <td>$d \in \{33 \ldots 38\}$</td>
    </tr>
    <tr>
      <td>Order (SVRspell)</td>
      <td>Order patterns controlled in each group and the duration in each consecutive state are left random</td>
      <td>ABC</td>
      <td>CBA</td>
    </tr>
  </tbody>
</table>

Note: The simulation tests sensitivity to group differences in each of the four dimensions of trajectory features by mixing the two features of the two equal-sized groups of sequences of length 100 in increasing mixing proportions of 0, 2, 4, 6, 8, and 10 percent except for the order dimension, which uses the mixing proportions of 0, 20, 40, 60, 80, and 100 percent. For example, a 10 percent mixing means a simulated set of sequences contains group 1 features, and another simulated set of sequences of equal size has 10 percent of its sequences containing group 2 features. The simulation is performed 1,000 times, with $N$ for the combined group size of 100, 200, 500, 1,000, and 2,000.

To assess group differences in terms of state occurrence, timing, duration, and order, we use the distance measures and their criteria provided in the first, second, third, and fourth rows of Table 2, respectively, the last three of which are a simpler version of those used in Studer and Ritschard (2016:497, Table 2). To assess group similarity in state occurrence, we follow the idea that two sequences with no common state are maximally dissimilar (Dijkstra and Taris 1995; Elzinga 2003); in this case, we include the three states “A, B, and C” in group 1 and the three states “D, E, and F” in group 2. To assess similarity in state timing, a randomly chosen time point $t$ at which state B begins is drawn from integer values ranging from 22 to 27, or $t \in \{22 \ldots 27\}$, for group 1 and $t \in \{30 \ldots 35\}$ for group 2. The criteria we used for assessing similarity in terms of duration and order are defined according to the last two rows of Table 2, namely, $d \in \{27 \ldots 32\}$ for group 1 and $d \in \{33 \ldots 38\}$ for group 2 for state duration, and the order patterns of “ABC” for group 1 and “CBA” for group 2 for state order/sequencing.

To assess sensitivity to state occurrence, we chose a particular distance measure, OMspell with an expansion cost of 0.5 and an indel cost of 2, to compute dissimilarity matrices for the simulation study. This measure with such specified parameters is the most neutral with regard to sensitivity to timing, duration, and order (located nearest to the center of Figure 1, in Studer and Ritschard 2016:500). Using this distance measure, our simulation minimizes the influences of timing, duration, and order; the difference between two sets of sequences would be driven almost entirely by their distinctive

---


## Page 11

&lt;page_number&gt;54&lt;/page_number&gt;
<header>Sociological Methodology 51(1)</header>

&lt;img&gt;Figure 1. Boxplots assessing difference between two groups of simulated sequences over 1,000 resamples: state occurrence (distance = OMspell, e = 0.5, i = 2, sm = indels). Note: Dashed lines mark -2, 2, 6, and 10 in the first panel and 3.81 in the second panel. BIC = Bayesian information criterion; LR = likelihood ratio.&lt;/img&gt;

states. For the distance measure for assessing timing, we chose Hamming distance, which is located toward the outer extreme in the timing dimension of Studer and Ritschard’s (2016:500) Figure 1. Similarly, for duration, we chose OMstran (w = 0.5, sm = indelslog), and for order, SVRspell (a = 1 and b = 0), which are located at or near the outer extremes of the duration and order dimensions, respectively, in Studer and Ritschard’s Figure 1.

To test the sensitivity of how the two groups differ, we randomly allocated 0, 2, 4, 6, 8, and 10 percent of the sequences containing group 2 characteristics into group 2 sequences that otherwise contain group 1 features. Thus, in the first simulation of the two groups at 0 percent, groups 1 and 2 both contain entirely group 1 features. In the simulation of the two groups at 10 percent, group 1 contains entirely group 1 features,

---


## Page 12

<header>Liao and Fasang</header>
&lt;page_number&gt;55&lt;/page_number&gt;

and 10 percent of group 2 contains group 2 features. This 0 to 10 percent range is applied to the assessment of occurrence, timing, and duration. For order, we adopted the range of 0 to 100 percent because, as they have only three possible states, our sequences are relatively short, and order difference is less sensitive to the criterion specific in Table 2 than are the other three types. We have three objectives for the simulation: (1) we expect to see difference as measured by a statistic—be it the BIC, the LRT, or the pseudo-$R^2$—to increase, with the two groups differing more with increasing proportions of group-specific features; (2) we are interested in identifying how sample size may affect each statistic; and (3) we wish to see how these statistics compare in their performance in terms of their relative sensitivity to increasing proportional differences between the two groups.

We performed this operation over the combined sample of equal sizes of the two groups with $N = 100, 200, 500, 1,000$, and 2,000, all with equal sequence length of 100. We simulated each of the percent mixtures of the two state sets at each of the sample sizes 1,000 times to compute BIC differences, LRT statistics, and pseudo-$R^2$ statistics from a discrepancy analysis. The simulation results regarding sensitivity to state occurrence, timing, duration, and order are summarized in Figures 1, 2, 3, and 4, respectively.

The figures all contain three panels of boxplots. The top panel displays BIC differences, the middle panel shows LRT statistics, and the bottom panel has pseudo-$R^2$ statistics. Each panel contains 30 boxplots, arranged first by five sample sizes and then within the same sample size, by the degree of mixing of the two sets of states (i.e., the percentage of group 2 sequences actually containing group 2-specific features, as defined by Table 2). The red dashed lines in the top panel provide the reference lines for interpreting BIC differences (see Table 1, including possible negative values within the lower bound). For example, a BIC difference of at least 2 gives positive evidence for the model treating the groups as different, a value of at least 6 provides strong evidence supporting the model of separate groups, and so on. In contrast, a negative value of at least 2 gives positive evidence for the model treating the two groups as no different, and so on (see note 3). The red reference line in the middle panel marks the chi-square statistic threshold value of 3.841 (d.f. = 1 for testing two groups vs. one). The pseudo-$R^2$ statistics are obtained from a discrepancy analysis.

We can make several general observations from the boxplots in Figures 1, 2, 3, and 4. First, as intended, with increased mixing proportions, BIC, LRT, and pseudo-$R^2$ statistics all show an increasing difference between the two groups. That is, when group 2 contains 10 percent of the sequences with uniquely group 2 features, a given statistic gives a greater value than when group 2 contains 8 percent of the sequences with uniquely group 2 features, which in turn has statistics of greater value than when group 2 contains 6 percent of the sequences with uniquely group 2 features, and so on. As noted earlier in our definition for the simulation of group-specific orders of states of just three letters, we must vary the range of mixtures from 0 to 100 percent to produce sensitivity similar in range to testing the other trajectory features (i.e., occurrence, timing, and duration).

Second, all three statistics simulated are a function of sample size, but in a different way. The BICs and LRTs tend to increase with sample size, whereas pseudo-$R^2$ values

---


## Page 13

&lt;page_number&gt;56&lt;/page_number&gt;
<header>Sociological Methodology 51(1)</header>

&lt;img&gt;
    <caption>Figure 2. Boxplots assessing difference between two groups of simulated sequences over 1,000 resamples: state timing (distance = HAM).
Note: Dashed lines mark -2, 2, 6, and 10 in the first panel and 3.841 in the second panel. BIC = Bayesian information criterion; LR = likelihood ratio.</caption>
&lt;/img&gt;

tend to decrease. However, it is important to note the starting values for the situations when the two groups do not differ (i.e., 0 percent). The BICs and LRTs both have an identical starting value for the 0 percent difference between the two groups regardless of sample size variations, whereas pseudo-R² values tend to have a greater starting value for smaller sample sizes. This property does not appear to be desirable because, when the two groups are identical, a statistic ideally should not display a different value when sample size increases. Specifically, the pseudo-R² values run the risk of suggesting meaningful group differences in small samples, even when groups are identical.

Third, because these statistics can be a function of sample size, we must exercise caution when comparing sets of sequences of different sample sizes. For timing

---


## Page 14

<header>Liao and Fasang</header>
&lt;page_number&gt;57&lt;/page_number&gt;

&lt;img&gt;
A figure containing three boxplots assessing the difference between two groups of simulated sequences over 1,000 resamples. The x-axis is labeled "Percentage of Group 2 Containing Group 2 Features" and ranges from 0 to 10. The y-axis of the top plot is "BIC Difference" and ranges from 0 to 80. The y-axis of the middle plot is "LR Statistic" and ranges from 0 to 100. The y-axis of the bottom plot is "Pseudo-R Square" and ranges from 0.00 to 0.04. Each plot has four sets of boxplots corresponding to sample sizes N=100, N=200, N=500, and N=2,000. Dashed lines are present in the first and second panels.
&lt;/img&gt;

**Figure 3.** Boxplots assessing difference between two groups of simulated sequences over 1,000 resamples: state duration (distance = OMstran, otto = 0.5, sm = indelslog).
Note: Dashed lines mark -2, 2, 6, and 10 in the first panel and 3.841 in the second panel. BIC = Bayesian information criterion; LR = likelihood ratio.

(Figure 2), the second sample size (i.e., 200 for combined groups or 100 for each group) captures a wide range of BIC values covering the different levels of statistical evidence. For duration and order (Figures 3 and 4), the sample size that covers a range of sensitivity appears to be between 100 and 200. On the basis of these simulations, we suggest that for practical considerations, in balance, a sequence data analyst may want to choose a sample size of 100 in each group for comparing trajectory groups.

In summary, the simulation study supports that (1) BICs, LRTs, and pseudo-R² values all perform well in terms of increasing with increased mixing proportions between the two groups; (2) they are all sensitive to sample size, although in different ways for BICs and LRTs compared with pseudo-R²s; and (3) it is therefore advisable to compare

---


## Page 15

&lt;page_number&gt;58&lt;/page_number&gt;
<header>Sociological Methodology 51(1)</header>

&lt;img&gt;
<caption>Figure 4. Boxplots assessing difference between two groups of simulated sequences over 1,000 resamples: state order (distance = SVRspell, a = 1, b = 0)
Note: Dashed lines mark -2, 2, 6, and 10 in the first panel and 3.841 in the second panel. BIC = Bayesian information criterion; LR = likelihood ratio.</caption>
&lt;/img&gt;

groups of bootstrapped samples of about 100 sequences from each group, regardless of the statistics used for such a comparison, using the method described later.

One may question whether this LRT statistic follows a chi-square distribution. Unlike for testing normality, there is no simple way to test the difference between an empirical and a theoretical chi-square distribution. To deal with this issue, we rely on the basic statistical property that when d.f. = 1, $\chi^2 = Z^2$ where Z is the standard normal variable. Therefore, sqrt(LRT) should follow a normal distribution if LRT follows a chi-square distribution. Appendix Figure A2 presents the QQ plots for the OMstran and SVRspell simulation results when N = 2,000 (or 1,000 in each group) for the six levels of shared similarities between the two groups. We use the largest sample size here to achieve stability and asymptotic property. The QQ normal plots show that

---


## Page 16

Liao and Fasang
&lt;page_number&gt;59&lt;/page_number&gt;

almost all the LRTs closely follow a chi-square distribution. Only the first-row plots show a slight curvature, but these LRT values are much below the $p = .05$ threshold, so they should not affect any decisions on significance tests.

**A Practical Guide for Comparing Groups of Sequences Using BICs and LRTs**

On the basis of the simulation results, we propose to compare two groups of sequences with sizes $n_1$ and $n_2$ in the following way. The objective is to compute the BIC and LRT using all sequences from both groups in computations of 100 sequences from each group. Let us assume $n_1$ has more sequences than $n_2$.

1. When $n_1 > 100$, we round up $n_1$ to the next multiple of 100, say, $k \times 100$, where $k$ is an integer (e.g., for $n_1 = 567$, $k = 6$).
2. We draw $k-1$ random samples of size 100 without replacement from the original $n_1$ sequences.
3. For the $k$th sample of 100 sequences, the sequences not previously drawn are now used, plus some additional sequences sampled with replacement (e.g., when $n_1 = 567$, the sixth sample includes the 67 sequences not drawn before, plus 33 resampled ones with replacement from the $n_1$ sequences).
4. For the $n_2$ sequences, we do the same except we draw $k-m$ random samples of size 100 without replacement, where $m$ is the difference between $n_1$ and $n_2$ as multiples of 100 before drawing $m$ samples with replacement. That is, sampling with replacement may start sooner for the second group of sequences.
5. The two random samples of 100 from the first and second groups are included in the computation of BICs and LRTs.
6. The resulting BICs and LRTs are averaged over the $k$ samples.

With this procedure, we compute the BIC and LRT using the 100 sequences in computations, with all sequences used at least once, for a total sample size as close to the original size as possible. The R program implementing the method presented in this section is publicly available as the seqCompare program (with the two functions of seqBIC and seqLRT) as part of the TraMineRextras package on CRAN.

**A Comparison with Discrepancy Analysis**

To compare the BIC and LRT performances with those based on pseudo-F ratios and pseudo-$R^2$ values from a discrepancy analysis, we conducted another round of simulations using the trajectory features for timing differences between the two groups (Table 2) and the Hamming distance measure, essentially the same simulation definition as for Figure 2. Because the difference between the pseudo-$R^2$ values and the BICs/LRTs is fairly consistent across different distance measures, as shown earlier, we focus on only one measure, for the sake of space. We ran the simulation 1,000 times, each of which included 1,000 permutations for computing $p$ values for pseudo-F ratios and pseudo-$R^2$ values. Figure 5 presents the simulation results (because the $p$ values are identical for the pseudo-F and pseudo-$R^2$, we include only one set here).

---


## Page 17

&lt;page_number&gt;60&lt;/page_number&gt;
<header>Sociological Methodology 51(1)</header>

&lt;img&gt;
A figure containing three boxplots arranged vertically.
The top plot is titled "Pseudo-F Ratio" and shows a y-axis ranging from 0 to 40. The x-axis is labeled "Percentage of Group 2 Containing Group 2 Features" and ranges from 0 to 10. There are five sets of boxplots corresponding to sample sizes N=100, N=200, N=500, N=1,000, and N=2,000. The median values for the Pseudo-F Ratio increase with both the percentage of features and the sample size.
The middle plot is titled "Pseudo-R Square" and shows a y-axis ranging from 0 to 0.08. The x-axis is the same as the top plot. The median values for the Pseudo-R Square also increase with both the percentage of features and the sample size, but the overall values are much smaller than the Pseudo-F Ratio.
The bottom plot is titled "P-Value for Pseudo-R2 or F" and shows a y-axis ranging from 0.0 to 1.0. The x-axis is the same as the other plots. The median values for the P-Value decrease with both the percentage of features and the sample size. A red dashed line is drawn at the y-value of 0.05.
&lt;/img&gt;

**Figure 5.** Boxplots assessing difference between two groups of simulated sequences over 1,000 resamples: pseudo-F ratios, pseudo-R² values, and p values from 1,000 permutations for each of the resamples (distance = HAM).
Note: Dashed lines mark .001, .01, and .05 in the second and fourth panels.

Comparing Figure 5 and Figure 2, which contains BICs/LRTs, we make the following observations. First, the pseudo-F ratios, more so than the pseudo-R² values, display a similar increasing trend with sample size with the starting value for the 0 percent difference between the two groups at a close to minimum value (the pseudo-R² values, in contrast, tend to start at a higher value for smaller sample sizes). We also see two differences between these statistics and the BICs/LRTs: the discrepancy statistics show a

---


## Page 18

Liao and Fasang
&lt;page_number&gt;61&lt;/page_number&gt;

slightly greater nonlinear trend within each sample size than do the BICs/LRTs, and their distributions tend to be more skewed, judged by the shape of the boxplots. The skewness may more easily lead to a biased result or conclusion if one does not use a constant sample size of 100, as proposed earlier.

Second, judged by their p values in the bottom panel, for smaller sample sizes, the discrepancy statistics are not sensitive enough because at all mixture levels, the boxes are above the .05 level for the combined N = 100. In comparison, the BICs have values above 2 with 8 percent differences, and the LRTs are associated with the .05 probability level with approximately 6 percent differences. The reverse is true when the sample size is large: the discrepancy statistics produced a p value greater than .05 for simulations with 2 percent differences or less, whereas for the BICs and LRTs, we find no difference only for the 0 percent. Third, if we use the previously favored sample size of 100 in each group, then the BICs provide a more nuanced gradation in levels of statistical evidence, with the 6, 8, and 10 percent groupwise difference above the 2, 6, and 10 benchmarks, respectively. Instead, for the discrepancy statistics, the difference must be 10 percent or greater to be statistically significant at the .05 level.

In summary, discrepancy statistics and the BICs and LRTs have different performances regarding sample size. The statistical significance of discrepancy analysis tends to work better with larger sample sizes, and BICs and LRTs seem to work better with smaller sample sizes. The BICs and LRTs, however, have at least two advantages: they tend to be less likely to take on extreme values (on the basis of less skewed simulation results), and they can be more nuanced, especially for medium sample sizes (100 or so in each group), by providing statistical evidence at different levels of support. In addition, they are not at risk of inflating group differences for small sample sizes, as seems to be the case for discrepancy analysis.

## EXAMPLE APPLICATION TO EMPLOYMENT LIFE-COURSES IN EAST GERMANY AND WEST GERMANY

We test the applicability of the proposed methods with an example application to employment life-courses in East Germany and West Germany. After World War II, Germany was divided into the communist German Democratic Republic (GDR) in the East and the democratic social market economy in the Federal Republic of Germany (FRG) in the West (Diewald, Goedicke, and Mayer 2006). With German reunification in 1990, the West German institutional model was installed in the East and set the former GDR on a difficult path of transition to a democratic market economy. The reunification is often framed as a quasi-experiment (e.g., Fasang 2014; Schnettler and Klüsener 2014; Struffolino et al. 2016). This is problematic for several reasons. The reunification was not a clear stimulus, as many changes occurred simultaneously; East Germany and West Germany before reunification were not static ideal types but changed considerably over time (Huinink et al. 1995; Trappe 1995); and temporary “transition” effects are difficult to separate from longer term adaptations. Instead of simply comparing two “before” and “after” periods around the German reunification,

---


## Page 19

&lt;page_number&gt;62&lt;/page_number&gt;
Sociological Methodology 51(1)

&lt;img&gt;Figure 6. Lexis diagram of life-courses ages 15 to 40 of the study cohorts (1944–1971), grouped into six cohort groups and placed in historical time (1959–2010). Note: Red vertical line marks German reunification in 1990.&lt;/img&gt;

we use the proposed methods to compare multiple birth cohorts as they move through different phases of the GDR, FRG, and reunified Germany at specific ages.

The lexis diagram in Figure 6 illustrates life-courses between ages 15 and 40 years of our study cohorts born 1944 to 1971. The oldest cohort groups, born 1944 to 1949 and 1950 to 1953, experienced almost all of their education, labor market entry, and career building in divided Germany. In contrast, the two youngest cohorts, born after 1962, went through most of these life-course stages in reunified Germany.

The GDR

In the GDR, centrally planned employment was virtually universal, but wages were low (Mayer 2006). Despite the communist ideology, the GDR was not a classless or egalitarian society, but inequality in education and income was lower, with a higher proportion of lower class jobs compared with the FRG (Solga 1995, 2006). Universal public childcare enabled high female employment, about 90 percent (Huinink et al. 1995). The GDR changed considerably over time, creating cohort-specific opportunity structures for early careers (Huinink et al. 1995:14–17). After the immediate postwar years, the socialist state was gradually installed during the 1950s. Many highly educated East Germans emigrated until the building of the Berlin Wall in 1961. Those who remained in the East were disproportionally from lower class backgrounds. The two oldest cohort groups, born 1944 to 1949 and 1950 to 1953, completed education and transitioned into the labor market in relatively favorable conditions of economic

---


## Page 20

<header>Liao and Fasang</header>
&lt;page_number&gt;63&lt;/page_number&gt;

decentralization and improved educational/job opportunities in the 1960s (Solga 1995). These opportunities ceased to exist in the 1970s with a “return to centralization,” falling job mobility, strong restrictions on access to higher education (Blossfeld, Blossfeld, and Blossfeld 2015), and a massive expansion of pronatalist family policies to counter falling birth rates (Trappe 1995). The younger cohorts, born 1954 to 1957 and 1958 to 1961, therefore faced less favorable labor market entry conditions (see Figure 6). The economic stagnation and growing public discontent of the 1980s foreshadowed the decline of the GDR. Social upward mobility and even status maintenance became increasingly difficult, given the restrictions on higher education and a lack of vacancies in leadership positions in the political establishment (Huinink et al. 1995). Declining social mobility, supply shortages, economic inefficiency, and favoritism among the party elites contradicted the promises of socialism and undermined the communist system (Mayer and Solga 1994).

### The FRG

In the postwar period, the FRG installed a parliamentary democracy, market economy, and a corporatist conservative welfare state (Esping-Andersen 1990). A segmented labor market protected male breadwinners with generous wages and benefits, but made it difficult for outsiders, including women, to enter core protected employment (Hall and Soskice 2011). Female employment was low and often part-time (Brückner 2004). Joint taxation of married couples and a lack of childcare for children under age 3 discouraged women’s employment (Cooke 2011). Although not as extensively as the East, the West also changed over time. Economic growth with the “Wirtschaftswunder” in the 1950s and 1960s ensured close to full employment. Men born 1944 to 1949 and 1950 to 1953 (Figure 6) entered into a booming economy, whereas women mostly remained out of the labor force as mothers and homemakers (Sørensen and Trappe 1995). The oil crises in 1973 and 1979 triggered an economic downturn and restructuring. With skill-biased technological change and outsourcing of the production sector, unemployment increased to about 10 percent in the 1980s, particularly affecting cohorts born 1954 to 1957 and 1958 to 1961 (Biemann, Fasang, and Grunow 2011; Kurz, Hillmert, and Grunow 2006). Educational expansion since the 1970s increased the number and diversity of students with tertiary education for cohorts born after 1950 (Blossfeld et al. 2015). And in the wake of a strong student movement to strengthen civil and minority rights and a slow expansion of public childcare and family leave options in the 1960s, female labor force participation slowly increased (Sørensen and Trappe 1995). Starting in the 1980s, the generous conservative welfare state was no longer tenable, leading to welfare retrenchment that peaked during the 1990s and coincided with the economic recession following reunification.

### Reunified Germany

Insecurity, restructuring, and an economic downturn marked the immediate reunification period in the East. In 2008, 18 years after reunification, the former East Germany still had significantly lower rates of property ownership, lower average earnings, and

---


## Page 21

&lt;page_number&gt;64&lt;/page_number&gt;
<header>Sociological Methodology 51(1)</header>

higher rates of female employment (Goldstein and Kreyenfeld 2011; Matysiak and Steinmetz 2008). The incentives of West German institutions and policies operated less effectively given different compositional features in the East. For instance, joint taxation of spouses set few incentives for marriage for couples with similar earnings, which remained much more common in the East (Bastin, Kreyenfeld, and Schnor 2012).

### Expectations

How will employment life-courses converge or diverge around reunification given these contextual and compositional differences? On the basis of modernization theory, scholars initially assumed *quick convergence* of postcommunist societies to the Western path of modernity (Fukuyama 2006; Mau and Zapf 1998; Zapf and Mau 1993). The counter-argument of *hybrid and diverging modernization* (Eisenstadt 2000; Levitt 1993; Tilly 1984) assumed that East Germany would take an enduringly different path from its Western counterpart (Nauck, Schneider, and Tölke 1995). In line with the life-course paradigm (Alwin and McCammon 2003; Elder et al. 2003), the unique experiences in divided Germany would continue to affect cohorts born during the division after reunification (Schneider, Naderi, and Ruppenthal 2012). Convergence of institutions does not necessarily translate to micro-level similarity of life-courses, and different types of institutional change could affect specific features of the life-course in different ways. For example, some institutions might affect the timing when individuals experience certain transitions, whereas others could establish new life stages (e.g., the introduction of parental leave or sudden appearance of unemployment), thereby altering the typical duration and order of life-course states. Life-course theory of social change via cohort replacement (Alwin and McCammon 2003; Mayer 1990, 2004; Ryder 1985) emphasizes the specificity of given sociohistorical locations for individual life-courses: “individuals have continuity over time to a degree that social structures do not” (Abbott 2016:5).

Modernization theory would predict divergence for cohorts who experienced most of their early adult employment lives in divided Germany and convergence for those who experienced most of their lives after reunification. Note that short-term transitional shocks with continued divergence could still be compatible with this theory. In contrast, patterns that deviate from divergence followed by convergence support the idea that social change happens at a slower pace and is both enabled and limited by the broader sociohistorical location of different birth cohorts.

### DATA

We use the first wave of the German NEPS (starting cohort 6) (Leopold, Skopek, and Raab 2011), which contains retrospective life-course information for 11,649 individuals born between 1944 and 1986 who were first surveyed in 2009 and 2010. We do not use later available waves, to avoid issues of panel attrition. The survey instruments cover detailed questions on education, work, and work interruptions measured in monthly intervals. We examine employment life-courses from ages 15 to 40, starting with the oldest available cohort (born 1944) until 1971, which is the last birth cohort

---


## Page 22

<header>Liao and Fasang</header>
&lt;page_number&gt;65&lt;/page_number&gt;

<table>
  <caption>Table 3. Sample Sizes by Region, Gender, and Cohort</caption>
  <thead>
    <tr>
      <th rowspan="2">Sample</th>
      <th rowspan="2">Cohort</th>
      <th colspan="2">All</th>
      <th colspan="2">Men</th>
      <th colspan="2">Women</th>
    </tr>
    <tr>
      <th>N</th>
      <th>%</th>
      <th>N</th>
      <th>%</th>
      <th>N</th>
      <th>%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Total</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td></td>
      <td>1944–1949</td>
      <td>932</td>
      <td>14</td>
      <td>492</td>
      <td>15</td>
      <td>440</td>
      <td>13</td>
    </tr>
    <tr>
      <td></td>
      <td>1950–1953</td>
      <td>865</td>
      <td>13</td>
      <td>446</td>
      <td>14</td>
      <td>419</td>
      <td>12</td>
    </tr>
    <tr>
      <td></td>
      <td>1954–1957</td>
      <td>1,051</td>
      <td>16</td>
      <td>490</td>
      <td>15</td>
      <td>561</td>
      <td>16</td>
    </tr>
    <tr>
      <td></td>
      <td>1958–1961</td>
      <td>1,344</td>
      <td>20</td>
      <td>624</td>
      <td>19</td>
      <td>720</td>
      <td>21</td>
    </tr>
    <tr>
      <td></td>
      <td>1962–1965</td>
      <td>1,313</td>
      <td>19</td>
      <td>603</td>
      <td>19</td>
      <td>710</td>
      <td>20</td>
    </tr>
    <tr>
      <td></td>
      <td>1966–1971</td>
      <td>1,240</td>
      <td>18</td>
      <td>579</td>
      <td>18</td>
      <td>661</td>
      <td>19</td>
    </tr>
    <tr>
      <td>Total</td>
      <td></td>
      <td>6,754</td>
      <td>100</td>
      <td>3,234</td>
      <td>100</td>
      <td>3,511</td>
      <td>100</td>
    </tr>
    <tr>
      <td>West</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td></td>
      <td>1944–1949</td>
      <td>787</td>
      <td>14</td>
      <td>420</td>
      <td>16</td>
      <td>367</td>
      <td>13</td>
    </tr>
    <tr>
      <td></td>
      <td>1950–1953</td>
      <td>677</td>
      <td>12</td>
      <td>350</td>
      <td>13</td>
      <td>327</td>
      <td>11</td>
    </tr>
    <tr>
      <td></td>
      <td>1954–1957</td>
      <td>843</td>
      <td>15</td>
      <td>388</td>
      <td>14</td>
      <td>455</td>
      <td>16</td>
    </tr>
    <tr>
      <td></td>
      <td>1958–1961</td>
      <td>1,119</td>
      <td>20</td>
      <td>523</td>
      <td>20</td>
      <td>596</td>
      <td>21</td>
    </tr>
    <tr>
      <td></td>
      <td>1962–1965</td>
      <td>1,102</td>
      <td>20</td>
      <td>510</td>
      <td>19</td>
      <td>592</td>
      <td>20</td>
    </tr>
    <tr>
      <td></td>
      <td>1966–1971</td>
      <td>1,050</td>
      <td>19</td>
      <td>491</td>
      <td>18</td>
      <td>559</td>
      <td>19</td>
    </tr>
    <tr>
      <td>Total</td>
      <td></td>
      <td>5,578</td>
      <td>100</td>
      <td>2,682</td>
      <td>100</td>
      <td>2,896</td>
      <td>100</td>
    </tr>
    <tr>
      <td>East</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td></td>
      <td>1944–1949</td>
      <td>145</td>
      <td>12</td>
      <td>72</td>
      <td>13</td>
      <td>73</td>
      <td>12</td>
    </tr>
    <tr>
      <td></td>
      <td>1950–1953</td>
      <td>188</td>
      <td>16</td>
      <td>96</td>
      <td>17</td>
      <td>92</td>
      <td>15</td>
    </tr>
    <tr>
      <td></td>
      <td>1954–1957</td>
      <td>208</td>
      <td>18</td>
      <td>102</td>
      <td>18</td>
      <td>106</td>
      <td>17</td>
    </tr>
    <tr>
      <td></td>
      <td>1958–1961</td>
      <td>225</td>
      <td>19</td>
      <td>101</td>
      <td>18</td>
      <td>124</td>
      <td>20</td>
    </tr>
    <tr>
      <td></td>
      <td>1962–1965</td>
      <td>211</td>
      <td>18</td>
      <td>93</td>
      <td>17</td>
      <td>118</td>
      <td>19</td>
    </tr>
    <tr>
      <td></td>
      <td>1966–1971</td>
      <td>190</td>
      <td>16</td>
      <td>88</td>
      <td>16</td>
      <td>102</td>
      <td>17</td>
    </tr>
    <tr>
      <td>Total</td>
      <td></td>
      <td>1,167</td>
      <td>100</td>
      <td>552</td>
      <td>100</td>
      <td>615</td>
      <td>100</td>
    </tr>
  </tbody>
</table>

observed until age 40 in the retrospective data (Figure 6). Examining sequences until age 40 is important to ensure that individuals have reached occupational maturity (Aisenbrey and Brückner 2008) and to identify delay patterns, where some cohorts go through a similar order and sequencing as others, but later and at a slower pace. We have complete employment information from ages 15 to 40 for 5,578 West Germans and 1,167 (former) East Germans. Birth cohorts are organized in the following six groups: 1944 to 1949, 1950 to 1953, 1954 to 1957, 1958 to 1961, 1962 to 1965, and 1966 to 1971 (see Figure 6). Table 3 gives a descriptive overview of case numbers for subsamples by region, gender, and cohort.

Employment sequences are operationalized with different states of being in and out of the labor force and using the Erikson-Goldthorpe-Portocarero (EGP) occupational classification (Erikson, Goldthorpe, and Portocarero 1979). We specify 13 employment states, including “out of the labor force/gap year,” “unemployment,” “military,” “education,” and “parental leave.” The employed state is disaggregated into the EGP classes of “higher grade professionals,” (I) “lower grade professionals,” (II) “routine non-manual employees, higher grade,” (IIIa) “routine non-manual employees, lower grade,” (IIIb) “small proprietors, farmers,” (IVa, IVb, and IVc) “lower grade technicians and supervisors of manual workers,” (V) “skilled manual workers,” (VI) and “manual workers in primary production” (VIIa and VIIb). Like all social class

---


## Page 23

&lt;page_number&gt;66&lt;/page_number&gt;
<header>Sociological Methodology 51(1)</header>

schemes, the EGP has limitations. Most measures tend to overstate women’s position in society, as typically female jobs are assigned fairly high social class but provide low earnings. This distorts gender comparisons, but not our within-gender comparisons of East German and West German men and women. Other ways of measuring social status, such as income, are not reliably available in our data and are unsuitable for comparing social status in a state socialist system, in which income distributions were much flatter. The EGP categories correspond closely with the class scheme for the GDR developed by Solga (1995) and therefore are a reasonably reliable indicator of class positions in the GDR and the FRG.

To ensure a rigorous East-West comparison, the East and West samples include only respondents who were born in the respective region and still living there at the time of the interview (2009 and 2010). We exclude foreign-born respondents, individuals who migrated between the East and West, and respondents living in Berlin, because the survey did not distinguish between the former East and West regions of Berlin at the time of the interview. The data are weighted using a calibrated design weight that combines a sampling design weight and a calibration factor (multiplier) to reflect the German Microcensus 2009 (Aßmann and Zinn 2011).

## RESULTS AND DISCUSSION

In the example application, G number of groups equals 2 (East vs. West), and the BIC and LRT are calculated for all combinations of gender and cohort groups. We follow the guidelines for the proposed BIC and LRT statistics to the region-gender-cohort-specific samples by taking subregion-gender-cohort samples of size 100 and guaranteeing each trajectory in a subsample is selected, per the descriptions in the guidelines section. For the simulation study, we use the Hamming distance to capture timing, the SVRspell distance (subsequence kweight = 1 and exponential weight of spell length = 0) for the order of states, and OMstran (w = 0.5, sm = indelslog) to measure differences in the duration in life-course states (Elzinga and Studer 2015; Studer and Ritschard 2016).

Table 4 shows BIC differences and LRTs for the three different distance measures. Overall, the BIC differences range between −1.176 and 11.676, with many comparisons showing small differences between −1 and 1. Yet most comparisons based on the LRT are statistically significant. This underlines the added value of complementing the assessment of statistical significance with a measure for the level of evidence, so as not to overstate the social and substantive significance of small, statistically significant group differences. We therefore focus on levels of BIC difference rather than statistical significance in the following interpretation. Note that our computation of the East-West comparison of the entire sample, gender specific or not, followed the guidelines we suggested earlier, because using the entire samples directly would yield unreasonably high values of BICs and LRTs even though combining the cohorts should dampen the regional differences.⁴

Table 4 also presents pseudo-R² statistics from a discrepancy analysis. The pseudo-R² statistics mostly, but not consistently, follow similar trends as the BICs and LRTs:

---


## Page 24

<table>
   <thead>
    <tr>
     <td rowspan="2">
      <b>
       Measures
      </b>
     </td>
     <td colspan="3">
      <b>
       Hamming/Timing
      </b>
     </td>
     <td colspan="3">
      <b>
       SVRspell/Order
      </b>
     </td>
     <td colspan="3">
      <b>
       OMstran/Duration
      </b>
     </td>
    </tr>
    <tr>
     <td>
      <b>
       SBIC
      </b>
     </td>
     <td>
      <b>
       BF
      </b>
     </td>
     <td>
      <b>
       LRT
      </b>
     </td>
     <td>
      <b>
       P
      </b>
     </td>
     <td>
      <b>
       Pseudo-R
       <sup>
        2
       </sup>
      </b>
     </td>
     <td>
      <b>
       SBIC
      </b>
     </td>
     <td>
      <b>
       BF
      </b>
     </td>
     <td>
      <b>
       LRT
      </b>
     </td>
     <td>
      <b>
       P
      </b>
     </td>
     <td>
      <b>
       Pseudo-R
       <sup>
        2
       </sup>
      </b>
     </td>
     <td>
      <b>
       SBIC
      </b>
     </td>
     <td>
      <b>
       BF
      </b>
     </td>
     <td>
      <b>
       LRT
      </b>
     </td>
     <td>
      <b>
       P
      </b>
     </td>
     <td>
      <b>
       Pseudo-R
       <sup>
        2
       </sup>
      </b>
     </td>
    </tr>
   </thead>
   <tbody>
    <tr>
     <td>
      <b>
       Group
      </b>
     </td>
     <td>
     </td>
     <td>
     </td>
     <td>
     </td>
     <td>
     </td>
     <td>
     </td>
     <td>
     </td>
     <td>
     </td>
     <td>
     </td>
     <td>
     </td>
     <td>
     </td>
     <td>
     </td>
     <td>
     </td>
     <td>
     </td>
     <td>
     </td>
     <td>
     </td>
    </tr>
    <tr>
     <td>
      Total
     </td>
     <td>
      −600
     </td>
     <td>
      .866
     </td>
     <td>
      4.698
     </td>
     <td>
      .0364
     </td>
     <td>
      .013
     </td>
     <td>
      −.2714
     </td>
     <td>
      1.088
     </td>
     <td>
      5.027
     </td>
     <td>
      .0326
     </td>
     <td>
      .012
     </td>
     <td>
      3.115
     </td>
     <td>
      9.050
     </td>
     <td>
      8.413
     </td>
     <td>
      .0078
     </td>
     <td>
      .013
     </td>
    </tr>
    <tr>
     <td>
      Men
     </td>
     <td>
      .535
     </td>
     <td>
      1.873
     </td>
     <td>
      5.833
     </td>
     <td>
      .0227
     </td>
     <td>
      .019
     </td>
     <td>
      −.034
     </td>
     <td>
      1.593
     </td>
     <td>
      5.2643
     </td>
     <td>
      .0349
     </td>
     <td>
      .014
     </td>
     <td>
      3.586
     </td>
     <td>
      13.274
     </td>
     <td>
      8.885
     </td>
     <td>
      .0056
     </td>
     <td>
      .018
     </td>
    </tr>
    <tr>
     <td>
      Women
     </td>
     <td>
      .74
     </td>
     <td>
      1.607
     </td>
     <td>
      6.038
     </td>
     <td>
      .0162
     </td>
     <td>
      .015
     </td>
     <td>
      1.8044
     </td>
     <td>
      3.475
     </td>
     <td>
      7.1027
     </td>
     <td>
      .0124
     </td>
     <td>
      .018
     </td>
     <td>
      4.974
     </td>
     <td>
      18.827
     </td>
     <td>
      10.272
     </td>
     <td>
      .0022
     </td>
     <td>
      .017
     </td>
    </tr>
    <tr>
     <td>
      Men
     </td>
     <td>
      1944–1949
     </td>
     <td>
      −831
     </td>
     <td>
      .813
     </td>
     <td>
      4.467
     </td>
     <td>
      .0430
     </td>
     <td>
      .013
     </td>
     <td>
      .2885
     </td>
     <td>
      1.183
     </td>
     <td>
      5.5868
     </td>
     <td>
      .0186
     </td>
     <td>
      .013
     </td>
     <td>
      −1.176
     </td>
     <td>
      1.4737
     </td>
     <td>
      4.122
     </td>
     <td>
      .0968
     </td>
     <td>
      .013
     </td>
    </tr>
    <tr>
     <td>
     </td>
     <td>
      1950–1953
     </td>
     <td>
      2.639
     </td>
     <td>
      11.209
     </td>
     <td>
      7.937
     </td>
     <td>
      .0143
     </td>
     <td>
      .026
     </td>
     <td>
      1.5021
     </td>
     <td>
      3.187
     </td>
     <td>
      6.8004
     </td>
     <td>
      .0137
     </td>
     <td>
      .020
     </td>
     <td>
      5.441
     </td>
     <td>
      150.252
     </td>
     <td>
      10.739
     </td>
     <td>
      .0057
     </td>
     <td>
      .025
     </td>
    </tr>
    <tr>
     <td>
     </td>
     <td>
      1954–1957
     </td>
     <td>
      2.914
     </td>
     <td>
      6.683
     </td>
     <td>
      8.212
     </td>
     <td>
      .0078
     </td>
     <td>
      .025
     </td>
     <td>
      .9383
     </td>
     <td>
      2.267
     </td>
     <td>
      6.2366
     </td>
     <td>
      .0181
     </td>
     <td>
      .020
     </td>
     <td>
      6.662
     </td>
     <td>
      135.32
     </td>
     <td>
      11.961
     </td>
     <td>
      .0024
     </td>
     <td>
      .025
     </td>
    </tr>
    <tr>
     <td>
     </td>
     <td>
      1958–1961
     </td>
     <td>
      2.846
     </td>
     <td>
      5.009
     </td>
     <td>
      8.145
     </td>
     <td>
      .0056
     </td>
     <td>
      .025
     </td>
     <td>
      2.4425
     </td>
     <td>
      4.2
     </td>
     <td>
      7.7409
     </td>
     <td>
      .0068
     </td>
     <td>
      .013
     </td>
     <td>
      6.242
     </td>
     <td>
      37.339
     </td>
     <td>
      11.539
     </td>
     <td>
      .0012
     </td>
     <td>
      .024
     </td>
    </tr>
    <tr>
     <td>
     </td>
     <td>
      1962–1965
     </td>
     <td>
      1.058
     </td>
     <td>
      3.312
     </td>
     <td>
      6.356
     </td>
     <td>
      .0213
     </td>
     <td>
      .022
     </td>
     <td>
      −.7388
     </td>
     <td>
      .973
     </td>
     <td>
      4.5595
     </td>
     <td>
      .0534
     </td>
     <td>
      .013
     </td>
     <td>
      3.439
     </td>
     <td>
      23.395
     </td>
     <td>
      8.738
     </td>
     <td>
      .0098
     </td>
     <td>
      .022
     </td>
    </tr>
    <tr>
     <td>
     </td>
     <td>
      1966–1971
     </td>
     <td>
      .723
     </td>
     <td>
      1.721
     </td>
     <td>
      6.021
     </td>
     <td>
      .0188
     </td>
     <td>
      .019
     </td>
     <td>
      3.4353
     </td>
     <td>
      22.733
     </td>
     <td>
      8.7336
     </td>
     <td>
      .0096
     </td>
     <td>
      .018
     </td>
     <td>
      3.296
     </td>
     <td>
      6.015
     </td>
     <td>
      8.594
     </td>
     <td>
      .0039
     </td>
     <td>
      .018
     </td>
    </tr>
    <tr>
     <td>
      Women
     </td>
     <td>
      1944–1949
     </td>
     <td>
      2.666
     </td>
     <td>
      4.183
     </td>
     <td>
      7.965
     </td>
     <td>
      .0052
     </td>
     <td>
      .022
     </td>
     <td>
      9.0647
     </td>
     <td>
      166.547
     </td>
     <td>
      14.363
     </td>
     <td>
      .0000
     </td>
     <td>
      .034
     </td>
     <td>
      8.229
     </td>
     <td>
      90.319
     </td>
     <td>
      13.527
     </td>
     <td>
      .0000
     </td>
     <td>
      .024
     </td>
    </tr>
    <tr>
     <td>
     </td>
     <td>
      1950–1953
     </td>
     <td>
      4.521
     </td>
     <td>
      11.316
     </td>
     <td>
      9.819
     </td>
     <td>
      .0021
     </td>
     <td>
      .025
     </td>
     <td>
      4.6544
     </td>
     <td>
      16.487
     </td>
     <td>
      9.9527
     </td>
     <td>
      .0023
     </td>
     <td>
      .028
     </td>
     <td>
      11.676
     </td>
     <td>
      430.435
     </td>
     <td>
      16.974
     </td>
     <td>
      .0000
     </td>
     <td>
      .027
     </td>
    </tr>
    <tr>
     <td>
     </td>
     <td>
      1954–57
     </td>
     <td>
      3.970
     </td>
     <td>
      10.015
     </td>
     <td>
      9.269
     </td>
     <td>
      .0036
     </td>
     <td>
      .024
     </td>
     <td>
      4.2964
     </td>
     <td>
      13.595
     </td>
     <td>
      9.5947
     </td>
     <td>
      .0056
     </td>
     <td>
      .024
     </td>
     <td>
      9.368
     </td>
     <td>
      284.436
     </td>
     <td>
      14.667
     </td>
     <td>
      .0000
     </td>
     <td>
      .026
     </td>
    </tr>
    <tr>
     <td>
     </td>
     <td>
      1958–1961
     </td>
     <td>
      2.054
     </td>
     <td>
      3.469
     </td>
     <td>
      7.352
     </td>
     <td>
      .0082
     </td>
     <td>
      .018
     </td>
     <td>
      2.4059
     </td>
     <td>
      5.667
     </td>
     <td>
      7.7042
     </td>
     <td>
      .0088
     </td>
     <td>
      .019
     </td>
     <td>
      6.432
     </td>
     <td>
      75.184
     </td>
     <td>
      11.730
     </td>
     <td>
      .0011
     </td>
     <td>
      .021
     </td>
    </tr>
    <tr>
     <td>
     </td>
     <td>
      1962–1965
     </td>
     <td>
      1.413
     </td>
     <td>
      2.191
     </td>
     <td>
      6.711
     </td>
     <td>
      .0108
     </td>
     <td>
      .017
     </td>
     <td>
      2.2449
     </td>
     <td>
      4.377
     </td>
     <td>
      7.5432
     </td>
     <td>
      .0093
     </td>
     <td>
      .020
     </td>
     <td>
      3.852
     </td>
     <td>
      7.9162
     </td>
     <td>
      9.151
     </td>
     <td>
      .0031
     </td>
     <td>
      .020
     </td>
    </tr>
    <tr>
     <td>
     </td>
     <td>
      1966–1971
     </td>
     <td>
      −586
     </td>
     <td>
      .771
     </td>
     <td>
      4.712
     </td>
     <td>
      .0312
     </td>
     <td>
      .013
     </td>
     <td>
      3.6033
     </td>
     <td>
      12.040
     </td>
     <td>
      8.9017
     </td>
     <td>
      .0083
     </td>
     <td>
      .018
     </td>
     <td>
      1.076
     </td>
     <td>
      1.958
     </td>
     <td>
      6.374
     </td>
     <td>
      .0145
     </td>
     <td>
      .014
     </td>
    </tr>
   </tbody>
  </table>
  <p>
   <i>
    Note:
   </i>
   BF = Bayes factor; BIC = Bayesian information criterion; LRT = likelihood-ratio test.
  </p>
  <p>
   &lt;page_number&gt;
    67
   &lt;/page_number&gt;
  </p>

---


## Page 25

&lt;page_number&gt;68&lt;/page_number&gt;
<header>Sociological Methodology 51(1)</header>

&lt;img&gt;
Employment, Hamming/timing
BIC
1944-49 1950-53 1954-57 1958-61 1962-65 1966-71
Cohort
Women
Men

Employment, SVRspell/order
BIC
1944-49 1950-53 1954-57 1958-61 1962-65 1966-71
Cohort
Women
Men

Employment, OMstran/duration
BIC
1944-49 1950-53 1954-57 1958-61 1962-65 1966-71
Cohort
Women
Men
&lt;/img&gt;

Figure 7. Bayesian information criterion (BIC) mean differences comparing East German and West German employment trajectories, separately for timing, order, and duration, based on cohort-region-specific samples of size 100 with total sizes equaling the original sample sizes.

they are low and quite similar for all group comparisons, ranging from .013 to .034. Therefore, unlike the BIC difference values, they do not properly discriminate between varying degrees of group differences. Note that the BIC difference is simply a function of the BF, where BF = exp(BIC/2). However, because we take the average of BICs over the k samples, as suggested in step 6 of the practical guide presented earlier, there are two ways to compute BFs: taking the average of BFs over the k samples or transforming the averaged BIC difference into the BF. Both methods are programmed in seqCompare, and the average of BFs is reported in Table 4.

Figure 7 depicts the BIC differences for the three distance measures. The solid lines show differences for men, the dotted lines differences for women. Figures 8 and 9 show state distribution plots and relative frequency (RF) sequence plots (Fasang and

---


## Page 26

<header>Liao and Fasang</header>
&lt;page_number&gt;69&lt;/page_number&gt;

&lt;img&gt;Figure 8. State distribution plots of employment sequences by cohort, East German and West German men and women. Note: LF = labor force.&lt;/img&gt;

Liao 2014) separately for each region-gender-cohort-specific group to augment the substantive interpretation of the degree of difference reflected in the BIC and their statistical significances given with the LRT statistics. Note that the RF sequence plots in Figure 9 are based on the Hamming distance. RF plots based on the SVRspell and OMstran distances are shown in Appendix Figures B1 and B2, as they provide a

---


## Page 27

&lt;page_number&gt;70&lt;/page_number&gt;
<header>Sociological Methodology 51(1)</header>

&lt;img&gt;Relative frequency plots of employment sequences by cohort, number of groupings = 75, sorted by age of first job, based on Hamming distance, East German and West German men and women.&lt;/img&gt;

Figure 9. Relative frequency plots of employment sequences by cohort, number of groupings = 75, sorted by age of first job, based on Hamming distance, East German and West German men and women.
Note: LF = labor force.

similar overall picture as those based on the Hamming distance (Figure 9). Appendix Table B1 shows the average state duration and additional sequence indicators for each gender by region by cohort group.

---


## Page 28

<header>Liao and Fasang</header>
&lt;page_number&gt;71&lt;/page_number&gt;

## Timing

BIC differences between East German and West German life-courses in terms of timing (Hamming distance, Figure 7, top panel) are moderate for all cohorts, ranging between -0.8 for men born 1944 to 1949 and 4.5 for women born 1950 to 1953 (see Table 4 and BIC threshold values in Table 1). The timing of transitions in employment lives slightly diverged for men and women from the cohorts born 1944 to 1949 and those born in the 1950s. Convergence set in much earlier than the reunification for cohorts born after 1950 and is more pronounced for women than for men. West German women’s greater employment participation, the introduction of parental leave plans, and shorter family-related interruptions already made them more similar to their Eastern counterparts during the late 1970s and 1980s (see Figures 8 and 9 and Appendix Table B1). Among cohorts born in the 1960s who experienced almost all of their employment lives in reunified Germany, BIC values are statistically significant but very low, between -0.586 and 1.4, indicating overall very similar timing in women’s employment lives.

For East German and West German men, we see moderate positive BIC differences, hovering between 2 and 3 for most cohorts. There is a notable drop from the 1958–1961 cohort (2.8) to the cohorts born 1962 to 1965 (1.0) and 1966 to 1971 (0.7), and this decline coincides with reunification (Table 4). Implementation of the West German education system in the East equalized the timing of completing various educational tracks and entering the labor market for cohorts born after 1960. Figures 8 and 9 and Appendix Table B1 further support a more similar timing of completing education and entering a first job among the youngest cohorts of East German and West German men, which is primarily driven by a delay in these transitions in the East.

## Order

The middle panel in Figure 7 shows differences in the order of employment states (SVRspell). Higher BIC difference values indicate greater differences compared with timing for women. Notable convergence for East German and West German women set in long before reunification: following the same secular trend, West German women entered the labor market in greater numbers similar to their Eastern peers. For East German and West German men, differences are low and not worth a mention, with modest fluctuations. There is a notable divergence related to the immediate transition shock in the East between the two youngest cohorts: the BIC value jumps from -0.74 to 3.4. East German men born 1966 to 1971 were hardest hit by the recession and restructuring around reunification. They experienced more volatile employment lives, with a larger number of transitions, leading to a more complex order of employment states (see Appendix Table B1). Both short-term job mobility between fixed-term positions and recurrent unemployment were more common for them compared with their Western peers (see also Figures 8 and 9 and Appendix Table B1). In particular, Figure 9 (and Appendix Figures B1 and B2) shows that among the two youngest cohorts in the East, it was common for men to first work in low-EGP class

---


## Page 29

&lt;page_number&gt;72&lt;/page_number&gt;
Sociological Methodology 51(1)

occupations and then return to education shortly after reunification, which drastically devalued their East German degrees. After retraining, a sizable proportion entered higher EGP class occupations in the westernized occupational structure, which offered fewer low EGP class positions. This order of states, low EGP-education-higher EGP, is distinct for East German men born in the late 1960s, who experienced reunification in their early and mid-20s. Similar career experiences were a rare exception among their West German peers.

### Duration

For East German and West German women, BIC differences are largest for cohorts born 1950 to 1953, at a sizable 11.6. This continuously drops to 1.076 for cohorts born 1966 to 1971. These sizable BIC differences of the earlier cohorts underline the central role of duration spent in and out of employment for the convergence of West German women's employment lives to the East German pattern of high employment and shorter parental leaves (Appendix Table B1). Postreunification unemployment among younger cohorts of East German women further aligned their employment lives with West German women's experiences (Figures 8 and 9, Appendix Table B1). For men, the duration of employment states converges around reunification but still displays some differences. Time spent in education, unemployment, and different EGP classes gradually equalized as the West German education system and occupational structure were implemented in the East.

Taken together, for East German and West German men, our findings support convergence of the timing and duration of employment states that strongly coincides with reunification and reflects installation of the West German educational system and occupational structure in the East. In contrast, we find a, possibly temporary, divergence in the order of employment states between East German and West German men. In the immediate transition period, employment lives of East German men were more unstable following a more complex order of back-and-forth movements between jobs and in and out of employment because of unemployment or retraining. For East German and West German women, employment lives converged before reunification as West German women entered the labor market in greater numbers and changing gender norms and family policies enabled them to take shorter parental leaves. This is supported by all three distance measures on timing, order, and duration, albeit to varying degrees. The largest BIC differences and strongest convergence are found for the duration spent in different employment states, with only moderate differences in the timing of transitions.

### CONCLUSION

In this article, we proposed a statistical method for group comparisons of life-course sequences to address the lack of clear-cut statistical assessment that has been a longstanding inadequacy of social sequence analysis (Piccarreta and Studer 2018). A simulation study supported the gravity-center-based BIC and LRT methods. The BIC provides a way to assess level of evidence, and the LRT further yields a statistical

---


## Page 30

Liao and Fasang
&lt;page_number&gt;73&lt;/page_number&gt;

significance, although its statistical significance tends to be a little more easily achieved than a higher level of evidence of BIC difference. The discrepancy statistics and the proposed BICs and LRTs have somewhat different advantages in terms of sample size, but we demonstrated that the proposed statistics have three advantages. First, they are not affected by sample size variations when applied as suggested by our guidelines. Second, the BICs are more nuanced by providing statistical evidence at different levels of support. And third, unlike discrepancy analysis, they do not inflate group differences for smaller sample sizes.

More generally, each of the statistics, discrepancy or the proposal here, has its own characteristics. The pseudo-F ratio is useful only when accompanied by computationally intensive permutation tests, and it may have inconclusive intervals (Studer et al. 2011). The pseudo-R² statistic can be useful, yet it often produces rather low values and thus relies on permutation tests to provide statistical significance like its F-ratio counterpart. The BIC typically favors statistical parsimony, which, in the context of group differences, means one overall group instead of two separate groups will be favored. We did not find this, however, in the last section of the simulation, in which the BIC was actually less likely to support the no-difference model than were the F-ratio or R² statistics, especially when sample sizes were small. Finally, the LRT complements the BIC by providing an exact test for a significance level. In summary, on the basis of our study, our suggestion is to use the BIC and LRT with a medium sample size of 100 and to use the procedure we provided in the practical guide subsection for sample sizes smaller or greater than 100.

Our empirical application used high-quality retrospective life-course data from the German NEPS to assess divergence and convergence of individual-level life-courses across cohorts that experienced profound macro-level changes around German reunification. We located divergence and convergence for specific birth cohorts separately in terms of the timing, order, and duration of employment states, and we highlighted gender-specific patterns of change.

We have shown that statistically comparing samples of life-course sequences is a difficult task. Comparing sets of single-value measurements such as income requires simply testing mean or log-mean differences of the two samples. Comparing single life-course transitions can be relatively straightforward too, by computing the mean difference in the timing of transitions. Complete life-course sequences, however, contain a set of complex measurements, including the timing of multiple life-course events, the implied duration of such events, and their ordering. The proposed gravity-center-based BIC statistic (and its related LRT) therefore provides a new method for adequately assessing differences between sets of sequences statistically via Jeffreys’s (1961) levels of statistical evidence as implemented through BIC differences. The proposed method proved particularly useful in combination with sophisticated sequence visualization to examine the qualitative content of life-course differences. We thus provide information on statistical significance of differences, the degree of differences in the absolute BIC value, and the qualitative content of differences through the auxiliary visualizations. The latter two allow a more informed assessment of the “social,” in addition to the statistical significance of differences between groups of sequences

---


## Page 31

&lt;page_number&gt;74&lt;/page_number&gt;
<header>Sociological Methodology 51(1)</header>

(Bernardi et al. 2017; McShane et al. 2019). As such, the assessment method holds promise for analyzing other kinds of life-course sequences (e.g., migration and health histories) as well as other types of non-life-course social sequence data.

Using the proposed method and augmented by existing sophisticated sequence visualization, we offer new insights into East German and West German life-courses before and after reunification. The lower dissimilarities and stronger convergence around reunification between East German and West German men, compared with women, suggests that because of their higher labor market integration, men were more similar initially and more moldable by a sudden change of educational and labor market institutions. Persistent gender differences in life-course trajectories may also result from the deeper differences in gender ideology between the two Germanys, as implementing institutional changes is easier and arguably quicker than effecting ideological transformations. The convergence of women’s employment lives set in much earlier than reunification. It mirrors a longer lasting ideological and institutional shift toward a more gender-egalitarian division of paid work in the West, which was already in place during communism in the East.

Seen in the light of the macro-micro link, our BIC-facilitated analysis of East German and West German individual life-course trajectories highlights the importance of laying out historical and situational social mechanisms for specific birth cohorts as they move through different historical periods at specific ages. In addition, the unique life-course experiences of each cohort remain relevant in the long term and transcend brief historical periods. The proposed method enabled us to question the standard, simple before and after comparisons of German reunification as a “natural experiment” by highlighting large variations in life-course experiences for different cohorts within divided as well as reunified Germany.

**APPENDIX A: TESTING THE EFFECT OF NONNORMAL DISTANCES**

Equation (2) relies on the feasibility of using errors or distances in likelihood computations. The equation follows from Burnham and Anderson’s (1998, 2004) expression of

<table>
  <thead>
    <tr>
      <th colspan="4">Table A1. Comparing Observed BIC Differences and BIC Differences Computed from Normalized Distances</th>
    </tr>
    <tr>
      <th>Distance</th>
      <th>Data</th>
      <th>Observed</th>
      <th>Normal</th>
      <th>% Difference</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3">Hamming</td>
      <td>west-east</td>
      <td>46.309</td>
      <td>46.281</td>
      <td>.035</td>
    </tr>
    <tr>
      <td>west-east, m</td>
      <td>29.558</td>
      <td>29.528</td>
      <td>.037</td>
    </tr>
    <tr>
      <td>west-east, f</td>
      <td>35.295</td>
      <td>35.281</td>
      <td>.016</td>
    </tr>
    <tr>
      <td rowspan="3">SVRspell</td>
      <td>west-east</td>
      <td>46.980</td>
      <td>46.640</td>
      <td>.422</td>
    </tr>
    <tr>
      <td>west-east, m</td>
      <td>18.171</td>
      <td>18.304</td>
      <td>.165</td>
    </tr>
    <tr>
      <td>west-east, f</td>
      <td>43.850</td>
      <td>43.932</td>
      <td>.102</td>
    </tr>
    <tr>
      <td rowspan="3">OMstran</td>
      <td>west-east</td>
      <td>80.627</td>
      <td>80.621</td>
      <td>.007</td>
    </tr>
    <tr>
      <td>west-east, m</td>
      <td>46.148</td>
      <td>46.167</td>
      <td>.024</td>
    </tr>
    <tr>
      <td>west-east, f</td>
      <td>62.979</td>
      <td>63.019</td>
      <td>.049</td>
    </tr>
  </tbody>
</table>

---


## Page 32

<header>Liao and Fasang</header>
&lt;page_number&gt;75&lt;/page_number&gt;

&lt;img&gt;Figure A1. Comparing observed distances and normalized distances.&lt;/img&gt;

least squares as likelihoods for constructing the closely related statistic of the BIC in the form of the Akaike information criterion, by specifying the case of “least squares (LS) estimation with normally distributed errors” (Burnham and Anderson 2004:268). However, neither these authors nor anyone else in the literature has explored if such a treatment would still be feasible if data are nonnormal.
Because distance data can be rather nonnormal, we set up a simulation to investigate the feasibility of constructing the BIC using equations (2) and (3), where the input in equation (3), $s_i$, is the distance to the gravity center. We began with the observed distance data for the overall West and East samples and the subsamples for men and women separately. We used distance measures Hamming, SVRspell, and OMstran, the ones we will use later in the empirical applications. By keeping the numbers of observations, the means, the standard deviations, and the sum distances identical to

---


## Page 33

&lt;page_number&gt;76&lt;/page_number&gt;
<header>Sociological Methodology 51(1)</header>

&lt;img&gt;
Normal QQ Plot, OMstran, 0%
Normal QQ Plot, SVRspell, 0%
Normal QQ Plot, OMstran, 2%
Normal QQ Plot, SVRspell, 20%
Normal QQ Plot, OMstran, 4%
Normal QQ Plot, SVRspell, 40%
Normal QQ Plot, OMstran, 6%
Normal QQ Plot, SVRspell, 60%
Normal QQ Plot, OMstran, 8%
Normal QQ Plot, SVRspell, 80%
Normal QQ Plot, OMstran, 10%
Normal QQ Plot, SVRspell, 100%
&lt;/img&gt;

Figure A2. Normal QQ plot of the square root of likelihood ratio test by distance measure and percentage difference.

---


## Page 34

<header>Liao and Fasang</header>
&lt;page_number&gt;77&lt;/page_number&gt;

the observed data, we generated 10,000 sets of normally distributed distances. In addition, for distance measures such as the Hamming, we generated integer values, and for metric measures, we generated metric or continuous values. In other words, these sets of normally distributed distances display all the important characteristics of the original data. Figure A1 presents the histograms of the original distance data and their normally distributed versions by taking the average of 10,000 simulated values for each observation. It is obvious that for all subsamples, the three types of distances all show skewness, especially the SVRspell distances. The simulated distances via random normal generation all display a normal-looking histogram.

Using these normally generated distance data, we further computed BIC differences and report the results in Table A1. The first column of the table contains the BICs based on the observed data, the second column contains the BICs based on simulated data, and the third column contains their absolute differences in percentage terms. It is reassuring to see that the simulated data and the original skewed data resulted in almost identical BIC differences, with the largest difference being just 0.422 percent. The simulation strongly confirms that the BIC construction is not affected by the distributional shape of the distance data. What affects a BIC difference size is just distance sizes, nothing else.

<table>
  <thead>
    <tr>
      <th></th>
      <th>BICs based on observed data</th>
      <th>BICs based on simulated data</th>
      <th>Absolute differences in percentage terms</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>SVRspell</td>
      <td>123.45</td>
      <td>123.45</td>
      <td>0.000%</td>
    </tr>
    <tr>
      <td>Hamming</td>
      <td>110.23</td>
      <td>110.23</td>
      <td>0.000%</td>
    </tr>
    <tr>
      <td>Euclidean</td>
      <td>98.76</td>
      <td>98.76</td>
      <td>0.000%</td>
    </tr>
    <tr>
      <td>Maximum</td>
      <td>150.00</td>
      <td>150.00</td>
      <td>0.000%</td>
    </tr>
    <tr>
      <td>Minimum</td>
      <td>80.00</td>
      <td>80.00</td>
      <td>0.000%</td>
    </tr>
    <tr>
      <td>Average</td>
      <td>110.00</td>
      <td>110.00</td>
      <td>0.000%</td>
    </tr>
  </tbody>
</table>

---


## Page 35

APPENDIX B

Table B1. Descriptive Information on Mean Duration and Median Age of Central Transitions by Gender and Cohort Group

<table>
  <thead>
    <tr>
      <th rowspan="2">Cohort</th>
      <th colspan="6">East Men</th>
      <th colspan="6">East Women</th>
      <th rowspan="2">West Men</th>
      <th rowspan="2">West Women</th>
    </tr>
    <tr>
      <th>All</th>
      <th>44-49</th>
      <th>50-53</th>
      <th>54-57</th>
      <th>58-61</th>
      <th>62-65</th>
      <th>66-71</th>
      <th>All</th>
      <th>44-49</th>
      <th>50-53</th>
      <th>54-57</th>
      <th>58-61</th>
      <th>62-65</th>
      <th>66-71</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>EGP I</td>
      <td>28</td>
      <td>32</td>
      <td>34</td>
      <td>27</td>
      <td>31</td>
      <td>26</td>
      <td>22</td>
      <td>26</td>
      <td>25</td>
      <td>32</td>
      <td>36</td>
      <td>26</td>
      <td>24</td>
      <td>17</td>
    </tr>
    <tr>
      <td>EGP II</td>
      <td>23</td>
      <td>34</td>
      <td>13</td>
      <td>17</td>
      <td>18</td>
      <td>23</td>
      <td>37</td>
      <td>51</td>
      <td>45</td>
      <td>61</td>
      <td>43</td>
      <td>40</td>
      <td>55</td>
      <td>39</td>
    </tr>
    <tr>
      <td>EGP IIIa</td>
      <td>4</td>
      <td>7</td>
      <td>4</td>
      <td>3</td>
      <td>3</td>
      <td>6</td>
      <td>3</td>
      <td>44</td>
      <td>55</td>
      <td>43</td>
      <td>56</td>
      <td>40</td>
      <td>38</td>
      <td>24</td>
    </tr>
    <tr>
      <td>EGP IIIb</td>
      <td>10</td>
      <td>9</td>
      <td>12</td>
      <td>3</td>
      <td>7</td>
      <td>12</td>
      <td>13</td>
      <td>28</td>
      <td>34</td>
      <td>25</td>
      <td>26</td>
      <td>25</td>
      <td>37</td>
      <td>24</td>
    </tr>
    <tr>
      <td>EGP IVa, IVb, and IVc</td>
      <td>4</td>
      <td>2</td>
      <td>.5</td>
      <td>8</td>
      <td>5</td>
      <td>9</td>
      <td>5</td>
      <td>.8</td>
      <td>.4</td>
      <td>0</td>
      <td>.2</td>
      <td>.5</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <td>EGP V</td>
      <td>17</td>
      <td>8</td>
      <td>20</td>
      <td>27</td>
      <td>24</td>
      <td>4</td>
      <td>20</td>
      <td>14</td>
      <td>21</td>
      <td>18</td>
      <td>11</td>
      <td>22</td>
      <td>15</td>
      <td>7</td>
    </tr>
    <tr>
      <td>EGP VI</td>
      <td>71</td>
      <td>60</td>
      <td>77</td>
      <td>81</td>
      <td>79</td>
      <td>74</td>
      <td>57</td>
      <td>48</td>
      <td>34</td>
      <td>45</td>
      <td>37</td>
      <td>32</td>
      <td>10</td>
      <td>27</td>
    </tr>
    <tr>
      <td>EGP VIIa and VIIb</td>
      <td>58</td>
      <td>70</td>
      <td>73</td>
      <td>68</td>
      <td>54</td>
      <td>36</td>
      <td>58</td>
      <td>65</td>
      <td>56</td>
      <td>50</td>
      <td>55</td>
      <td>56</td>
      <td>25</td>
      <td>63</td>
    </tr>
    <tr>
      <td>Education</td>
      <td>59</td>
      <td>66</td>
      <td>52</td>
      <td>58</td>
      <td>58</td>
      <td>58</td>
      <td>58</td>
      <td>56</td>
      <td>50</td>
      <td>50</td>
      <td>55</td>
      <td>56</td>
      <td>60</td>
      <td>63</td>
    </tr>
    <tr>
      <td>Gap/out of LF</td>
      <td>6</td>
      <td>5</td>
      <td>6</td>
      <td>6</td>
      <td>10</td>
      <td>4</td>
      <td>5</td>
      <td>7</td>
      <td>11</td>
      <td>7</td>
      <td>0</td>
      <td>6</td>
      <td>4</td>
      <td>10</td>
    </tr>
    <tr>
      <td>Military</td>
      <td>14</td>
      <td>12</td>
      <td>19</td>
      <td>0</td>
      <td>14</td>
      <td>12</td>
      <td>15</td>
      <td>12</td>
      <td>0</td>
      <td>16</td>
      <td>16</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <td>Parental leave</td>
      <td>.4</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>.6</td>
      <td>1</td>
      <td>6</td>
      <td>23</td>
      <td>16</td>
      <td>21</td>
      <td>8</td>
      <td>22</td>
      <td>26</td>
    </tr>
    <tr>
      <td>Unemployed</td>
      <td>5</td>
      <td>19.1</td>
      <td>19.5</td>
      <td>19.0</td>
      <td>19.1</td>
      <td>18.8</td>
      <td>18.3</td>
      <td>19.3</td>
      <td>19.2</td>
      <td>19.0</td>
      <td>19.1</td>
      <td>19.2</td>
      <td>19.3</td>
      <td>19.3</td>
    </tr>
    <tr>
      <td>Median age first job</td>
      <td>19.1</td>
      <td>19.5</td>
      <td>19.0</td>
      <td>19.1</td>
      <td>18.8</td>
      <td>18.3</td>
      <td>19.3</td>
      <td>19.2</td>
      <td>19.0</td>
      <td>19.1</td>
      <td>19.2</td>
      <td>19.3</td>
      <td>19.3</td>
      <td>19.3</td>
    </tr>
    <tr>
      <td>Mean number of transitions</td>
      <td>5.2</td>
      <td>3.8</td>
      <td>4.2</td>
      <td>5.6</td>
      <td>5.3</td>
      <td>5.6</td>
      <td>6.6</td>
      <td>6.8</td>
      <td>4.6</td>
      <td>5.3</td>
      <td>6.4</td>
      <td>7.6</td>
      <td>7.6</td>
      <td>8.4</td>
    </tr>
    <tr>
      <td>EGP I</td>
      <td>34</td>
      <td>38</td>
      <td>32</td>
      <td>32</td>
      <td>33</td>
      <td>35</td>
      <td>35</td>
      <td>39</td>
      <td>39</td>
      <td>32</td>
      <td>34</td>
      <td>19</td>
      <td>13</td>
      <td>14</td>
    </tr>
    <tr>
      <td>EGP II</td>
      <td>38</td>
      <td>41</td>
      <td>38</td>
      <td>43</td>
      <td>34</td>
      <td>39</td>
      <td>39</td>
      <td>45</td>
      <td>48</td>
      <td>52</td>
      <td>60</td>
      <td>51</td>
      <td>52</td>
      <td>52</td>
    </tr>
    <tr>
      <td>EGP IIIa</td>
      <td>6</td>
      <td>7</td>
      <td>8</td>
      <td>8</td>
      <td>5</td>
      <td>4</td>
      <td>5</td>
      <td>34</td>
      <td>45</td>
      <td>39</td>
      <td>34</td>
      <td>24</td>
      <td>26</td>
      <td>26</td>
    </tr>
    <tr>
      <td>EGP IIIb</td>
      <td>21</td>
      <td>25</td>
      <td>23</td>
      <td>21</td>
      <td>19</td>
      <td>18</td>
      <td>6</td>
      <td>38</td>
      <td>41</td>
      <td>36</td>
      <td>29</td>
      <td>44</td>
      <td>39</td>
      <td>38</td>
    </tr>
    <tr>
      <td>EGP IVa, IVb, and IVc</td>
      <td>7</td>
      <td>9</td>
      <td>8</td>
      <td>6</td>
      <td>8</td>
      <td>19</td>
      <td>6</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
    </tr>
    <tr>
      <td>EGP V</td>
      <td>24</td>
      <td>13</td>
      <td>9</td>
      <td>8</td>
      <td>18</td>
      <td>28</td>
      <td>35</td>
      <td>2</td>
      <td>5</td>
      <td>5</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>7</td>
    </tr>
    <tr>
      <td>EGP VI</td>
      <td>39</td>
      <td>47</td>
      <td>47</td>
      <td>45</td>
      <td>34</td>
      <td>33</td>
      <td>39</td>
      <td>27</td>
      <td>17</td>
      <td>19</td>
      <td>14</td>
      <td>20</td>
      <td>11</td>
      <td>22</td>
    </tr>
    <tr>
      <td>EGP VIIa and VIIb</td>
      <td>34</td>
      <td>36</td>
      <td>33</td>
      <td>34</td>
      <td>33</td>
      <td>33</td>
      <td>39</td>
      <td>73</td>
      <td>65</td>
      <td>72</td>
      <td>50</td>
      <td>59</td>
      <td>65</td>
      <td>72</td>
    </tr>
    <tr>
      <td>Education</td>
      <td>72</td>
      <td>65</td>
      <td>73</td>
      <td>65</td>
      <td>73</td>
      <td>72</td>
      <td>65</td>
      <td>73</td>
      <td>65</td>
      <td>72</td>
      <td>50</td>
      <td>40</td>
      <td>34</td>
      <td>21</td>
    </tr>
    <tr>
      <td>Gap/out of LF</td>
      <td>6</td>
      <td>5</td>
      <td>7</td>
      <td>9</td>
      <td>6</td>
      <td>7</td>
      <td>6</td>
      <td>7</td>
      <td>5</td>
      <td>9</td>
      <td>6</td>
      <td>7</td>
      <td>5</td>
      <td>6</td>
    </tr>
    <tr>
      <td>Military</td>
      <td>13</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
      <td>13</td>
      <td>13</td>
      <td>11</td>
      <td>11</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
      <td>17</td>
      <td>14</td>
      <td>11</td>
    </tr>
    <tr>
      <td>Parental leave</td>
      <td>.5</td>
      <td>.1</td>
      <td>.4</td>
      <td>.4</td>
      <td>.4</td>
      <td>.4</td>
      <td>1</td>
      <td>1</td>
      <td>23</td>
      <td>3.8</td>
      <td>9</td>
      <td>15</td>
      <td>22</td>
      <td>34</td>
    </tr>
    <tr>
      <td>Unemployed</td>
      <td>7</td>
      <td>3</td>
      <td>9</td>
      <td>5</td>
      <td>10</td>
      <td>8</td>
      <td>8</td>
      <td>13</td>
      <td>10</td>
      <td>13</td>
      <td>17</td>
      <td>14</td>
      <td>11</td>
      <td>12</td>
    </tr>
    <tr>
      <td>Median age first job</td>
      <td>20.3</td>
      <td>19.3</td>
      <td>19.9</td>
      <td>20.2</td>
      <td>20.9</td>
      <td>20.6</td>
      <td>20.7</td>
      <td>19.8</td>
      <td>18.6</td>
      <td>18.7</td>
      <td>19.1</td>
      <td>19.8</td>
      <td>20.2</td>
      <td>20.1</td>
    </tr>
    <tr>
      <td>Mean number of transitions</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>4.5</td>
      <td>5.0</td>
      <td>5.4</td>
      <td>5.5</td>
      <td>5.5</td>
      <td>6.3</td>
      <td>4.5</td>
      <td>5.2</td>
      <td>6.1</td>
      <td>6.6</td>
      <td>7.3</td>
      <td>6.7</td>
    </tr>
  </tbody>
</table>

Note: EGP = Erikson-Goldthorpe-Portocarero; LF = labor force.

&lt;page_number&gt;78&lt;/page_number&gt;

---


## Page 36

<header>Liao and Fasang</header>
&lt;page_number&gt;79&lt;/page_number&gt;

&lt;img&gt;Relative frequency plots of employment sequences by cohort, number of groupings = 75, sorted by age of first job, based on SVRspell distance (for order), East German and West German men and women.&lt;/img&gt;

Figure B1. Relative frequency plots of employment sequences by cohort, number of groupings = 75, sorted by age of first job, based on SVRspell distance (for order), East German and West German men and women.

<table>
  <tr>
    <td rowspan="3">East</td>
    <td colspan="3">Women</td>
    <td colspan="3">Men</td>
  </tr>
  <tr>
    <td>1944-1949, R2=1</td>
    <td>1950-1953, R2=.73</td>
    <td>1954-1957, R2=.51</td>
    <td>1944-1949, R2=1</td>
    <td>1950-1953, R2=.65</td>
    <td>1954-1957, R2=.57</td>
  </tr>
  <tr>
    <td>1958-1961, R2=.43</td>
    <td>1962-1965, R2=.37</td>
    <td>1966-1971, R2=.51</td>
    <td>1958-1961, R2=.46</td>
    <td>1962-1965, R2=.71</td>
    <td>1966-1971, R2=.69</td>
  </tr>
  <tr>
    <td rowspan="3">West</td>
    <td>1944-1949, R2=.06</td>
    <td>1950-1953, R2=.09</td>
    <td>1954-1957, R2=.02</td>
    <td>1944-1949, R2=.21</td>
    <td>1950-1953, R2=.14</td>
    <td>1954-1957, R2=.14</td>
  </tr>
  <tr>
    <td>1958-1961, R2=.02</td>
    <td>1962-1965, R2=.05</td>
    <td>1966-1971, R2=.09</td>
    <td>1958-1961, R2=.04</td>
    <td>1962-1965, R2=.05</td>
    <td>1966-1971, R2=.05</td>
  </tr>
</table>

<table>
  <tr>
    <td>[I] Higher grade professionals</td>
  </tr>
  <tr>
    <td>[II] Lower grade professionals</td>
  </tr>
  <tr>
    <td>[IIIa] Routine non manual employees, higher grade</td>
  </tr>
  <tr>
    <td>[IIIb] Routine non manual employees, lower grade</td>
  </tr>
  <tr>
    <td>[IVa,b,c] Small proprietors, farmers</td>
  </tr>
  <tr>
    <td>[V] Lower grade technicians; supervisors of manual workers</td>
  </tr>
  <tr>
    <td>[VI] Skilled manual workers</td>
  </tr>
  <tr>
    <td>[VIIa,b] Manual worker in primary production</td>
  </tr>
  <tr>
    <td>education</td>
  </tr>
  <tr>
    <td>gap/out of LF</td>
  </tr>
  <tr>
    <td>military</td>
  </tr>
  <tr>
    <td>parental leave</td>
  </tr>
  <tr>
    <td>unemployed</td>
  </tr>
</table>

---


## Page 37

&lt;page_number&gt;80&lt;/page_number&gt;
<header>Sociological Methodology 51(1)</header>

&lt;img&gt;Relative frequency plots of employment sequences by cohort, number of groupings 75, sorted by age of first job, based on OMstran distance (for duration), East German and West German men and women.&lt;/img&gt;

Figure B2. Relative frequency plots of employment sequences by cohort, number of groupings 75, sorted by age of first job, based on OMstran distance (for duration), East German and West German men and women.

---


## Page 38

<header>Liao and Fasang</header>
&lt;page_number&gt;81&lt;/page_number&gt;

## Acknowledgments

&lt;img&gt;NORFACE NETWORK&lt;/img&gt;
&lt;img&gt;EUROPEAN UNION&lt;/img&gt;

This article uses data from the National Educational Panel Study (NEPS): Starting Cohort Adults, doi:10.5157/NEPS:SC6:11.0.0. From 2008 to 2013, NEPS data was collected as part of the Framework Program for the Promotion of Empirical Educational Research funded by the German Federal Ministry of Education and Research (BMBF). As of 2014, NEPS is carried out by the Leibniz Institute for Educational Trajectories (LIfBi) at the University of Bamberg in cooperation with a nationwide network. For this research, Tim Futing Liao benefited from a 2017–2018 fellowship at the Center for Advanced Study in the Behavioral Sciences (Stanford University) and a leave from the University of Illinois. Anette Eva Fasang gratefully acknowledges funding from the Swedish Research Council for Health, Working Life and Welfare (FORTE, grant 2018-01612) for supporting a research visit at the Stockholm Demography Unit in spring 2019 and funding from the project EQUALLIVES, which is financially supported by the NORFACE Joint Research Programme on Dynamics of Inequality Across the Life-Course, which is cofunded by the European Commission through the European Union’s Horizon 2020 Research and Innovation Programme under grant agreement 724363.

## ORCID iD

Tim Futing Liao &lt;img&gt;ORCID logo&lt;/img&gt; https://orcid.org/0000-0002-1296-7660

## Notes

1. In a similar tradition, Oh and Raftery (2001) adapted BIC for assessing dimension choice in multidimensional scaling.
2. The computation of gravity centers is done with the disscenter function of TraMineR.
3. A negative BIC indicates support for the null model mirroring the levels on the positive side as long as the negative value is above 1 – log(n) the lower bound (Adrian E. Raftery, personal communication, January 31, 2018).
4. The computational results can be obtained from the authors upon request.

## References

Aassve, Arnstein, Francesco C. Billari, and Raffaella Piccarreta. 2007. “Strings of Adulthood: A Sequence Analysis of Young British Women’s Work-Family Trajectories.” *European Journal of Population* 23(3/4):369–88.
Abbott, Andrew. 2016. *Processual Sociology*. Chicago: University of Chicago Press.
Abbott, Andrew, and John Forrest. 1986. “Optimal Matching Methods for Historical Sequences.” *Journal of Interdisciplinary History* 16(3):471–94.
Aisenbrey, Silke, and Hannah Brückner. 2008. “Occupational Aspirations and the Gender Gap in Wages.” *European Sociological Review* 24(5):633–49.
Aisenbrey, Silke, and Anette E. Fasang. 2010. “New Life for Old Ideas: The ‘Second Wave’ of Sequence Analysis. Bringing the ‘Course’ Back into the Life Course.” *Sociological Methods & Research* 38(3):420–62.
Alwin, Duane F., and Ryan J. McCammon. 2003. “Generations, Cohorts, and Social Change.” Pp. 23–49 in *Handbook of the Life Course*. Boston: Springer.
Aßmann, Christian, and Sabine Zinn. 2011. “NEPS Data Manual. Starting Cohort 6: Adult Education and Lifelong Learning [Supplement C] Weighting.” Bamberg, Germany: National Education Panel Study Data Center (NEPS) Documentation, University of Bamberg.
Barban, N, X. de Luna, E. Lundholm, I. Svensson, and F. C. Billari. 2017. “Causal Effects of the Timing of Life-Course Events: Age at Retirement and Subsequent Health.” *Sociological Methods & Research* 49(1):216–49.

---


## Page 39

&lt;page_number&gt;82&lt;/page_number&gt;
<header>Sociological Methodology 51(1)</header>

Bastin, Sonja, Michaela Kreyenfeld, and Christine Schnor. 2012. “Diversität von Familienformen in Ost- und Westdeutschland” ["Diversity of Family Forms in East and West Germany"]. Max-Planck-Institut für demografische Forschung, Arbeitspapier, 1.

Berchtold, André, and Adrian E. Raftery. 2002. “The Mixture Transition Distribution Model for High-Order Markov Chains and Non-Gaussian Time Series.” *Statistical Science* 17:328–56.

Bernardi, F., L. Chakhaia, and L. Leopold. 2017. “‘Sing Me a Song with Social Significance’: The (Mis)use of Statistical Significance Testing in European Sociological Research.” *European Sociological Review* 33(1):1–15.

Biemann, Thomas, Anette E. Fasang, and Daniela Grunow. 2011. “Do Economic Globalization and Industry Growth Destabilize Careers? An Analysis of Career Complexity and Career Patterns over Time.” *Organization Studies* 32(12):1639–63.

Blanchard, Philippe, Felix Bühlmann, and Jacques-Antoine Gauthier, eds. 2014. *Advances in Sequence Analysis: Methods, Theories and Applications*. New York: Spinger.

Blossfeld, H.-P., H.-G. Roßbach, and J. von Maurice, eds. 2011. “Education as a Lifelong Process: The German National Educational Panel Study (NEPS).” *Zeitschrift für Erziehungswissenschaft* 14.

Blossfeld, Pia N., Gwendolyn J. Blossfeld, and Hans P. Blossfeld. 2015. “Educational Expansion and Inequalities in Educational Opportunity: Long-Term Changes for East and West Germany.” *European Sociological Review* 31(2):144–60.

Brückner, Hannah. 2004. *Gender Inequality in the Life Course: Social Change and Stability in West Germany, 1975–1995*. Piscataway, NJ: Transaction.

Burnham, Kenneth P., and David R. Anderson. 1998. *Model Selection and Inference: A Practical Information-Theoretical Approach*. New York: Springer.

Burnham, Kenneth P., and David R. Anderson. 2004. “Multimodel Inference: Understanding AIC and BIC in Model Selection.” *Sociological Methods & Research* 33(2):261–304.

Cooke, Lynn P. 2011. *Gender-Class Equality in Political Economies*. New York: Routledge.

Cornwell, Benjamin. 2015. *Social Sequence Analysis: Methods and Applications*, Vol. 37. Cambridge, UK: Cambridge University Press.

Diewald, Martin, Anne Goedicke, and Karl Ulrich Mayer, eds. 2006. *After the Fall of the Wall: Life Courses in the Transformation of East Germany*. Stanford, CA: Stanford University Press.

Dijkstra, W., and T. Taris. 1995. “Measuring the Agreement between Sequences.” *Sociological Methods & Research* 24(2):214–31.

Elder, Glen H., Monica Kirkpatrick Johnson, and Robert Crosnoe. 2003. “The Emergence and Development of Life Course Theory.” Pp. 3–19 in *Handbook of the Life Course*. Boston: Springer.

Elzinga, Cees H. 2003. “Sequence Similarity: A Nonaligning Technique.” *Sociological Methods & Research* 32(1):3–29.

Elzinga, Cees H., and Matthias Studer. 2015. “Spell Sequences, State Proximities, and Distance Metrics.” *Sociological Methods & Research* 44(1):3–47.

Eisenstadt, Shmuel N. 2000. “Multiple Modernities.” *Daedalus* 129(1):1–29.

Erikson, Robert, John H. Goldthorpe, and Lucienne Portocarero. 1979. “Intergenerational Class Mobility in Three Western European Societies: England, France and Sweden.” *British Journal of Sociology* 30(4):415–41.

Esping-Andersen, Gøsta. 1990. *The Three Worlds of Welfare Capitalism*. Princeton, NJ: Princeton University Press.

Fasang, Anette E. 2012. “Retirement Patterns and Income Inequality.” *Social Forces* 90(3):685–711.

Fasang, Anette E. 2014. “New Perspectives on Family Formation: What Can We Learn from Sequence Analysis?” In *Advances in Sequence Analysis: Methods, Theories and Applications*, edited by P. Blanchard, F. Bühlmann, and J.-A. Gauthier. New York: Springer.

Fasang, Anette E., and Tim F. Liao. 2014. “Visualizing Sequences in the Social Sciences: Relative Frequency Sequence Plots.” *Sociological Methods & Research* 43(4):643–76.

Fukuyama, Francis. 2006. *The End of History and the Last Man*. New York: Simon & Schuster.

Gabadinho, Alexis, and Gilbert Ritschard. 2016. “Analyzing State Sequences with Probabilistic Suffix Trees: The PST R Package.” *Journal of Statistical Software* 72(3):1–39.

---


## Page 40

<header>Liao and Fasang</header>
&lt;page_number&gt;83&lt;/page_number&gt;

Gabadinho, Alexis, Gilbert Ritschard, Nicolas Müller, and Matthias Studer. 2011. “Analyzing and Visualizing State Sequences in R with TraMineR.” *Journal of Statistical Software* 40(4):1–37.

Gelman, Andrew. 2004. “Exploratory Data Analysis for Complex Models.” *Journal of Computational and Graphical Statistics* 13(4):755–79.

Goldstein, Joshua, and Michaela Kreyenfeld. 2011. “Has East Germany Overtaken West Germany? Recent Trends in Order-Specific Fertility.” *Population and Development Review* 37(3):453–72.

Gross, J. H. 2015. “Testing What Matters (if You Must Test at All): A Context-Driven Approach to Substantive and Statistical Significance.” *American Journal of Political Science* 59(3):775–88.

Hall, Peter A., and David Soskice, eds. 2001. *Varieties of Capitalism: The Institutional Foundations of Comparative Advantage*. Oxford, UK: Oxford University Press.

Helske, Satu, and Jouni Helske. 2017. “Mixture Hidden Markov Models for Sequence Data: The seqHMM Package in R.” *arXiv*. Retrieved September 13, 2020. https://arxiv.org/abs/1704.00543.

Huinink, Johannes, et al. 1995. *Kollektiv und Eigensinn: Lebensverläufe in der DDR und Danach* [Collective and Self-Determination: Life-Courses in the GDR and After]. Munich, Germany: Oldenbourg Verlag.

Jalovaara, Marika, and Anette Fasang. 2020. “Family Life Courses, Gender, and Mid-life Earnings.” *European Sociological Review* 36(2):159–78.

Jeffreys, Harold. 1961. *Theory of Probability*. 3rd ed. Oxford, UK: Oxford University Press.

Kass, Robert E., and Adrian E. Raftery. 1995. “Bayes Factor.” *Journal of the American Statistical Association* 90(430):773–95.

Kurz, Karin, Steffen Hillmert, and Daniela Grunow. 2006. “Increasing Instability in Employment Careers of West German Men? A Comparison of the Birth Cohorts 1940, 1955 and 1964.” Pp. 75–113 in *Globalization, Uncertainty, and Men’s Careers: An International Comparison*, edited by Hans-Peter Blossfeld and Heather Anne Hofmeister. London: Elgar.

Leopold, T., J. Skopek, and M. Raab. 2011. “NEPS Data Manual. Starting Cohort 6: Adult Education and Lifelong Learning.” Bamberg, Germany: National Education Panel Study Data Center (NEPS) Documentation, University of Bamberg.

Levine, J. H. 2000. “But What Have You Done for Us Lately? Commentary on Abbott and Tsay.” *Sociological Methods & Research* 29(1):34–40.

Levitt, Theodore. 1993. “The Globalization of Markets.” Pp. 249–66 in *Readings in International Business: A Decision Approach*, edited by Robert Z. Aliber and Reid W. Click. Cambridge, MA: MIT Press.

Liao, Tim Futing. 2002. *Statistical Group Comparison*. Hoboken, NJ: John Wiley.

Liao, Tim Futing. 2004. “Comparing Social Groups: Wald Statistics for Testing Equality among Multiple Logit Models.” *International Journal of Comparative Sociology* 45(1):3–16.

Matysiak, Anna, and Stephanie Steinmetz. 2008. “Finding Their Way? Female Employment Patterns in West Germany, East Germany, and Poland.” *European Sociological Review* 24(3):331–45.

Mau, Steffen, and Wolfgang Zapf. 1998. “Zwischen Schock und Anpassung: Ostdeutsche Familienbildung im Übergang” [“Between Shock and Adaptation: East German Family Formation in Transition”]. *Informationsdienst Soziale Indikatoren* 20:1–4.

Mayer, Karl Ulrich. 1990. *Lebensverläufe und sozialer Wandel* [Life Courses and Social Change], Vol. 31. Opladen, Germany: Westdeutscher Verlag.

Mayer, Karl Ulrich. 2004. “Whose Lives? How History, Societies, and Institutions Define and Shape Life Courses.” *Research in Human Development* 1(3):161–87.

Mayer, Karl Ulrich. 2006. “Society of Departure: The German Democratic Republic.” Pp. 29–43 in *After the Fall of the Wall: Life Courses in the Transformation of East Germany*, edited by M. Diewald, A. Goedicke, and K. U. Mayer. Stanford, CA: Stanford University Press.

Mayer, Karl Ulrich, and Heike Solga. 1994. “Mobilität und Legitimität: Zum Vergleich der Chancenstrukturen in der alten DDR und der alten BRD oder: Haben Mobilitätschancen zu Stabilität und Zusammenbruch der DDR beigetragen? Ralf Dahrendorf zum 65. Geburtstag” [“Mobility and Legitimacy: Comparing Opportunity Structures in the Old GDR and the Old FRG”]. *Kölner Zeitschrift für Soziologie und Sozialpsychologie* 46(2):193–208.

---


## Page 41

&lt;page_number&gt;84&lt;/page_number&gt;
<header>Sociological Methodology 51(1)</header>

McShane, Blakeley B., David Gal, Andrew Gelman, Christian Robert, and Jennifer L. Tackett. 2019. “Abandon Statistical Significance.” *American Statistician* 73(Suppl. 1):235–45.

Nauck, Bernhard, Norbert F. Schneider, and Andrea Tölke. 1995. “Familie im gesellschaftlichen Umbruch—Nachholende oder divergierende Modernisierung” [“Family during Societal Change—Delayed or Diverging Modernization?”]. In *Familie und Lebensverlauf im gesellschaftlichen Umbruch* [Family and Life-Course during Societal Change], edited by B. Nauck, N. F. Schneider, and A. Tölk. Stuttgart, Germany: Enke.

Oh, Man-Suk, and Adrian E. Raftery. 2001. “Bayesian Multidimensional Scaling and Choice of Dimension.” *Journal of the American Statistical Association* 96(455):1031–44.

Piccarreta, Raffaella. 2012. “Graphical and Smoothing Techniques for Sequence Analysis.” *Sociological Methods & Research* 41(2):362–80.

Piccarreta, Raffaella. 2017. “Joint Sequence Analysis: Association and Clustering.” *Sociological Methods & Research* 46(2):252–87.

Piccarreta, Raffaella, and Mathias Studer. 2018. “Holistic Analysis of the Life Course: Methodological Challenges and New Perspectives.” *Advances in Life Course Research* 41:100251.

Raab, Marcel, Anette Eva Fasang, Aleksi Karhula, and Jani Erola. 2014. “Sibling Similarity in Family Formation.” *Demography* 51(6):2127–54.

Raftery, Adrian E. 1986. “Choosing Models for Cross-Classifications.” *American Sociological Review* 51(1):145–6.

Raftery, Adrian E. 1995. “Bayesian Model Selection in Social Research.” *Sociological Methodology* 25:111–63.

Robette, Nicolas, and Xavier Bry. 2012. “Harpoon or Bait? A Comparison of Various Metrics in Fishing for Sequence Patterns.” *Bulletin of Sociological Methodology/Bulletin de Méthodologie Sociologique* 116(1):5–24.

Ryder, Norman B. 1985. “The Cohort as a Concept in the Study of Social Change.” Pp. 9–44 in *Cohort Analysis in Social Research*. New York: Springer.

Schneider, Norbert F., R. Naderi, and S. Ruppenthal. 2012. “Familie in Deutschland nach dem gesellschaftlichen Umbruch. Sind Ost-West-Differenzierungen in der Familienforschung zwanzig Jahre nach der Wiedervereinigung noch Sinnvoll?” *Zeitschrift für Familienforschung*, Sonderheft 9:29–53.

Schnettler, Sebastian, and Sebastian Klüsener. 2014. “Economic Stress or Random Variation? Revisiting German Reunification as a Natural Experiment to Investigate the Effect of Economic Contraction on Sex Ratios at Birth.” *Environmental Health* 13(1):117.

Solga, Heike. 1995. *Auf dem Weg in eine klassenlose Gesellschaft? Klassenlagen und Mobilität zwischen Generationen in der DDR* [On the Way to a Classless Society? Class Positions and Intergenerational Mobility in the GDR]. Munich, Germany: Oldenbourg Verlag.

Solga, Heike. 2006. “The Rise of Meritocracy? Class Mobility in East Germany before and after 1989.” Pp. 140–69 in *After the Fall of the Wall: Life Courses in the Transformation of East Germany*, edited by M. Diewald, A. Goedicke, and K. U. Mayer. Stanford, CA: Stanford University Press.

Sørensen, Annemette, and Heike Trappe. 1995. “The Persistence of Gender Inequality in Earnings in the German Democratic Republic.” *American Sociological Review* 60(3):398–406.

Struffolino, Emanuela, Matthias Studer, and Anette E. Fasang. 2016. “Gender, Education, and Family Life Courses in East and West Germany: Insights from New Sequence Analysis Techniques.” *Advances in Life Course Research* 29:66–79.

Studer, Matthias. 2013. “WeightedCluster Library Manual: A Practical Guide to Creating Typologies of Trajectories in the Social Sciences with R.” LIVES Working Papers 24. Retrieved September 13, 2020. https://www.lives-nccr.ch/fr/publication/weightedcluster-library-manual-practical-guide-creating-typologies-trajectories-social.

Studer, Matthias, and Gilbert Ritschard. 2016. “What Matters in Differences between Life Trajectories: A Comparative Review of Sequence Dissimilarity Measures.” *Journal of the Royal Statistical Society: Series A (Statistics in Society)* 179(2):481–511.

---


## Page 42

<header>Liao and Fasang</header>
&lt;page_number&gt;85&lt;/page_number&gt;

Studer, Matthias, Gilbert Ritschard, Alexis Gabadinho, and Nicholas S. Müller. 2011. “Discrepancy Analysis of State Sequences.” *Sociological Methods & Research* 40(3):471–510.

Studer, Matthias, Emanuela Struffolino, and Anette E. Fasang. 2018. “Estimating the Relationship between Time-Varying Covariates and Trajectories: The Sequence Analysis Multistate Model Procedure.” *Sociological Methodology* 48(1):103–35.

Tilly, Charles. 1984. *Big Structures, Large Processes, Huge Comparisons*. New York: Russell Sage.

Trappe, Heike. 1995. *Emanzipation oder Zwang? Frauen in der DDR zwischen Beruf, Familie und Sozialpolitik [Emancipation or Force? Women in the GDR between Occupation, Family, and Social Policy]*. Berlin: Walter de Gruyter.

Tukey, John W. 1977. *Exploratory Data Analysis*. Reading, MA: Addison-Wesley.

Tukey, John W., and Paul A. Tukey. 1988a. “Graphical Display of Data Sets in 3 or More Dimensions.” Pp. 189–288 in *The Collected Works of John W. Tukey, Vol. V, Graphics: 1965–1985*, edited by W. S. Cleveland. Pacific Groves, CA: Wadsworth & Brooks.

Tukey, John W., and Paul A. Tukey. 1988b. “Some Graphics for Studying Four-Dimensional Data.” Pp. 171–88 in *The Collected Works of John W. Tukey, Vol. V, Graphics: 1965–1985*, edited by W. S. Cleveland. Pacific Groves, CA: Wadsworth & Brooks.

Wu, Lawrence L. 2000. “Some Comments on ‘Sequence Analysis and Optimal Matching Methods in Sociology: Review and Prospect.’” *Sociological Methods & Research* 29(1):41–64.

Zapf, Wolfgang, and Steffen Mau. 1993. “Eine demographische Revolution in Ostdeutschland? Dramatischer Rückgang von Geburten, Eheschließungen und Scheidungen” [“A Revolution in East Germany? Dramatic Decline of Births, Marriages and Divorces”]. *Informationsdienst Soziale Indikatoren* 10:1–5.

## Author Biographies

**Tim Futing Liao** is a professor and head of sociology and a professor of statistics at the University of Illinois at Urbana-Champaign and a former editor of *Sociological Methodology*. His research focuses on inequality, the life-course, and sequence analysis and, more generally, social science methodology.

**Anette Eva Fasang** is a professor of sociology at Humboldt University of Berlin and a fellow at the WZB Berlin Social Science Center. Her research interests include family demography, stratification, life-course sociology, and quantitative methods for longitudinal data analysis.