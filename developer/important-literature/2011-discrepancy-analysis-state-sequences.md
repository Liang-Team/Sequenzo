## Page 1

<header>Article</header>

# Discrepancy Analysis of State Sequences

Sociological Methods & Research
40(3) 471–510
© The Author(s) 2011
Reprints and permission:
sagepub.com/journalsPermissions.nav
DOI: 10.1177/0049124111415372
http://smr.sagepub.com
&lt;img&gt;SAGE&lt;/img&gt;

Matthias Studer,¹ Gilbert Ritschard,¹ Alexis Gabadinho,¹ and Nicolas S. Müller¹

## Abstract
In this article, the authors define a methodological framework for analyzing the relationship between state sequences and covariates. Inspired by the principles of analysis of variance, this approach looks at how the covariates explain the discrepancy of the sequences. The authors use the pairwise dissimilarities between sequences to determine the discrepancy, which makes it possible to develop a series of statistical significance-based analysis tools. They introduce generalized simple and multifactor discrepancy-based methods to test for differences between groups, a pseudo-R² for measuring the strength of sequence-covariate associations, a generalized Levene statistic for testing differences in the within-group discrepancies, as well as tools and plots for studying the evolution of the differences along the time frame and a regression tree method for discovering the most significant discriminant covariates and their interactions. In addition, the authors extend all methods to account for case weights. The scope of the proposed methodological framework is illustrated using a real-world sequence data set.

## Keywords
distance, dissimilarities, analysis of variance, regression tree, tree-structured ANOVA, state sequence, optimal matching, homogeneity in discrepancies, Levene test, permutation test

¹University of Geneva, Geneva, Switzerland

**Corresponding Author:**
Matthias Studer, Institute for Demographic and Life Course Studies, University of Geneva, Bvd Pont D'Arve 40, 1211 Geneva 4, Switzerland
Email: matthias.studer@unige.ch

---


## Page 2

&lt;page_number&gt;472&lt;/page_number&gt;
Sociological Methods & Research 40(3)

Optimal matching (OM) and, more generally, clustering of state sequences have become popular tools for analyzing life trajectories since their initial introduction in the social sciences in the late 1980s (Abbott and Forrest 1986; Abbott and Hrycak 1990). The popularity of these techniques is largely attributable to the holistic view they provide on the life-course construction process. As emphasized by Billari (2005), such an approach considers the complete life sequence as a single unit of analysis, as opposed to event history analysis, for instance, which focuses on specific events of the life process such as marriage or the start of a new job.

In general, the OM approach consists of measuring the pairwise dissimilarities between sequences and then using the obtained values in an unsupervised clustering algorithm to build a typology of the observed sequences (i.e., to find homogeneous groups of sequences). Such analysis has proven to be an effective exploratory tool for discovering the main characteristics of a set of sequences without formulating any a priori hypothesis. Furthermore, it permits the identification of typical trajectories and recurring structures in the sequences, thus bringing out fundamental descriptive information and making the data easier to comprehend. From a sociological point of view, groups determined by clustering techniques are useful for characterizing typical trajectories. Even more, it is often assumed that cases in a cluster all follow more or less an associated ideal type of trajectory. Clustering serves in this way for identifying the ideal types, which in turn provide nice, simplified interpretations of the clusters.

Aside the understanding of the intrinsic characteristics of the sequences, in this article, we are interested in how the individual sequences are influenced by their context. A common practice in that perspective is to investigate the relationship of the identified sequence types, the clusters, with covariates such as sex and birth cohort. This is achieved, for example, by looking at the association between cluster membership and the covariates, or by explaining cluster membership by means of logistic regressions or classification trees. The downside of this cluster-based approach, however, is that by representing the diversity of trajectories with a limited number of clusters, one inevitably loses information about the diversity within each cluster. In the causal perspective, reducing the set of sequences to a limited number of standard trajectories is too crude an approximation and would lead to considering deviations from the standard inside a cluster as unexplained error terms. Furthermore, knowledge of cluster membership alone does not inform one about the distances and differences between clusters. As a result, wrong conclusions may be drawn about the sequence-covariate relationships.

---


## Page 3

<header>Studer et al.</header>
&lt;page_number&gt;473&lt;/page_number&gt;

As a solution to this problem, we propose a set of methods, the discrepancy analysis, that allow a direct analysis of the sequence-covariate links (i.e., without any prior clustering). Our approach focuses on the discrepancy of the sequences, which we measure from their pairwise dissimilarities (e.g., OM distances). This trick allows a generalization of the principles of analysis of variance (ANOVA). The basic idea behind the ANOVA approach is to measure and test the part of the discrepancy among sequences that can be explained by covariates. For instance, one can assess what fraction of the differences between individuals’ academic careers can be explained by sex or how the construction of one’s familial life may differ according to social origin.

The article is organized as follows. In the next section, we briefly review the relevant literature, and we then describe the data set on the transition from school to work used for illustrating the discussed methods. We next introduce a dissimilarity-based measure of discrepancy for a set of possibly weighted sequences and discuss its interpretation. We then propose methods to study the relationship between sequences and a single categorical variable; that is, we focus on the comparison of groups of sequences defined according to the levels of a given covariate. We derive a pseudo-R² for measuring the share of sequence discrepancy explained by the grouping variable and introduce a pseudo-F test to assess the significance of the association. We extend both statistics to account for case weights. We also propose a Levene-like statistic to test the homogeneity of within-group sequence discrepancies. To conclude the section, we illustrate the behavior of the statistics with simulations. Next, we show how results can be further investigated and rendered to characterize the differences between groups. In the following section, we deal with the multifactor case, for which we introduce original formulas that can account for case weights. We then exploit the previous material in a tree-structured fashion to reveal the factors that best discriminate sequences. Finally, we provide a brief overview on how to apply the presented methods in R with our TraMineR package.

To avoid overloading the reader, detailed mathematical developments and discussions have been taken out of the main text and included in Appendices A to C.

## Literature Review

Several approaches have been used to study the association between objects described by their pairwise dissimilarities and a categorical covariate (Anderson 2001; Cuadras 2008; Gower and Krzanowski 1999; McArdle

---


## Page 4

&lt;page_number&gt;474&lt;/page_number&gt;
Sociological Methods & Research 40(3)

and Anderson 2001; Mielke and Berry 2007). Apart from the most popular ones, which generally adopt principles of ANOVA or multivariate analysis of variance, Mielke and Berry (1983, 2007) suggested using ad hoc statistics on the basis of sums of distances. Reiss, Stevens, Shehzad, Petkova, and Milham (2009) showed that the two approaches are equivalent under certain conditions. The statistical significance of the association is generally assessed through permutation tests, although Mielke and Berry (2007) proposed a parameterized approximation for the empirical distribution generated by the permutations. All of these authors considered only metric distances. In contrast, McArdle and Anderson (2001) extended the same approach to semi-metric dissimilarities. This latter solution can also be applied in a multifactor case. As far as weights are concerned, an interesting contribution is an article by Delicado (2007), who accounted for weights with a test derived from Gower and Krzanowski’s (1999) formulation for the single-factor case.

The different methods are used in various domains, such as genetics (Zapala and Schork 2006), the study of income density functions (Delicado 2007), and the study of neuroimaging data (Reiss et al. 2009). Gower and Krzanowski (1999) dealt with situations in which the number of dependent (response) variables is larger than the number of observations. In line with the work by McArdle and Anderson (2001), the main application field is ecology, especially the analysis of ecosystems by means of semimetrics, such as that of Bray-Curtis. To the best of our knowledge, however, ANOVA-like tools have not been applied so far to life trajectory analysis.

The methods presented in this article are inspired from the work of Mielke and Berry (2007) and McArdle and Anderson (2001). We adapt their solutions for the study of state sequences in the social sciences and extend them, so that we can also account for case weights. We also complement the theoretical background of their approaches with notions such as the contribution to discrepancy derived from Batagelj’s (1988) developments of the Ward criterion.

The analysis of differences in discrepancies between groups (i.e., the test of equality in within-group discrepancies) has rarely been considered for dissimilarity-based discrepancies. Anderson (2006) proposed two tests on the basis of distances in an associated principal coordinate space. Studer, Ritschard, Gabadinho, and Müller (2010) considered a generalization of Bartlett’s test that uses directly the original dissimilarities. The latter approach is not suitable, however, for weighted data.

Regarding tree-structured methods, there have been only a few recent attempts to apply them to objects characterized by their dissimilarities. Piccarreta and Billari (2007) proposed a nonsupervised dissimilarity-based divisive tree algorithm that grows the tree independently of any covariates.

---


## Page 5

<header>Studer et al.</header>
&lt;page_number&gt;475&lt;/page_number&gt;

<table>
  <caption>Table I. List of Covariates</caption>
  <thead>
    <tr>
      <th>Variable</th>
      <th>Description</th>
      <th>Values</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>sex</td>
      <td>Gender</td>
      <td>Female, male</td>
    </tr>
    <tr>
      <td>region</td>
      <td>Location of school in Northern Ireland</td>
      <td>Belfast, northeastern, southeastern, southern, western</td>
    </tr>
    <tr>
      <td>religion</td>
      <td>Religion</td>
      <td>Catholic, protestant</td>
    </tr>
    <tr>
      <td>funemp</td>
      <td>Father unemployed at time of survey</td>
      <td>Yes, no</td>
    </tr>
    <tr>
      <td>fmpr</td>
      <td>Father has a professional, managerial or related job</td>
      <td>Yes, no</td>
    </tr>
    <tr>
      <td>livboth</td>
      <td>Living with both parents at time of first sweep of survey</td>
      <td>Yes, no</td>
    </tr>
    <tr>
      <td>grammar</td>
      <td>Grammar school secondary education</td>
      <td>Yes, no</td>
    </tr>
    <tr>
      <td>gcse5eq</td>
      <td>Qualifications gained by the end of compulsory education: five or more GCSEs at Grades A to C, or equivalent</td>
      <td>Yes, no</td>
    </tr>
  </tbody>
</table>

Note: GCSE = general certificate of secondary education.

They applied it to sequence data. Similarly to the present article, Geurts, Wehenkel, and d’Alché Buc (2006) proposed the use of a kernel-based supervised method; but, their approach was limited to Euclidean distances. A more general dissimilarity-based method can be found in Piccarreta (2010). Unlike our proposal, her tree-growing method is not controlled by a statistical significance and therefore requires postpruning. The tree algorithm considered below in “Tree-Structured Analysis of Sequences” is essentially the same as the one introduced by Studer, Ritschard, Gabadinho, and Müller (2009) and Studer et al. (2010). Here, we build on it and adapt it for weighted data.

## Illustrative Data Set

We begin by describing the application setting that will serve as an illustration throughout the article. We use data and the research question from McVicar and Anyadike-Danes’s (2002) study of the school-to-work transition in Northern Ireland. The aim is to “identify the ‘at-risk’ young people at age 16 years and to characterize their post-school career trajectories” using information such as qualification at the end of compulsory schooling, family background, and demographic characteristics. Table 1 presents a list of covariates available in the data set, named mvad. In addition, the data contain a “weight” variable for adjusting for response bias.

---


## Page 6

&lt;page_number&gt;476&lt;/page_number&gt;
Sociological Methods & Research 40(3)

# Discrepancy of a Set of Sequences

In this section, we define a measure of the discrepancy of a set of sequences. In a life-course framework, the discrepancy measures the between-individual variability of the life trajectories. Therefore, higher discrepancy, for example, would reflect a greater level of uncertainty about the path followed by the individuals. Depending on the situations, such uncertainty may be interpreted either as a form of precariousness or, on the contrary, as a reflection of the multiplicity of choices the individuals face. The discrepancy concept considered here must be clearly distinguished from the within-individual longitudinal state diversity that can be measured with the longitudinal entropy (Widmer and Ritschard 2009), the turbulence (Elzinga 2010), or the complexity index (Gabadinho, Ritschard, Studer, and Müller 2010). The latter measure the diversity of states and transitions inside sequences, while the discrepancy assesses the diversity of the trajectories.

Aside from this diversity interpretation, the discrepancy is also the key concept for measuring the association between sequences and covariates. Decomposing it into explained between-groups and residual within-groups discrepancy permits measurement of the explained share of discrepancy and testing for differences between groups. These topics are addressed in the next sections.

Because we cannot directly observe the distance to some “mean sequence,” the discrepancy of sequences will be defined from their pairwise dissimilarities. There are many different ways of computing such dissimilarities, either through proximity measures counting common characteristics (Elzinga 2007) or using edit distances (Lesnard 2010). The most popular dissimilarity measure used for sequence analysis in the social sciences is the OM edit distance, also known as generalized Levenshtein distance in computer science. It is defined as the lowest cost of transforming one sequence into another by means of state insertions-deletions (indels) and state substitutions. The total transformation cost depends on the individual cost of each used operation. Those costs can be organized into an augmented substitution cost matrix between states by considering each indel as a substitution with a null element (i.e., by defining a row and a column for this null element).¹ The resulting OM distance satisfies surely the triangle inequality, as long as the elements of this augmented substitution cost matrix verify it (Yujian and Bo 2007). When that is not the case, the resulting OM dissimilarity between two sequences x and y could be greater than the sum of their dissimilarities with some other sequence. Although we use the OM distance with the costs defined by McVicar and Anyadike-Danes (2002) for our application

---


## Page 7

<header>Studer et al.</header>
&lt;page_number&gt;477&lt;/page_number&gt;

example, the way of measuring the discrepancy described hereafter is in no way limited to the OM distance alone. Any other measure of dissimilarity between sequences could be used instead.

The presentation in the remainder of this section is based on the generalization of the Ward criterion made by Batagelj (1988). The concepts introduced may also be found in Anderson (2001), Reiss et al. (2009), and Mielke and Berry (2007), though these articles deal only with the unweighed case, while we propose here formulas that account for weights as well.

### Discrepancy Based on Dissimilarities

In the Euclidean case, the sum of squares SS, or inertia, may be expressed in terms of the pairwise squared Euclidean distances. Let $y = (y_i)$ be a vector of length $n$, $w_i$ the weight associated with case $i$, and $W$ the total sum of weights. The sum of squares can be expressed as (see Appendix A)

$$
SS = \sum_{i=1}^{n} w_i (y_i - \bar{y})^2 = \frac{1}{W} \sum_{i=1}^{n} \sum_{j=i+1}^{n} w_i w_j d_{e,ij}^2,
\quad (1)
$$

where $d_{e,ij}$ is the Euclidean distance between $i$ and $j$.

Following Mielke and Berry (2007), the concept of the sum of squares can be generalized to other dissimilarity measures by replacing the squared Euclidean distance $d_{e,ij}^2$ on the right-hand side of equation (1) with $d_{ij}^v$, where $d_{ij}$ is any possibly non-Euclidean measure of dissimilarity, and $v$ is a real positive exponent, yielding

$$
SS = \frac{1}{W} \sum_{i=1}^{n} \sum_{j=i+1}^{n} w_i w_j d_{ij}^v.
\quad (2)
$$

Although Mielke and Berry (2007) studied a wide range of values for the $v$ exponent, existing literature usually considers only either $v = 1$ or $v = 2$. We address the choice between these two values in Appendix B, in which we argue in favor of $v = 2$ for Euclidean metrics and $v = 1$ for non-Euclidean ones such as OM.

Applying the definition $s^2 = \frac{1}{W} SS$ of the sample variance, we get a fairly intuitive measure of the discrepancy of the sequence objects. Because the variance is theoretically defined for Euclidean distances, we prefer the term discrepancy for this more general setting. Interestingly, the discrepancy $s^2$ is equal to half of the weighted average of the pairwise dissimilarities:

---


## Page 8

&lt;page_number&gt;478&lt;/page_number&gt;
Sociological Methods & Research 40(3)

$$
s^2 = \frac{1}{2W^2} \sum_{i=1}^{n} \sum_{j=1}^{n} w_i w_j d_{ij}^v.
\quad (3)
$$

**Contribution to the Sum of Squares**

Batagelj (1988) showed that the previous generalization of the sum of squares $SS$ also implies that the dissimilarity $d_{x\bar{g}}^v$ between a sequence $x$ and the (possibly virtual) gravity center $\bar{g}$ of a set $G$ of sequences is (see equation [A6] in Appendix A)

$$
d_{x\bar{g}}^v = \frac{1}{W} (\sum_{i=1}^{n} w_i d_{xi}^v - SS)
\quad (4)
$$

According to Batagelj (1988), the notion of a gravity center holds for any kind of distances and objects, even though it is not clearly defined for complex nonnumeric objects such as sequences. It is likely that the gravity center does not itself belong to the object space, exactly as the mean of integer values may be a real noninteger value.

Because $SS = \sum_x w_x d_{x\bar{g}}^v$, each term $w_x d_{x\bar{g}}^v$ under this summation may be interpreted as the contribution of $x$ to the total sum of squares. Even though the gravity center may not be observable, equation (4) provides a comprehensive way to compute the most central sequence, the medoid, of a set using weights. Searching the $x$ that minimizes equation (4) is equivalent to minimizing the sum of the weighted distances from $x$ to all other sequences. The same solution was, for instance, considered in the unweighed setting by Abbott (1990) for finding a representative sequence.

The nonnegativity of this contribution automatically results when $d^v$ satisfies the triangle inequality (see Appendix A), while negative contributions to the discrepancy can occur when the triangle inequality does not hold. For non-Euclidean dissimilarities such as OM, it is therefore preferable to proceed with $v = 1$, which ensures the triangle inequality, rather than with squared dissimilarities (see Appendix B).

**Comparing Groups of Sequences**

Aside from evaluating the variability of a set of sequences, measuring discrepancy from pairwise dissimilarities permits generalization of the principles of ANOVA to any dissimilarity measure. It allows computation of the share of discrepancy “explained” by a covariate and thus evaluation of the strength of the association between trajectories and a covariate. Although

---


## Page 9

<header>Studer et al.</header>
&lt;page_number&gt;479&lt;/page_number&gt;

classical tests based on normality assumptions are not applicable in this case, the significance of the relation can be assessed through permutation tests, as discussed by Anderson (2001).
Below in “Testing Differences in Within-Group Discrepancies,” we introduce a new test to compare group discrepancy. In some situations, it may be of interest to test whether the discrepancies within groups differ significantly. We then discuss the interpretation and the visualization of the difference between state sequences. In the last subsection, we provide empirical insights on the behavior of the proposed tests with simulations.

## Measuring Association

When generalizing the notion of sum of squares to non-Euclidean measures of dissimilarity, the Huygens theorem (equation [5]) that states that the total sum of squares ($SS_T$) is the between sum of squares ($SS_B$) plus the residual within sum of squares ($SS_W$) remains valid (Batagelj 1988):

$$
SS_T = SS_B + SS_W.
\quad (5)
$$

Thus, we can apply the ANOVA machinery to sequence objects.
All terms in equation (5) can be derived from equation (2). The total sum of squares ($SS_T$) and the within sum of squares ($SS_W$) are computed directly with equation (2), $SS_W$ being simply the sum of the within sums of squares of each subgroup. The between sum of squares $SS_B$ is then obtained by taking the difference between $SS_T$ and $SS_W$. Using equation (5), we can assess the share of discrepancy explained by a categorical or discretized continuous variable. In the spirit of ANOVA, this reduction of discrepancy is due to a difference in the positioning of the gravity centers $\tilde{g}_k$ of the classes $k$ (Batagelj 1988). Hence, conceptually, we look for the part of the discrepancy that is explained by differences in group positioning, and we measure it with the $R^2$ formula (equation [6]). Alternatively, we may consider the $F$ that compares the explained discrepancy to the residual discrepancy. The $F$ formula is provided in equation (7), where $m$ is the number of groups:

$$
R^2 = \frac{SS_B}{SS_T};
\quad (6)
$$

$$
F = \frac{SS_B/(m-1)}{SS_W/(W-m)}.
\quad (7)
$$

---


## Page 10

&lt;page_number&gt;480&lt;/page_number&gt;
Sociological Methods & Research 40(3)

## Assessing Statistical Significance

The statistical significance of the association (i.e., of the explained part of the discrepancy) cannot be assessed with Fisher’s F distribution as in classical ANOVA.² The F statistic (equation [7]) does not follow a Fisher distribution with sequence objects for which the normality assumption is hardly defendable. Therefore, we consider a permutation test (Anderson 2001; Manly 2007), which works as follows. At each step, we change the group—the value of the covariates—assigned to each sequence by means of a randomly chosen permutation of the group membership vector. We thus get an F<sub>perm</sub> value for each permutation. Repeating this operation R times, we obtain an empirical nonparametric distribution of F that characterizes its distribution under independence (i.e., assuming the sequences are assigned to the cases independently of the explanatory factors). From this distribution, we can assess the significance of the observed F<sub>obs</sub> statistic by means of the proportion of F<sub>perm</sub> values that are higher than F<sub>obs</sub>. It is generally thought that 5,000 permutations should be used to assess a significance threshold of 1 percent and 1,000 for a threshold of 5 percent (see Manly 2007; Appendix C).

The issue is how we can or even if we should adapt such permutation tests to account for weights. We propose three solutions:

1. Replicate cases a number of times corresponding to the weights before performing the permutation. This approach supposes that weights are integer counts.
2. Replace at each step the simple permutation with a random assignment of covariate profiles to the sequences using distributions defined by the weights.
3. Proceed with permutations ignoring weights and use them for computing the statistics for each permutation.

When the weights stand for counts of aggregated cases, we should restore individual cases by replicating the aggregated ones. By permuting aggregates only, we would miss possible permutations of cases within aggregated groups and therefore end up with a less powerful test. The second option is more or less equivalent but can be used with noninteger weights. Both of these techniques assume that weights reflect an aggregation of independently drawn cases. However, when weights do not result from aggregation but are intended to improve the sample representativeness, as it is the case in the example mvad data, it would not be correct to replicate cases. For example, a weight of 4 would not mean that four cases were drawn, and hence replicating it four times would incorrectly inflate the sample size.

---


## Page 11

<header>Studer et al.</header>
&lt;page_number&gt;481&lt;/page_number&gt;

&lt;img&gt;Figure 1. Empirical Distribution of the F Statistic Under Independence With the livboth Covariate&lt;/img&gt;

Thus, the first and the second solutions should be used with counts reflecting aggregation. The third one should be applied in cases in which weights are aimed at improving the sample’s representativeness.

Figure 1 shows the empirical density curve of the $F_{perm}$ statistics obtained with 5,000 permutations of the values of the variable livboth (“living with both parents”). The observed $F_{obs}$ statistic is equal to 2.49. The associated significance is .21 and corresponds to the red (gray) area in the plot. With 21 percent of the random F values greater than $F_{obs}$, we cannot conclude that the trajectory of young people differs significantly with the values of the livboth covariate.

Table 2 summarizes the results of the discrepancy analysis using both squared and nonsquared dissimilarities in the weighted case and nonsquared dissimilarities only in the unweighted case. From the tests using weights, the trajectories significantly differ with qualification at the end of compulsory schooling (gcse5eq), type of compulsory schooling (grammar), father employment status (funemp and fmpr), sex, and region. We measured the strongest association for the end of compulsory schooling qualification, which lets us think that selection has occurred before the start of the sequences.

It is worth noting that results based on squared and nonsquared dissimilarities are quite similar—both the ranking of covariates and the significance levels are the same—although the pseudo-$R^2$ values are higher in the case of squared dissimilarities. This is a general result. The p values associated with the unweighted test are lower, which may be explained by the additional variability that weights introduce in the estimation of the null F curve.

The significant effects found here agree with the significant effects on the cluster membership found by McVicar and Anyadike-Danes (2002), except

---


## Page 12

&lt;page_number&gt;482&lt;/page_number&gt;
<header>Sociological Methods & Research 40(3)</header>

**Table 2. Test of Association of Each Covariate With the School-to-Work Trajectories**

<table>
  <thead>
    <tr>
      <th rowspan="2">Variable</th>
      <th colspan="3">Nonsquared dissimilarity</th>
      <th colspan="3">Squared dissimilarity</th>
      <th colspan="3">Nonsquared, unweighted</th>
    </tr>
    <tr>
      <th>F</th>
      <th>R²</th>
      <th>p</th>
      <th>F</th>
      <th>R²</th>
      <th>p</th>
      <th>F</th>
      <th>R²</th>
      <th>p</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>gcse5eq</td>
      <td>104.09</td>
      <td>.128</td>
      <td>.000</td>
      <td>184.09</td>
      <td>.206</td>
      <td>.000</td>
      <td>67.69</td>
      <td>.087</td>
      <td>.000</td>
    </tr>
    <tr>
      <td>grammar</td>
      <td>59.34</td>
      <td>.077</td>
      <td>.000</td>
      <td>89.87</td>
      <td>.112</td>
      <td>.000</td>
      <td>23.16</td>
      <td>.032</td>
      <td>.000</td>
    </tr>
    <tr>
      <td>fmpr</td>
      <td>13.72</td>
      <td>.019</td>
      <td>.000</td>
      <td>23.00</td>
      <td>.031</td>
      <td>.000</td>
      <td>8.76</td>
      <td>.012</td>
      <td>.000</td>
    </tr>
    <tr>
      <td>funemp</td>
      <td>11.98</td>
      <td>.017</td>
      <td>.000</td>
      <td>24.05</td>
      <td>.033</td>
      <td>.000</td>
      <td>9.51</td>
      <td>.013</td>
      <td>.000</td>
    </tr>
    <tr>
      <td>sex</td>
      <td>11.03</td>
      <td>.015</td>
      <td>.000</td>
      <td>13.85</td>
      <td>.019</td>
      <td>.001</td>
      <td>6.84</td>
      <td>.010</td>
      <td>.000</td>
    </tr>
    <tr>
      <td>region</td>
      <td>5.44</td>
      <td>.030</td>
      <td>.000</td>
      <td>7.26</td>
      <td>.039</td>
      <td>.001</td>
      <td>5.50</td>
      <td>.030</td>
      <td>.000</td>
    </tr>
    <tr>
      <td>livboth</td>
      <td>2.49</td>
      <td>.003</td>
      <td>.211</td>
      <td>2.61</td>
      <td>.004</td>
      <td>.240</td>
      <td>2.23</td>
      <td>.003</td>
      <td>.033</td>
    </tr>
    <tr>
      <td>religion</td>
      <td>2.32</td>
      <td>.003</td>
      <td>.234</td>
      <td>1.88</td>
      <td>.003</td>
      <td>.365</td>
      <td>2.75</td>
      <td>.004</td>
      <td>.014</td>
    </tr>
  </tbody>
</table>

Note: See Table I for definitions of variables.

for the livboth and religion covariates, which are not significant in our weighted ANOVA-like model. Because only variables with significant effects in McVicar and Anyadike-Danes’s study were included in the data set made available by those authors, it is not possible to discover covariates that significantly explain the sequence discrepancy while not affecting significantly the cluster membership.

**Testing Differences in Within-Group Discrepancies**

In some situations, it may be of interest to test whether the discrepancies within groups differ significantly. For instance, Elzinga and Liefbroer (2007) were interested in testing the destandardization hypothesis stating that the family life trajectories of young adults become less similar over time (i.e., in testing for an increasing within-cohort discrepancy). However, their approach, which was to compare the mean pairwise similarities of four cohorts in 19 countries through 90 percent bootstrap confidence intervals, lends itself to criticism, because by bootstrapping similarities instead of individuals, it does not account for the strong correlation between similarities involving the same case. Better suited approaches are necessary to test differences in within-group discrepancies.

Studer et al. (2010) used a generalization of the Bartlett T (Bartlett 1937; Jobson 1991) for testing the homogeneity of the within-group discrepancies. Unfortunately, the value of the Bartlett T statistic is known to be very sensitive to the distribution of cases across groups (Manly 2007). It is therefore

---


## Page 13

<header>Studer et al.</header>
&lt;page_number&gt;483&lt;/page_number&gt;

**Table 3. Test of Homogeneity of the Within-Group Discrepancies**

<table>
  <thead>
    <tr>
      <th rowspan="2">Variable</th>
      <th colspan="2">Nonsquared dissimilarity</th>
      <th colspan="2">Squared dissimilarity</th>
      <th colspan="2">Nonsquared, unweighted</th>
    </tr>
    <tr>
      <th>L</th>
      <th>p</th>
      <th>L</th>
      <th>p</th>
      <th>L</th>
      <th>p</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>grammar</td>
      <td>25.10</td>
      <td>.001</td>
      <td>7.86</td>
      <td>.034</td>
      <td>1.15</td>
      <td>.282</td>
    </tr>
    <tr>
      <td>gcse5eq</td>
      <td>14.15</td>
      <td>.005</td>
      <td>7.94</td>
      <td>.021</td>
      <td>0.57</td>
      <td>.432</td>
    </tr>
    <tr>
      <td>funemp</td>
      <td>7.88</td>
      <td>.030</td>
      <td>6.00</td>
      <td>.054</td>
      <td>7.50</td>
      <td>.008</td>
    </tr>
    <tr>
      <td>religion</td>
      <td>7.87</td>
      <td>.034</td>
      <td>8.15</td>
      <td>.031</td>
      <td>14.92</td>
      <td>.001</td>
    </tr>
    <tr>
      <td>region</td>
      <td>4.31</td>
      <td>.040</td>
      <td>4.27</td>
      <td>.036</td>
      <td>5.83</td>
      <td>.000</td>
    </tr>
    <tr>
      <td>fmpr</td>
      <td>5.57</td>
      <td>.075</td>
      <td>3.52</td>
      <td>.156</td>
      <td>0.02</td>
      <td>.885</td>
    </tr>
    <tr>
      <td>livboth</td>
      <td>0.88</td>
      <td>.487</td>
      <td>1.13</td>
      <td>.439</td>
      <td>0.94</td>
      <td>.331</td>
    </tr>
    <tr>
      <td>sex</td>
      <td>0.00</td>
      <td>.994</td>
      <td>0.10</td>
      <td>.819</td>
      <td>4.50</td>
      <td>.033</td>
    </tr>
  </tbody>
</table>

Note: See Table I for definitions of variables.

unsuited for randomizations of weighted data that modify this distribution at each draw.

We consider here an alternative approach based on the Levene test (Brown and Forsythe 1974a). The Levene statistic is known to be powerful with randomization tests (Manly 2007). From a geometric point of view, when testing the homogeneity of discrepancy, we are interested in measuring differences in the radius of the distribution of sequences within each group. Radii may be measured from the contributions to the within sums of squares and their general equality tested with an ANOVA procedure. Let $z_{i\ell} = d^N(x_i, \bar{g}_\ell)$ be the dissimilarity between case $i$ and the gravity center of group $\ell$. The generalized Levene $L$ statistic for testing group homogeneity is then the $F$ statistic of this numeric variable. For the weighted case, it reads

$$
L = \frac{\sum_\ell w_\ell (\bar{z}_\ell - \bar{z})^2 / (m - 1)}{\sum_\ell \sum_i w_i (z_{i\ell} - \bar{z}_\ell)^2 / (W - m)}.
\quad (8)
$$

We propose again to evaluate the statistical significance of $L$ through permutation tests. Although we test here differences in real values, namely the dissimilarity $z_{i\ell}$ values, we do not recommend comparing the observed $L$ with the $F$ distribution, because those values are generally not normally distributed.

Table 3 provides the results of both tests for all considered covariates. As mentioned above, the Bartlett test is not reliable in the weighted case and is therefore shown only in the unweighted case. Using weighted data, we found,

---


## Page 14

&lt;page_number&gt;484&lt;/page_number&gt;
<header>Sociological Methods & Research 40(3)</header>

for instance, that young people who attended a grammar school have a discrepancy of their trajectories of 25.1, whereas those who did not have a discrepancy of 33.13. The difference is significative from our generalized Levene tests. In fact, grammar school opens the door to higher education trajectories, and most of the young Irish who attended a grammar school attempted to follow such a path, resulting in a significantly lower discrepancy. We may also note that children of unemployed fathers show a significantly higher discrepancy and that there are significant differences according to the religion as well as across regions.

**Simulation Study of the Tests' Behavior**

The aim of this section is to provide empirical insights into the behavior of the permutation tests of group differences. We study the tests on differences in both the general sequence pattern and the within discrepancy among sequences. We conducted a series of simulation studies to examine how the significance of the statistics evolves when we progressively change the characteristic sequence pattern of one of the groups. Three models (formally defined in Table 4) were considered for generating artificial data sets:

1. sequences with at most one transition and two possible states, with transition time (age at transition) generated with a normal model;
2. sequences with at most one transition and two possible states, with transition time generated through a more realistic shifted log-logistic model; and
3. sequences with at most three transitions and eight possible states derived from the combination of three events occurring according to log-logistic models.

The first retained model depicts a simple situation in which we can easily and independently control for the mean and variance of the age at which the transition occurs. The log-logistic model is more realistic for describing, for instance, the time to marriage or to a first childbirth. We retained a shifted version relevant for events that do not occur before a given age and studied the effect of varying once the start age (a positioning parameter) and once the log-logistic intensity (the inverse of the scale parameter) for the second group. With the third set of simulations, we investigate sequences that may contain up to four states out of the eight resulting from different combinations of three successive possible events. For example, if we assume that the three events of interest are leaving home (L), marriage (M), and childbirth (C), and that H stands for the initial state, the eight

---


## Page 15

<header>Studer et al.</header>
&lt;page_number&gt;485&lt;/page_number&gt;

possible states would be H, L, M, C, LM, LC, MC, and LMC. In that example, M would mean married before leaving home and MC married with a child at the parents’ home. The occurrences of the events are each modeled with a different log-logistic distribution, and we vary only the parameters of one of them.

The normal model, though not very realistic for describing the hazard of a transition, permits independent examination of the effects of changes in position and in dispersion. The other models are more realistic. The log-logistic distribution is, for instance, used in parametric approaches of event history analysis (Blossfeld and Rohwer 2002). What makes it particularly interesting is that it allows for nonmonotone risks. It is characterized by an intensity parameter $\lambda$ and a shape parameter $b$. The inverse of $\lambda$ is known as the scale parameter, and it is also the median of the distribution. Hence, an increase in $\lambda$ reduces discrepancy but also changes the location. To control positioning independently of the discrepancy, we consider a start $a$ parameter that specifies the threshold age at which the log-logistic risk starts. The retained values for its $b$ shape and $\lambda$ intensity parameter are based on estimates obtained by Billari (2001b) in an analysis of age at first marriage in Italy. Finally, the last model considers multiple events and states that correspond typically to situations encountered in life-course analysis.

The simulation design is as follows. For each of the three types of models, we start with the same set of parameters for the two groups and increase progressively in 20 steps one of the parameters for the second group while keeping the other parameters unchanged. At each step, we generate 1,000 artificial sets of sequences of length 40, with each set composed of 1,000 sequences for the reference group and 1,000(1 - $\theta$) for the second group, with $\theta$ being an arbitrarily chosen proportion. For each artificial set, we compute the pairwise OM dissimilarities with a constant substitution cost of two and an indel cost of one. We then derive from them the values of the $F$ and $L$ statistics and their permutation test $p$ values, using once nonsquared dissimilarities ($v = 1$) and once squared dissimilarities ($v = 2$).

The examined varying parameters are the mean age $a$ for the normal model and the start age $a$ for the log-logistic model. Those are location parameters. For changes in discrepancy, we varied the standard deviation of the normal model and the intensity parameter of the log-logistic model. In the more complex case with multiple states generated from three log-logistics, we varied the parameters of one log-logistic distribution only, namely, the one generating the second event.

Figure 2 summarizes the simulation results for a balanced distribution between the two groups (i.e., for $\theta = .5$).³ Results reported are the percentage

---


## Page 16

<table>
   <thead>
    <tr>
     <td colspan="4">
      <b>
       Table 4. Random Models Used for Generating the Simulated Data
      </b>
     </td>
    </tr>
    <tr>
     <td>
      <b>
       Transitions Parameters
      </b>
     </td>
     <td>
      <b>
       Normal
      </b>
     </td>
     <td>
      <b>
       Log-Logistic
      </b>
     </td>
     <td>
      <b>
       Multiple Log-Logistic
      </b>
     </td>
    </tr>
   </thead>
   <tbody>
    <tr>
     <td>
      States
     </td>
     <td>
      <i>
       t
      </i>
      ~
      <i>
       N
      </i>
      (
      <i>
       a
      </i>
      ,
      <i>
       σ
      </i>
      )
      <i>
       a
      </i>
      mean age
      <i>
       σ
      </i>
      age standard deviation
     </td>
     <td>
      <i>
       t
      </i>
      ~
      <i>
       a
      </i>
      +
      <i>
       L
      </i>
      (
      <i>
       λ
      </i>
      ,
      <i>
       b
      </i>
      )
      <i>
       a
      </i>
      start age
      <i>
       λ
      </i>
      intensity (inverse scale)
      <i>
       b
      </i>
      shape
     </td>
     <td>
      <i>
       t
      </i>
      <sub>
       <i>
        i
       </i>
      </sub>
      ~
      <i>
       a
      </i>
      <sub>
       <i>
        i
       </i>
      </sub>
      +
      <i>
       L
      </i>
      (
      <i>
       λ
      </i>
      <sub>
       <i>
        i
       </i>
      </sub>
      ,
      <i>
       b
      </i>
      <sub>
       <i>
        i
       </i>
      </sub>
      ),
      <i>
       i
      </i>
      = 1, 2, 3
     </td>
    </tr>
    <tr>
     <td>
      Constant parts
     </td>
     <td>
      <i>
       E
      </i>
      <sub>
       1
      </sub>
      for
      <i>
       i
      </i>
      &lt;
      <i>
       t
      </i>
      <i>
       E
      </i>
      <sub>
       2
      </sub>
      for
      <i>
       i
      </i>
      ≥
      <i>
       t
      </i>
     </td>
     <td>
      <i>
       E
      </i>
      <sub>
       1
      </sub>
      for
      <i>
       i
      </i>
      &lt;
      <i>
       t
      </i>
      <i>
       E
      </i>
      <sub>
       2
      </sub>
      for
      <i>
       i
      </i>
      ≥
      <i>
       t
      </i>
     </td>
     <td>
      <i>
       a
      </i>
      <sub>
       1
      </sub>
      ,
      <i>
       a
      </i>
      <sub>
       2
      </sub>
      ,
      <i>
       a
      </i>
      <sub>
       3
      </sub>
      start ages
      <i>
       λ
      </i>
      <sub>
       1
      </sub>
      ,
      <i>
       λ
      </i>
      <sub>
       2
      </sub>
      ,
      <i>
       λ
      </i>
      <sub>
       3
      </sub>
      intensities
      <i>
       b
      </i>
      <sub>
       1
      </sub>
      ,
      <i>
       b
      </i>
      <sub>
       2
      </sub>
      ,
      <i>
       b
      </i>
      <sub>
       3
      </sub>
      shapes
     </td>
    </tr>
    <tr>
     <td>
      Reference models
     </td>
     <td>
      <i>
       t
      </i>
      ~
      <i>
       N
      </i>
      (20, 4)
     </td>
     <td>
      <i>
       t
      </i>
      ~ 0 +
      <i>
       L
      </i>
      (0.126, 2.364)
     </td>
     <td>
      <i>
       t
      </i>
      <sub>
       1
      </sub>
      ~ 0 +
      <i>
       L
      </i>
      (0.078, 2.364)
     </td>
    </tr>
    <tr>
     <td>
      For position change
     </td>
     <td>
      <i>
       t
      </i>
      ~
      <i>
       N
      </i>
      (20, 4)
     </td>
     <td>
      <i>
       t
      </i>
      ~ 0 +
      <i>
       L
      </i>
      (0.078, 2.364)
     </td>
     <td>
      <i>
       t
      </i>
      <sub>
       2
      </sub>
      ~ 10 +
      <i>
       L
      </i>
      (0.126, 2.364)
     </td>
    </tr>
    <tr>
     <td>
      For scale change
     </td>
     <td>
      <i>
       t
      </i>
      ~
      <i>
       N
      </i>
      (20, 4)
     </td>
     <td>
      0 ≤
      <i>
       a
      </i>
      ≤ 5
     </td>
     <td>
      <i>
       t
      </i>
      <sub>
       2
      </sub>
      ~ 10 +
      <i>
       L
      </i>
      (0.078, 2.364)
     </td>
    </tr>
    <tr>
     <td>
      Variations
     </td>
     <td>
      20 ≤
      <i>
       a
      </i>
      ≤ 25
     </td>
     <td>
      0.078 ≤
      <i>
       λ
      </i>
      ≤ 0.205
     </td>
     <td>
      10 ≤
      <i>
       a
      </i>
      ≤ 15
     </td>
    </tr>
    <tr>
     <td>
     </td>
     <td>
      4 ≤
      <i>
       σ
      </i>
      ≤ 7
     </td>
     <td>
     </td>
     <td>
      0.078 ≤
      <i>
       λ
      </i>
      <sub>
       2
      </sub>
      ≤ 0.205
     </td>
    </tr>
    <tr>
     <td>
     </td>
     <td>
     </td>
     <td>
     </td>
     <td>
      8 states from the combination of 3 events
     </td>
    </tr>
   </tbody>
  </table>
&lt;page_number&gt;486&lt;/page_number&gt;

---


## Page 17

<header>Studer et al.</header>
&lt;page_number&gt;487&lt;/page_number&gt;

&lt;img&gt;
Figure 2. Simulation Results
&lt;/img&gt;

of $p$ values < .05 at each of the stepped values of the considered varying parameter. Let us first look at the results for the $F$ tests (blue curves). For the location parameter $a$ (left plots), we acheive better significance when we use nonsquared dissimilarities $v = 1$ than with $v = 2$ in the log-logistic case, while there are no clear differences between $v = 1$ and $v = 2$ in the normal case.

When we vary the scale parameter (i.e., the within-group discrepancy [right plots]), we observe that $F$ becomes significant with $v = 1$ in the normal case, which is a fallacious effect, because we do not change location in that case. With $v = 2$, we obtain, as expected, nonsignificant $F$ values. In the log-logistic case, $F$ becomes also significant when $\lambda$ increases, but it is not surprising here, because $\lambda$ also determines the position.

---


## Page 18

&lt;page_number&gt;488&lt;/page_number&gt;
Sociological Methods & Research 40(3)

We now look at the L tests for the difference in within-group discrepancy. Unsurprisingly, when we vary the location parameter while maintaining the scales at the same level, the tests remain nonsignificant. When we vary the scale parameter, we observe similar significance of the L tests in the normal case, while v = 1 dominates v = 2 in the log-logistic case.

Results for the third set of simulations are similar to those of the single log-logistic, though differences between v = 1 and v = 2 are somewhat smoothed. This may be attributable to higher censoring of the concerned log-logistic that results from the occurrences of the other events.

Because data encountered in social sciences look more like those generated by the log-logistic models than like normal data, the use of v = 1 (i.e., nonsquared dissimilarities) seems preferable in view of these simulation results. This goes in the direction of what we advocate in Appendix B on the basis that v = 1 guarantees the nonnegativity of the contribution to the sum of squares when the dissimilarity satisfies triangle inequality. Altogether, we promote testing differences with v = 1 and confirming the conclusion with the squared version when the Levene test exhibits significant differences in within discrepancies.

## Studying and Rendering Group Differences

From the previous results, we learn not only that the trajectories of grammar school students differ significantly from the trajectories of students attending other schools but also that the discrepancy of their trajectories is significantly lower. Nevertheless, these results by themselves do not tell us anything about how the trajectories differ among the different groups. To gain an idea of possibly existing differences, it is useful to display them visually. For example, index plots (Scherer 2001) can be used, whereby each sequence is represented by a timeline split into segments colored according to the corresponding occupational state. To account for weights, the line width can be adjusted according to the weight of the represented trajectory.

Figure 3 displays such weight-adjusted index plots of grammar school and non-grammar school trajectories. Furthermore, to improve readability, we ordered the sequences according to the first dimension of a weighted principal coordinate analysis (Gower 1966).<sup>4</sup> Although ordering sequences by a principal coordinate facilitates the interpretation of the index plot, the plots provide conversely useful information for interpreting the principal coordinate axis. For instance, we observe in our case that the sequences are organized in a continuum ranging from higher education trajectories to training trajectories, while middle values correspond to employment-dominated

---


## Page 19

<header>Studer et al.</header>
&lt;page_number&gt;489&lt;/page_number&gt;

&lt;img&gt;Figure 3. Trajectories of Grammar and Non-Grammar School Students&lt;/img&gt;

trajectories. Comparing both populations, it appears that young people who attended grammar schools typically remain in the “school” state and are more likely to proceed to higher education, while those who attended other types of schools follow more diverse trajectories.

Figure 4 shows the evolution of the strength of association between the grammar covariate and a sliding six-month-long subsequence of the trajectory. We choose a window length of six months, which corresponds to a concrete horizon for most respondents. It has the effect of smoothing the curves while still rendering the main changes along time. The OM dissimilarity

---


## Page 20

&lt;page_number&gt;490&lt;/page_number&gt;
Sociological Methods & Research 40(3)

&lt;img&gt;Figure 4. Time Evolution of Pseudo-R² and L, Six-Month Sliding Windows&lt;/img&gt;

matrix was calculated for each six-month windows and served for deriving the share of explained discrepancy and the value of the pseudo-Levene statistic. Observing the evolution of these two statistics helps in identifying the periods over which the sequences differ the most. We observe that attending a grammar school has a long-term effect, with the strongest association appearing near the end of the studied trajectory. The curve of the Levene statistic indicates large differences in the discrepancy at the beginning of the sequences, where most grammar students continue in the same school state while nongrammar students experience more diverse trajectories.

It is interesting also to look at how the within discrepancy evolves inside each group. Figure 5 depicts these evolution patterns using the same six-month sliding windows. It shows the discrepancy for each value of the Grammar variable and the overall discrepancy for comparison. We can see that the discrepancy of non-grammar school students gradually declines, which may be explained by the increasing number of youngsters who reach a final “employment” status. On the contrary, the discrepancy among grammar school students peaks after two years, when some switch to higher education while others enter the labor market. We also observe that the differences of within-group discrepancies diminish over time, which is in accordance with the evolution of the Levene statistics presented above.

Depicting the evolution of the discrepancy over time can be seen as an alternative to studying the sequence of transversal entropies measuring state diversity at each time position (Billari 2001a; Widmer and Ritschard 2009). The latter approach is an analysis with windows of length one. Moreover, it can be shown that for such one-period windows, the transversal Gini index, also known as quadratic entropy, is exactly the discrepancy derived from Hamming distances with a unique substitution cost (Geurts et al. 2006). With lengthier windows, the solution proposed here permits accounting for discrepancies in both the sequencing and the temporality of the states.

---


## Page 21

<header>Studer et al.</header>
&lt;page_number&gt;491&lt;/page_number&gt;

&lt;img&gt;Figure 5. Time Evolution of Within-Group and Overall Discrepancies, Six-Month Sliding Windows&lt;/img&gt;

The proposed discrepancy analysis includes not only measuring the influence of a factor on the trajectories but also depicting their diversity, both statistically and graphically. Conceptually, it aims to depict the influence of structures on the trajectory construction while assessing the ability for auto-determination within the structure.

Such an explanatory methodological framework based on the notion of discrepancy adheres particularly well to the life-course paradigm. Indeed, in his formalization of this paradigm, Elder (1999) highlighted the importance of studying the sociohistorical context of the individuals as well as their ability to make their own choices within that context. This is precisely what discrepancy analysis does. It allows for studying the links between individuals’ trajectories and their contexts, while at the same time preserving the notion of between-individual variability.

## Multifactor Discrepancy Analysis

Above, we examined how to measure the bivariate association between the trajectory and each of the covariates considered independently. We consider here the generalization to the multifactor case and, following the work of McArdle and Anderson (2001), adopt the framework of the general multivariate ANOVA.

Formally, let Y be the n × q matrix with n observed values of q centered variables. In his work on principal coordinate analysis, Gower (1966, 1982) showed that the outer product YY', which has the sum of squares on its diagonal, can be expressed in terms of the matrix of pairwise squared Euclidean distances. McArdle and Anderson (2001) derived their generalized multifactor multivariate ANOVA by combining this result with a multivariate linear model and replacing the Euclidean distances with dissimilarities. Here, we extend their proposition to account for weights.

---


## Page 22

&lt;page_number&gt;492&lt;/page_number&gt;
Sociological Methods & Research 40(3)

Let X be the n × m matrix with the values of m predictors—contrasts for coding M factors—including a first column consisting of ones for the constant and Ŷ the matrix of values predicted from X with the linear model. The sum of total weighted sums of squares (SS<sub>T</sub>) may be partitioned into predicted (SS<sub>B</sub>) and residual (SS<sub>R</sub>) weighted sums of squares,
$$
tr(Y'WY) = tr(Ŷ'WŶ) + tr(R'WR), \quad (9)
$$
where R is the matrix of residuals, and W is the weight diagonal matrix. According to the weighted linear regression model, we have WŶ = HWY and WR = (I - H)WY, where H = WX(X'WX)<sup>-1</sup>X'W is the symmetric idempotent “hat” matrix that is adapted for the weighted case here. Because W = W<sup>1/2</sup>W<sup>1/2</sup>, using the property tr(AB) = tr(BA) for conformable matrices, equation (9) may be rewritten as
$$
tr(WYY') = tr(W^{1/2}HW^{1/2}YY') + tr[W^{1/2}(I - H)W^{1/2}YY']. \quad (10)
$$
Let us now look at Gower’s result that expresses G = YY' in terms of the pairwise squared distances. In the formulation retained by McArdle and Anderson (2001), we have G = -1/2(I - 1/n11')D(I - 1/n11'), where 1 is a vector of ones of length n, I is the identity matrix, and D is the n × n matrix of the squared pairwise Euclidean distances. Here, we have to adapt this definition for y variables centered on their weighted means. In that case, it reads as follows:
$$
G = -\frac{1}{2}(I - \frac{1}{W}11'W)D(I - \frac{1}{W}W11'). \quad (11)
$$
The generic element of G is g<sub>ij</sub> = -1/2(d<sub>ij</sub><sup>2</sup> - d̄<sub>i</sub><sup>2</sup> - d̄<sub>j</sub><sup>2</sup> + d̄<sup>2</sup>), where d̄<sub>i</sub><sup>2</sup>, d̄<sub>j</sub><sup>2</sup>, and d̄<sup>2</sup> are, respectively, the row i, the column j, and the overall weighted average of the squared Euclidean distances. It can be shown that when q = 1, each diagonal element g<sub>ii</sub> of G is the contribution of the ith individual to the sum of squares, as defined in equation (4).<sup>6</sup>
Setting H<sub>W</sub> = W<sup>1/2</sup>H<sub>W</sub>W<sup>1/2</sup>, the total SS<sub>T</sub>, between SS<sub>B</sub>, and within SS<sub>W</sub> sums of squares of interest may be rewritten in terms of this adapted G matrix as
$$
SS_T = tr(WG), \quad (12)
$$
$$
SS_B = tr(H_WG), \quad (13)
$$
$$
SS_W = tr[(W - H_W)G]. \quad (14)
$$

---


## Page 23

<header>Studer et al.</header>
&lt;page_number&gt;493&lt;/page_number&gt;

The idea is to substitute the pairwise dissimilarities $d_{ij}^Y$ for the squared Euclidean distances that define D in equation (11). Assuming such a substitution and using equations (12) to (14), we can derive global pseudo-$R^2$ and pseudo-$F$ statistics as defined in equations (6) and (7). Instead of the number of groups, $m$ should be set here to the number of columns of X (i.e., to the total number of contrast and/or indicator variables necessary for coding the M factors).

For $M = 1$ (i.e., in the case of a single factor), computing $SS_T$ and $SS_W$ with equation (2) gives exactly the same results as the matrix formulation considered here. However, the direct computation of the sums of squares is about 10 times faster.

We may also consider the contribution of each covariate to the total discrepancy reduction. As with multifactor ANOVA, there are different ways of looking at these individual contributions. Shaw and Mitchell-Olds (1993) distinguished, among others, between two methods called Type I and Type II. The Type I method is incremental, which means that covariates are successively added to the model and the contribution of each covariate is measured by the $SS_B$ increase that results when it is introduced. With this method, the measured impact of each covariate depends on the order in which the covariates are introduced. With the Type II method, known to be robust in the absence of interaction effects, the contribution of each covariate is measured by the reduction of $SS_B$ that occurs when we drop it from the full model (i.e., from the model with all covariates). We retain the second method and hence compute the following $F$ for each covariate $v$:

$$
F_v = \frac{(SS_{B_c} - SS_{B_v})/p}{SS_{W_c}/(W - m - 1)},
\quad (15)
$$

where $SS_{B_c}$ and $SS_{W_c}$ are the explained and residual sums of squares of the full model, $SS_{B_v}$ is the explained sum of squares of the model after removing variable $v$, and $p$ is the number of indicators or contrasts used to encode the covariate $v$.

As in the single discrepancy analysis, the $F$ distribution is not relevant for the pseudo-$F$, and we consider again permutation tests for assessing the significance of the $F$ statistic. Because $F_v$ is intended for testing the conditional independence of $v$, its null distribution is obtained by permuting only the covariate $v$, while the global $F$ statistic is computed by permuting the whole profiles. Thus, for a complete multifactor analysis with profiles defined by $M$ factors, $1 + M$ permutation tests are required, which may be quite time-consuming.

---


## Page 24

&lt;page_number&gt;494&lt;/page_number&gt;
Sociological Methods & Research 40(3)

Let us look at what a multifactor analysis gives for our illustrative example. Table 5 shows the results for two models: the complete model with all variables and a model obtained after removing nonsignificant covariates through a backward stepwise process. The tests were conducted using 5,000 permutations.

Both models provide overall significant information about the discrepancy of the trajectories, because both global F statistics are significant. The full model explains a slightly higher part of the discrepancy ($R^2 = .190$) than does the backward model ($R^2 = .182$), but it contains nonsignificant covariates.

In the full model, the variable gcse5eq (“qualification gained at the end of compulsory education”) is the most significant covariate. If we remove this variable, the $R^2$ value of the model (.190) decreases by .060. The difference is significant, because we have $F_{gcse5eq} = 51.91$, which was never attained with 5,000 permutations. As before, the variable religion is not significant. Removing it from the model reduces the $R^2$ value by only .003 and results in $F_{religion} = 2.29$ and a $p$ value of .208. Likewise, the variable fmpr (“having a professional, managerial or related father”) loses its significance in the multifactor case. In fact, the variable fmpr becomes nonsignificant as soon as we control for funemp (“father’s unemployment”), because the two variables are strongly correlated and funemp is the most significant one.

The multifactor approach provides information about the proper effect of the covariates on the occupational trajectory (i.e., about the part of the total effect that is not accounted for by factors that are already introduced). In that sense, the multifactor approach is complementary to the single univariate discrepancy analysis, which informs on the raw effect of each covariate. Nevertheless, although the multifactor approach permits us to know which effects are significant, it does not tell us much about what the effects are (i.e., about how trajectories may change with the value of the covariates). To answer such questions, we propose a tree approach, which can be seen as an extension of the graphical display shown in Figure 3.

## Tree-Structured Analysis of Sequences

In this section, we complement the sequence discrepancy analysis with the regression tree method introduced in Studer et al. (2009, 2010), which we extend to account for weighted sequences. Regression trees work as follows (Breiman, Friedman, Olshen, and Stone 1984; Morgan and Sonquist 1963). They start with all individuals grouped in an initial node. Then, they recursively partition each node using values of a predictor. At each node, the

---


## Page 25

<header>Studer et al.</header>
&lt;page_number&gt;495&lt;/page_number&gt;

**Table 5. Multifactor Discrepancy Analysis**

<table>
  <thead>
    <tr>
      <th rowspan="2">Variable</th>
      <th colspan="2">Full Model</th>
      <th colspan="2">Backward Model</th>
    </tr>
    <tr>
      <th>$F_v$</th>
      <th>$\Delta R_v^2$</th>
      <th>Significance</th>
      <th>$F_v$</th>
      <th>$\Delta R_v^2$</th>
      <th>Significance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>gcse5eq</td>
      <td>51.91</td>
      <td>.060</td>
      <td>.000</td>
      <td>55.72</td>
      <td>.065</td>
      <td>.000</td>
    </tr>
    <tr>
      <td>grammar</td>
      <td>20.77</td>
      <td>.024</td>
      <td>.000</td>
      <td>21.44</td>
      <td>.025</td>
      <td>.000</td>
    </tr>
    <tr>
      <td>sex</td>
      <td>5.47</td>
      <td>.006</td>
      <td>.002</td>
      <td>5.30</td>
      <td>.006</td>
      <td>.003</td>
    </tr>
    <tr>
      <td>funemp</td>
      <td>3.59</td>
      <td>.004</td>
      <td>.039</td>
      <td>3.83</td>
      <td>.004</td>
      <td>.028</td>
    </tr>
    <tr>
      <td>fmpr</td>
      <td>3.30</td>
      <td>.004</td>
      <td>.054</td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>region</td>
      <td>3.19</td>
      <td>.015</td>
      <td>.004</td>
      <td>3.37</td>
      <td>.016</td>
      <td>.003</td>
    </tr>
    <tr>
      <td>religion</td>
      <td>2.29</td>
      <td>.003</td>
      <td>.212</td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>livboth</td>
      <td>1.80</td>
      <td>.002</td>
      <td>.405</td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Global</td>
      <td>$F_{tot}$ 14.96</td>
      <td>$R^2_{tot}$ .190</td>
      <td>.000</td>
      <td>$F_{tot}$ 19.55</td>
      <td>$R^2_{tot}$ .182</td>
      <td>.000</td>
    </tr>
  </tbody>
</table>

Note: See Table I for definitions of variables.

predictor and the split are chosen in such a way that the resulting child nodes differ as much as possible from one another or have, more or less equivalently, lowest within-group discrepancy. The process is repeated on each new node until a certain stopping criterion is reached.

The recursive partitioning generated by a tree is known to provide an easily comprehensible view of how each newly selected covariate nuances the effect of covariates introduced at earlier levels. This requires the display of relevant information about the distribution at each node. We could represent the medoid (i.e., the observed sequence that minimizes the dissimilarity, equation [4], between the sequence and the group gravity center). It would be instructive to render the within-group discrepancy as well. Although this is not obvious for any kind of complex objects, displaying index plots such as those used in Figure 3 provides a good solution for state sequences. For a somewhat more synthetic view, we could also consider representative plots (Gabadinho, Ritschard, Studer, and Müller 2011) that show the minimal set of sequences for each node that would be necessary to ensure a given coverage of the sequences at that node.

Beside the displayed node content, the originality of the tree-structured analysis of sequences resides in the use of a splitting criterion derived from the pairwise dissimilarities, namely the univariate pseudo-$R^2$ described previously. At each node, we select thus the predictor and the binary split for which we get the highest pseudo-$R^2$ value (i.e., the split that accounts for the greatest part of the object discrepancy). An alternative would be to use the

---


## Page 26

&lt;page_number&gt;496&lt;/page_number&gt;
Sociological Methods & Research 40(3)

significance of the univariate pseudo-F. However, because this significance must be determined through permutation tests, the time complexity would be excessive if we had to repeat it for each predictor and possible split. Therefore, we consider the F significance only as a stopping criterion; that is, we stop growing a branch as soon as we get a nonsignificant F value for the selected split. In that way, permutations need to be run only once at each node, which remains tractable. The extension of this tree method for weighted cases is obtained by using the pseudo-R² formula and the F testing method that we propose in this article at each step of the tree-growing process.

Using the pseudo-R² as the splitting criterion inevitably means that we could only build binary trees. The R² does not penalize for the number of groups and would hence always select the maximum number of groups if we allowed n-ary splits. Using the R² adjusted for the number of groups, as it is used in multiple regressions, would not solve the problem, because the adjusted R² is known to insufficiently penalize complexity. Using information criteria such as the Bayesian information criterion would also not be suitable, as such criteria are hardly derivable in a case where the distribution of the statistics (R², F, or SSw) under the independence hypothesis is not known.

The global quality of the tree can be assessed through the association strength between the sequences and the leaf (terminal node) membership. The global pseudo-F provides a way of testing the statistical significance of the obtained segmentation, while the global pseudo-R² provides a measure of the part of the total discrepancy that is explained by the tree.

Figure 6 shows the dissimilarity tree grown using our weighed example data set. The chosen stopping criteria are a p value of 5 percent for the F test, a minimal leaf size of 5 percent of the total sum of weights, and a maximal depth of 5. In each node, we see the plot of the individual sequences as well as the node size, the sum of weights, and the discrepancy s² within the node. At the bottom of each parent node, we indicate the retained split predictor with the associated R², while the definition of the binary split may be inferred from the indication at the top of the child nodes.

The overall R² value of the tree is .187, which falls between the global R² value of the full and backward models in Table 5. However, the results are now much easier to interpret. Moreover, the tree automatically accounts for interaction effects that were not considered in the multifactor discrepancy analysis. We observe, for instance, that “attending grammar school” discriminates better among students who finished the compulsory schooling with high grades than among those who obtained lower grades. Likewise, we can see that having an unemployed father seems to affect primarily young male Irish with low grades at the end of compulsory schooling (gcse5eq).

---


## Page 27

<header>Studer et al.</header>
&lt;page_number&gt;497&lt;/page_number&gt;

&lt;img&gt;Sequence Regression Tree diagram showing a hierarchical tree structure with nodes labeled by demographic categories (e.g., [yes], [no], [female], [male], [S.Eastern]) and statistical metrics (n, s2, R2 values). The root node is labeled "Root n: 712 s2: 33.0" with a legend indicating sequence types (Employment, Further Education, Higher Education, Joblessness, School, Training). The tree branches into splits based on variables like "funemp," "Grammar," "sex," and "region," with terminal nodes showing sequence visualizations.&lt;/img&gt;

Figure 6. Sequence Regression Tree

## Running Sequence Discrepancy Analysis in R With TraMineR

We implemented the methods presented in this article into TraMineR, (Gabadinho, Ritschard, Müller, and Studer 2011) which is a free package for the R statistical environment (R Development Core Team 2008). Below, we briefly show how to run the discrepancy analysis features discussed here. We would also like to refer our readers to the TraMineR user’s guide (Gabadinho, Ritschard, Studer, and Müller 2009), which provides a detailed overview of other features offered by the package, especially of the rendering of sequences and the computation of dissimilarities. Our readers can reproduce (with the R script provided on the Web site of the journal) the results we present here, as the mvad data set, which we use, has been made available as part of the TraMineR package, thanks to the authorization of McVicar and Anyadike-Danes (2002).

---


## Page 28

&lt;page_number&gt;498&lt;/page_number&gt;
Sociological Methods & Research 40(3)

To begin the analysis, we load the mvad data and create a weighted state sequence object using the commands below. The state variables from September 1993 to June 1999 are in columns 17 to 86 of the data frame.⁷

R> library(TraMineR)
R> data(mvad)
R> mvadseq <- seqdef(mvad[, 17:86], weights = mvad$weight)

Next, we compute the OM dissimilarity matrix with the indel and substitution costs used by McVicar and Anyadike-Danes (2002):

R> subm.custom <- matrix(
c(0,1,1,2,1,1,
1,0,1,2,1,2,
1,1,0,3,1,2,
2,2,3,0,3,1,
1,1,1,3,0,2,
1,2,2,1,2,0),
nrow = 6, ncol = 6, byrow = TRUE)
R> mvaddist <- seqdist(mvadseq, method = "OM", indel = 1.5, sm = subm.custom)

To perform the univariate discrepancy analysis and to test for homogeneity of discrepancy, we call the dissassoc() function, which takes four arguments: the dissimilarity matrix, the factor (group), the number of permutations (R = 1,000 by default), and an optional weights argument. The results presented above in “Comparing Groups of Sequences” were obtained with the following code:

R> dissassoc(mvaddist, group = mvad$gcse5eq, R = 5000, weights = mvad$weight, weight.permutation = "diss")

Likewise, we generated Figures 4 and 5 with seqdiff() as shown below:

R> Grammar.diff <- seqdiff(mvadseq, group = mvad$Grammar, seqdist_arg = list(method = "OM", indel = 1.5, sm = subm.custom))
R> plot(Grammar.diff, stat = c("Pseudo R2", "Levene"))
R> plot(Grammar.diff, stat = "discrepancy")

The multifactor results listed in Table 5 were obtained with the dissmfacw() function. The model is specified as a classical R formula

---


## Page 29

<header>Studer et al.</header>
&lt;page_number&gt;499&lt;/page_number&gt;

with the dissimilarity matrix on the left-hand side. We use the data argument to specify the data frame containing the covariates⁸:

```R
dissmfacw(
  mvaddist ~ gcse5eq + Grammar + funemp + catholic +
    male + fmpr + livboth + region, data = mvad,
  R = 5000, weights = mvad$weight)
```

To carry out a tree-structured analysis of the sequences, we use the `seqtree()` function. The dissimilarity matrix and the predictors are passed to the function in the same way as in `dissmfacw()`. Stopping criteria can be set with the arguments `minSize` for the minimum node size, `maxdepth` for the maximum tree depth, and `pval` for the minimum required $p$ value. As for `dissassoc()`, the `R` argument controls the number of permutations for computing the $p$ values. Notice that it is not necessary to specify the weights, because they are already attached to the state sequence `mvadseq` object.

```R
mvadtree <- seqtree(
  mvadseq ~ gcse5eq + Grammar + funemp + catholic +
    male + fmpr + livboth + region, data = mvad,
  minSize = 30, maxdepth = 5, R = 5000, pval = 0.01,
  diss = mvaddist, weight.permutation = "diss")
print(mvadtree)
```

The `print()` command produces a text output of the tree. The tree can also be plotted with `seqtreedisplay()`. This function uses the free GraphViz software (Gansner and North 1999).⁹ Hence, it must be installed and accessible for the function to work properly. The tree in Figure 6 was obtained with the following command:

```R
seqtreedisplay(mvadtree, type = "I",
  sortv = cmdscale(sqrt(mvaddist), k = 1))
```

## Conclusions

In this article, we have proposed a set of tools for analyzing the relationship between discrete sequences and one or more covariates. Besides the fact that the methods we propose are of interest for the analysis of state sequence, we believe that they also provide an innovative alternative to the traditional cluster-based sociological analysis of life trajectories.

---


## Page 30

&lt;page_number&gt;500&lt;/page_number&gt;
Sociological Methods & Research 40(3)

The starting point of the new methodology introduced here is the definition of the discrepancy of the sequences in terms of their pairwise dissimilarities. Afterwards, the methods proceed with the transposition of the concepts of ANOVA to this generalized discrepancy framework. They include single and multifactor ANOVAs, the measure of the strength of sequence-covariate associations with pseudo-R² values, a generalized Levene test of equality of within-group discrepancies, tools and plots for investigating the evolution of the group differences along the time frame, and a regression tree method for sequence data. Because normality of sequences is not defendable, the statistical significance of the proposed statistics is assessed through permutation tests. Up to this point, similar approaches have already been considered in the literature, but only for nonsequence complex objects such as ecosystems. In addition to the application on sequence data, the generalized Levene test and the procedure accounting for case weights in the measures and tests are the main original methodological contributions of this article.

As far as sociological analysis is concerned, the proposed methodology opens new perspectives besides the traditional cluster-based approach. In short, this cluster-based approach consists of associating each trajectory in a given set to some related ideal type. From a descriptive standpoint, this approach has proven to be effective in uncovering the underlying structure of a set of sequences, which makes the data easier to understand. However, relying on clusters for studying the relationship between sequences and their context can be criticized on the basis that reducing the set of sequences to a limited number of standard trajectories is a rather crude approximation and would lead to considering deviations from the standard inside a cluster as nonexplained error terms. As a result of this approximation, wrong conclusions may be drawn about relationships between the sequences and their context. On the other hand, the approach we propose here takes into account explicitly how the individual characteristics affect the trajectory followed by each individual.

Furthermore, in this article, we adopt an explanatory methodological framework that complies with the life course paradigm (Elder 1999) by accounting for the individuals' ability to make their own choices within their sociohistorical backgrounds when estimating the sequence-context relationship. By focusing on the discrepancy of the sequences, they allow studying the link between the trajectories and their context while preserving the notion of between-individual variability.

The choice of the measure of dissimilarity between sequences is a recurrent debate in the social sciences (Dijkstra and Taris 1995; Elzinga 2003; Wu 2000), which is beyond the scope of the present article. Although we have

---


## Page 31

<header>Studer et al.</header>
&lt;page_number&gt;501&lt;/page_number&gt;

illustrated the methods using an OM edit distance, the methods considered in this article are in no way limited to OM. They work with any dissimilarity measure. Moreover, running the statistical tests with different dissimilarity measures provides a way of assessing their respective discriminant power for the data at hand. Also, using, for instance, the multichannel approach considered by Pollock (2007), the proposed methods could be applied on parallel sequences such as those describing, for example, linked lives or joint occupational and cohabitational trajectories. The observed differences between groups could then result from any of the channel, or from the combination of channels. Even more generally, the discrepancy analysis is not limited to sequence data. If we except the graphical rendering of the results, they apply to any objects that can be characterized by a pairwise dissimilarity matrix.

Finally, we would like to remind our readers that all the proposed tools have been implemented in the TraMineR library for the R statistical environment. They are thus readily and freely accessible to any interested reader, as illustrated above.

**Acknowledgments**

We gratefully thank Nevena Zhelyazkova for her careful reading of this article and the anonymous reviewers for their constructive comments and suggestions.

**Declaration of Conflicting Interests**

The author(s) declared no potential conflicts of interest with respect to the research, authorship, and/or publication of this article.

**Funding**

This research was supported by Grant SNSF 122230 from the Swiss National Science Foundation.

**Appendix A: Proofs**

In this appendix, we present the mathematical developments underlying the results presented in the article. The presentation is largely inspired by Späth (1975) and Batagelj (1988).

We begin with a proof of equation (1) that expresses the sum of squares in terms of pairwise Euclidean distances. Here we demonstrate it for the more general multivariate case. Let $y_i$ be the data vector for case $i$, $w_i$ its associated weight, $W = \sum_{i=1}^n w_i$ the sum of the weights, and $\bar{y} = \frac{1}{W} \sum_{i=1}^n w_i y_i$ the vector of weighted averages. Letting $||y||^2$ denote the squared length of the

---


## Page 32

&lt;page_number&gt;502&lt;/page_number&gt;
Sociological Methods & Research 40(3)

vector, that is y'y = Σy_i^2, the multivariate result that we want to establish is as follows.

**Theorem 1: Sum of Squares in Terms of Pairwise Distances**
SS = Σ_{i=1}^n w_i ||y_i - \bar{y}||^2 = \frac{1}{W} Σ_{i=1}^n Σ_{j=i+1}^n w_i w_j ||y_i - y_j||^2. (A1)

**Proof.** We first show that the sum of squared distances to a point x is
Σ_{i=1}^n w_i ||y_i - x||^2 = Σ_{i=1}^n w_i ||y_i - \bar{y}||^2 + W ||\bar{y} - x||^2. (A2)
Because y_i - x = (y_i - \bar{y}) + (\bar{y} - x), its squared length is
||y_i - x||^2 = ||y_i - \bar{y}||^2 + 2(\bar{y} - x)'(y_i - \bar{y}) + ||\bar{y} - x||^2.
Weighting with w_i and summing over i, we obtain
Σ_{i=1}^n w_i ||y_i - x||^2 = 2(\bar{y} - x)' Σ_{i=1}^n w_i (y_i - \bar{y}) + Σ_{i=1}^n w_i ||y_i - \bar{y}||^2 + W ||\bar{y} - x||^2. (A3)
Because Σ_{i=1}^n w_i (y_i - \bar{y}) = 0, the middle term on the right-hand side vanishes, which yields equation (A2). Setting x = y_j, multiplying by w_j, and summing over j results in
Σ_{j=1}^n Σ_{i=1}^n w_i w_j ||y_i - y_j||^2 = Σ_{j=1}^n w_j Σ_{i=1}^n w_i ||y_i - \bar{y}||^2 (A4)
+ W Σ_{j=1}^n w_j ||\bar{y} - y_j||^2 = 2W Σ_{i=1}^n w_i ||y_i - \bar{y}||^2.
The left-hand side can be written as 2 Σ_{i=1}^n Σ_{j=i+1}^n w_i w_j ||y_i - y_j||^2. Then, dividing both sides by 2W, we obtain equation (A1) of Theorem 1.

**Theorem 2: Contribution to the Sum of Squares**
Contribution to the sum of squares can be expressed as follows in terms of pairwise distances:

---


## Page 33

<header>Studer et al.</header>
&lt;page_number&gt;503&lt;/page_number&gt;

$$
\|\bar{y} - x\|^2 = \frac{1}{W} \left( \sum_{i=1}^n w_i \|x - y_i\|^2 - SS \right). \quad \text{(A5)}
$$

*Proof.* Extracting $\|x - \bar{y}\|^2$ from equation (A2), we obtain
$$
\|\bar{y} - x\|^2 = \frac{1}{W} \left( \sum_{i=1}^n w_i \|y_i - x\|^2 - \sum_{i=1}^n w_i \|y_i - \bar{y}\|^2 \right). \quad \text{(A6)}
$$

The second term in the parentheses is just $SS$, which proves the theorem.

Replacing $x$ with $y_j$, we can see that $\|\bar{y} - y_j\|^2$ is the contribution of $y_j$ to $SS$ by multiplying equation (A5) by $w_j$ and summing over $j$. As a result, we obtain $\sum_j w_j \|\bar{y} - y_j\|^2 = 2SS - SS = SS$, hence the sum of squares. What makes the formula interesting is that it expresses the contribution in terms of pairwise distances. Equation (4) is just Theorem 2 with dissimilarities $d^\nu$ substituted in place of the squared Euclidean distances $\|.\|^2$.

We now prove the following result about the nonnegativity of the contribution.

**Theorem 3: Nonnegativity of the Contribution to the Sum of Squares (Sufficient Condition)**

Let $d$ be a dissimilarity measure, and assume the generalized sum of squares $SS$ is calculated with $d^\nu$. Then, the contribution of $x$ to the sum of squares $SS$,
$$
d_{xg}^\nu = \frac{1}{W} \left( \sum_{i=1}^n w_i d_{xi}^\nu - SS \right), \quad \text{(A7)}
$$
is nonnegative when $d^\nu$ respects the triangle inequality.

*Proof.* Replacing $SS$ by its expression in terms of pairwise dissimilarities (Theorem 1), the contribution (i.e., equation [A7]) can be rewritten as
$$
d_{xg}^\nu = \frac{1}{2W^2} \sum_{i=1}^n \sum_{j=1}^n w_i w_j (2d_{ij}^\nu - d_{ij}^\nu). \quad \text{(A8)}
$$

In this form, it appears that the smallest value of the contribution $d_{xg}^\nu$ is obtained when all $d_{ij}^\nu$ values take their maximal possible value. Under the triangle inequality, $d_{ij}^\nu$ cannot exceed $d_{xi}^\nu + d_{xj}^\nu$. Hence, $d_{xg}^\nu$ reaches its minimum when $d_{ij}^\nu = d_{xi}^\nu + d_{xj}^\nu$ for all $i$ and $j$. This minimum is zero, which implies $d_{xg}^\nu \geq 0$.

---


## Page 34

&lt;page_number&gt;504&lt;/page_number&gt;
Sociological Methods & Research 40(3)

**Appendix B: Should Dissimilarities Be Squared?**

In this appendix, we discuss the choice of the v exponent in equation (2). Should we square the dissimilarities when computing the generalized sum of squares, or is it preferable to substitute the squared Euclidean distances with the dissimilarities themselves?

If the chosen dissimilarity between sequences can be represented univocally as a distance in an associated Euclidean coordinate space, we would have to set v = 2 to obtain a generalized SS equal to the corresponding sum of squares in that space (Gower 1982). Although dissimilarities between strings of characters that can be expressed as kernels (Lodhi et al. 2002) have this property, most cost-minimizing distances such as OM cannot be expressed as Euclidean distances.

There are several arguments in favor of setting v = 1. According to Mielke and Berry (1983), this solution leads to a strongest congruence between analysis and data space. Moreover, it should produce more robust results when the corresponding points in the coordinate space are not normally distributed. From our point of view, the strongest argument to set v = 1 is related to the triangle inequality. Indeed, when the dissimilarity d respects the triangle inequality, $\sqrt{d}$ respects it too, whereas generally, $d^2$ does not. Because the triangle inequality of d ensures that SS cannot be greater than the sum of distances $\sum_i w_i d_{xi}^v$ to any arbitrary chosen object x, we would then be sure that SS does not exceed the sum $\sum_i w_i d_{xi}$ with v = 1, whereas it would not be the case with v = 2. The same argument can be formalized differently in terms of the contribution (equation [4]) to the sum of squares SS. The nonnegativity of this contribution automatically results when $d_{ij}^v$ satisfies the triangle inequality (see Appendix A), while negative contributions to the discrepancy can occur when the triangle inequality does not hold. Hence, v = 1 ensures nonnegative contributions when d satisfies the triangle inequality.

A negative value of the dissimilarity $d_{xg}^v$ between x and the center of gravity g means that accounting for the object x reduces the sum of squares. This can be the case when two objects, say y and z, become closer when we can pass through x (i.e., when $d_{yz} > d_{yx} + d_{xz}$). Such situations are common in social network analysis. Consider, for instance, a network between x, y, and z in which the dissimilarity is equal to 1 for two people who meet often and is equal to 10 when they never meet. The dissimilarity $d_{xg}$ would then be negative if x often meets both y and z while y never meets z. From a social network perspective, we would say that x plays a cohesive role in the network. Although a negative contribution to the discrepancy is relevant in such

---


## Page 35

Studer et al. &lt;page_number&gt;505&lt;/page_number&gt;

settings, it is most often not the case. Hence, the results should be interpreted with caution when $d_{ij}^v$ does not respect the triangle inequality, which may occur with $v = 2$ as noted above. In particular, in such situations, one should be ready to accept and give meaning to negative contributions to the discrepancy.
To summarize, we suggest defining SS with $v = 1$, except when we can express the dissimilarity measure as an Euclidean distance, in which case $v = 2$ is best suited.

## Appendix C: The Number of Permutations in Permutation Tests

It is generally thought that 1,000 permutations are sufficient to assess a result at the 5 percent level, while 5,000 are necessary at the 1 percent level. Here, we present some figures to support this claim.
Let $p$ be the true $p$ value of $F_{obs}$ and $\hat{p}$ be the proportion of $F$ values smaller than $F_{obs}$ among $R$ randomizations. Table C1 shows how the probability $P(\hat{p} < 1.2p|p)$ that the empirical $p$ value does not exceed the true $p$ value by more than 20 percent evolves with $R$ for $p = .05$ and $p = .01$. In the same table, we see that 1,000 randomizations ensure that this probability will be greater than 90 percent for $p = .05$, and $R = 5,000$ ensures this confidence for $p = .01$. The table shows also 95 percent inconclusive intervals,

**Table C1. Probability $P(\hat{p} < 1.2p|p)$ of Not Exceeding $p$ by More Than 20 Percent and 95 Percent Inconclusive Interval When the True Value Is $p$ for a Selection of $R$ Values**

<table>
  <thead>
    <tr>
      <th rowspan="2">R</th>
      <th colspan="2">$p = .05$</th>
      <th colspan="2">$p = .01$</th>
    </tr>
    <tr>
      <th>$P(\hat{p} < 1.2p|p)$</th>
      <th>Inconclusive Interval</th>
      <th>$P(\hat{p} < 1.2p|p)$</th>
      <th>Inconclusive Interval</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>100</td>
      <td>.677</td>
      <td>.007</td>
      <td>.093</td>
      <td>.580</td>
      <td>0</td>
      <td>.030</td>
    </tr>
    <tr>
      <td>200</td>
      <td>.742</td>
      <td>.020</td>
      <td>.080</td>
      <td>.612</td>
      <td>0</td>
      <td>.024</td>
    </tr>
    <tr>
      <td>500</td>
      <td>.848</td>
      <td>.031</td>
      <td>.069</td>
      <td>.673</td>
      <td>.001</td>
      <td>.019</td>
    </tr>
    <tr>
      <td>1,000</td>
      <td>.927</td>
      <td>.036</td>
      <td>.064</td>
      <td>.737</td>
      <td>.004</td>
      <td>.016</td>
    </tr>
    <tr>
      <td>1,300</td>
      <td>.951</td>
      <td>.038</td>
      <td>.062</td>
      <td>.766</td>
      <td>.005</td>
      <td>.015</td>
    </tr>
    <tr>
      <td>5,000</td>
      <td>.999</td>
      <td>.044</td>
      <td>.056</td>
      <td>.922</td>
      <td>.007</td>
      <td>.013</td>
    </tr>
    <tr>
      <td>7,000</td>
      <td>1.000</td>
      <td>.045</td>
      <td>.055</td>
      <td>.954</td>
      <td>.008</td>
      <td>.012</td>
    </tr>
    <tr>
      <td>10,000</td>
      <td>1.000</td>
      <td>.046</td>
      <td>.054</td>
      <td>.978</td>
      <td>.008</td>
      <td>.012</td>
    </tr>
    <tr>
      <td>50,000</td>
      <td>1.000</td>
      <td>.048</td>
      <td>.052</td>
      <td>1.000</td>
      <td>.009</td>
      <td>.011</td>
    </tr>
    <tr>
      <td>100,000</td>
      <td>1.000</td>
      <td>.049</td>
      <td>.051</td>
      <td>1.000</td>
      <td>.009</td>
      <td>.011</td>
    </tr>
  </tbody>
</table>

---


## Page 36

&lt;page_number&gt;506&lt;/page_number&gt;
Sociological Methods & Research 40(3)

that is, the interval that should contain 95 percent of the permutation $p$ values when the true $p$ value is $p$. This interval is equal to $p \pm 1.96 \sqrt{p(1-p)/R}$ (Manly 2007). Obtaining a permutation $p$ value in this interval does not permit one to conclude, because such a $p$ value would then not be significantly different from $p$.

**Notes**

1. Although this definition permits state dependent indel costs, unique indel costs are most often used.
2. We have also considered the use of the Brown-Forsythe $F^*$ statistic to account for unequal group discrepancy (Brown and Forsythe 1974b). However, in our experiments, results were always almost the same as with the traditional $F$. For sake of simplicity, we do not develop further this option.
3. We ran the same simulations with proportions $\theta = .1$ and $\theta = .9$. We do not report the results here, because they differ only marginally from those obtained with $\theta = .5$.
4. To be consistent with the rest of our analysis, we computed the principal coordinate from the square roots of the dissimilarities, since the principal coordinate analysis procedure automatically squares the provided dissimilarities. This amounts to set $v = 1$.
5. The Hamming distance is OM without indels.
6. Although it is not a concern here, this result can easily be extended for $q > 1$.
7. For details on TraMineR functions such as seqdef, seqdist, and dissassoc presented here, see the reference manual or type, for instance, ?dissassoc in the R console to access the online help.
8. The region factor was built from the five region binary dummies in the mvad data frame with coding not shown here.
9. The program can be downloaded at http://www.graphviz.org.
10. It should be noted, however, that such a multichannel analysis is not intended for studying the link between the channels.
11. All the numerical results and plots presented in this article can be reproduced with the R script provided as a supplemental file on the Web site of the journal.

**References**

Abbott, Andrew. 1990. “A Primer on Sequence Methods.” *Organization Science* 1:375-92.
Abbott, Andrew and John Forrest. 1986. “Optimal Matching Methods for Historical Sequences.” *Journal of Interdisciplinary History* 16:471-94.
Abbott, Andrew and Alexandra Hrycak. 1990. “Measuring Resemblance in Sequence Data: An Optimal Matching Analaysis of Musician’s Carrers.” *American Journal of Sociolgy* 96:144-85.

---


## Page 37

<header>Studer et al.</header>
&lt;page_number&gt;507&lt;/page_number&gt;

Anderson, Marti Jane. 2001. “A New Method for Non-Parametric Multivariate Analysis of Variance.” *Austral Ecology* 26:32-46.

Anderson, Marti Jane. 2006. “Distance-Based Tests for Homogeneity of Multivariate Dispersions.” *Biometrics* 62:245-53.

Bartlett, Maurice Stevenson. 1937. “Properties of Sufficiency and Statistical Tests.” *Proceedings of the Royal Society of London. Series A, Mathematical and Physical Sciences* 160:268-82.

Batagelj, Vladimir. 1988. “Generalized Ward and Related Clustering Problems.” Pp. 67-74 in *Classification and Related Methods of Data Analysis*, edited by Hans H. Bock. Amsterdam, the Netherlands: North-Holland.

Billari, Francesco Candeloro. 2001a. “The Analysis of Early Life Courses: Complex Description of the Transition to Adulthood.” *Journal of Population Research* 18:119-42.

Billari, Francesco Candeloro. 2001b. “A Log-Logistic Regression Model for a Transition Rate With a Starting Threshold.” *Population Studies* 55:15-24.

Billari, Francesco Candeloro. 2005. “Life Course Analysis: Two (Complementary) Cultures? Some Reflections With Examples From the Analysis of Transition to Adulthood.” Pp. 267-88 in *Towards an Interdisciplinary Perspective on the Life Course*, edited by René Levy, Paolo Ghisletta, Jean-Marie Le Goff, Dario Spini, and Eric Widmer. Amsterdam, the Netherlands: Elsevier.

Blossfeld, Hans-Peter and Götz Rohwer. 2002. *Techniques of Event History Modeling, New Approaches to Causal Analysis*. 2nd ed. Mahwah NJ: Lawrence Erlbaum.

Breiman, Leo, Jerome H. Friedman, R. A. Olshen, and C. J. Stone. 1984. *Classification and Regression Trees*. New York: Chapman & Hall.

Brown, Morton B. and Alan B. Forsythe. 1974a. “Robust Tests for the Equality of Variances.” *Journal of the American Statistical Association* 69:364-67.

Brown, Morton B. and Alan B. Forsythe. 1974b. “The Small Sample Behavior of Some Statistics Which Test the Equality of Several Means.” *Technometrics* 16:129-32.

Cuadras, Carles M. 2008. “Distance-Based Association and Multi-Sample Tests for General Multivariate Data.” In *Advances in Mathematical and Statistical Modeling*, edited by Barry C. Arnold, N. Balakrishnan, Jose-Maria Sarabia, and Roberto Minguez. Boston: Birkhäuser.

Delicado, Pedro. 2007. “Functional k-Sample Problem When Data Are Density Functions.” *Computational Statistics* 22:391-410.

Dijkstra, Will and Toon Taris. 1995. “Measuring the Agreement Between Sequences.” *Sociological Methods and Research* 24:214-31.

Elder, Glen H. 1999. *Children of the Great Depression*. Boulder, CO: Westview.

Elzinga, Cees H. 2003. “Sequence Similarity: A Non-Aligning Technique.” *Sociological Methods and Research* 31:214-31.

Elzinga, Cees H. 2007. “Sequence Analysis: Metric Representations of Categorical Time Series.” Unpublished manuscript, Department of Social Science Research Methods, Vrije Universiteit, Amsterdam, the Netherlands.

---


## Page 38

&lt;page_number&gt;508&lt;/page_number&gt;
Sociological Methods & Research 40(3)

Elzinga, Cees H. 2010. “Complexity of Categorical Time Series.” *Sociological Methods and Research* 38:463-81.
Elzinga, Cees H. and Aart C. Liefbroer. 2007. “De-Standardization of Family-Life Trajectories of Young Adults: A Cross-National Comparison Using Sequence Analysis.” *European Journal of Population* 23:225-50.
Gabadinho, Alexis, Gilbert Ritschard, Nicolas S. Müller, and Matthias Studer. 2011. “Analyzing and Visualizing Sate Sequences in R with TraMineR.” *Journal of Statistical Software* 40(4):1-37.
Gabadinho, Alexis, Gilbert Ritschard, Matthias Studer, and Nicolas S. Müller. 2009. “Mining Sequence Data in R With the TraMineR Package: A User’s Guide.” Technical report, Department of Econometrics and Laboratory of Demography, University of Geneva, Geneva, Switzerland.
Gabadinho, Alexis, Gilbert Ritschard, Matthias Studer, and Nicolas S. Müller. 2010. “Indice de Complexité pour le Tri et la Comparaison de Séquences Catégorielles.” *Revue des Nouvelles Technologies de l’Information* E-19:61-66.
Gabadinho, Alexis, Gilbert Ritschard, Matthias Studer, and Nicolas S. Müller. 2011. “Extracting and Rendering Representative Sequences.” Pp. 94-106 in *Knowledge Discovery, Knowledge Engineering and Knowledge Management*, edited by Ana Fred, Jan L. G. Dietz, Kecheng Liu, and Joaquim Filipe. Berlin, Germany: Springer-Verlag.
Gansner, Emden R. and Stephen C. North. 1999. “An Open Graph Visualization System and Its Applications to Software Engineering.” *Software—Practice and Experience* 30:1203-33.
Geurts, Pierre, Louis Wehenkel, and Florence d’Alché Buc. 2006. “Kernelizing the Output of Tree-Based Methods.” Pp. 345-52 in *ICML*, edited by William W. Cohen and Andrew Moore. New York: Association for Computing Machinery.
Gower, John Clifford. 1966. “Some Distance Properties of Latent Root and Vector Methods Used in Multivariate Analysis.” *Biometrika* 53:325-38.
Gower, John Clifford. 1982. “Euclidean Distance Geometry.” *Mathematical Scientist* 7:1-14.
Gower, John Clifford and Wojtek J. Krzanowski. 1999. “Analysis of Distance for Structured Multivariate Data and Extensions to Multivariate Analysis of Variance.” *Journal of the Royal Statistical Society: Series C (Applied Statistics)* 48:505-19.
Jobson, J. D. 1991. *Applied Multivariate Data Analysis, Volume I: Regression and Experimental Design*. New York: Springer-Verlag.
Lesnard, Laurent. 2010. “Setting Cost in Optimal Matching to Uncover Contemporaneous Socio-Temporal Patterns.” *Sociological Methods and Research* 38:389-419.
Lodhi, Huma, Craig Saunders, John Shawe-Taylor, Nello Cristianini, and Chris Watkins. 2002. “Text Classification Using String Kernels.” *Journal of Machine Learning Research* 2:419-44.
Manly, Bryan F. J. 2007. *Randomization, Bootstrap and Monte Carlo Methods in Biology*. 3rd ed. New York: Chapman & Hall.

---


## Page 39

<header>Studer et al.</header>
&lt;page_number&gt;509&lt;/page_number&gt;

McArdle, Brian H. and Marti J. Anderson. 2001. “Fitting Multivariate Models to Community Data: A Comment on Distance-Based Redundancy Analysis.” *Ecology* 82:290-97.

McVicar, Duncan and Michael Anyadike-Danes. 2002. “Predicting Successful and Unsuccessful Transitions From School to Work Using Sequence Methods.” *Journal of the Royal Statistical Society A* 165:317-34.

Mielke, Paul W. and Kenneth J. Berry. 1983. “Asymptotic Clarifications, Generalizations, and Concerns Regarding an Extended Class of Matched Pairs Tests Based on Powers of Ranks.” *Psychometrika* 48:483-85.

Mielke, Paul W. and Kenneth J. Berry. 2007. *Permutation Methods: A Distance Function Approach*. 2nd ed. New York: Springer.

Morgan, J. N. and J. A. Sonquist. 1963. “Problems in the Analysis of Survey Data, and a Proposal.” *Journal of the American Statistical Association* 58:415-34.

Piccarreta, Raffaella. 2010. “Binary Trees for Dissimilarity Data.” *Computational Statistics and Data Analysis* 54:1516-24.

Piccarreta, Raffaella and Francesco Candeloro Billari. 2007. “Clustering Work and Family Trajectories by Using a Divisive Algorithm.” *Journal of the Royal Statistical Society A* 170:1061-1078.

Pollock, Gary. 2007. “Holistic Trajectories: A Study of Combined Employment, Housing and Family Careers by Using Multiple-Sequence Analysis.” *Journal of the Royal Statistical Society A* 170:167-83.

R Development Core Team. 2008. “R: A Language and Environment for Statistical Computing.” Vienna, Austria: R Foundation for Statistical Computing.

Reiss, Philip T. M., Henry H. Stevens, Zarrar Shehzad, Eva Petkova, and Michael P. Milham. 2009. “On Distance-Based Permutation Tests for Between-Group Comparisons.” *Biometrics* 66:636-43.

Scherer, Stefani. 2001. “Early Career Patterns: A Comparison of Great Britain and West Germany.” *European Sociological Review* 17:119-44.

Shaw, Ruth G. and Thomas Mitchell-Olds. 1993. “ANOVA for Unbalanced Data: An Overview.” *Ecology* 74:1638-45.

Späth, Helmuth. 1975. *Cluster Analyse Algorithmen*. Munich, Germany: R. Oldenbourg Verlag.

Studer, Matthias, Gilbert Ritschard, Alexis Gabadinho, and Nicolas S. Müller. 2009. “Analyse de Dissimilarités par Arbre d’Induction.” *Revue des Nouvelles Technologies de l’Information* E-15:7-18.

Studer, Matthias, Gilbert Ritschard, Alexis Gabadinho, and Nicolas S. Müller. 2010. “Discrepancy Analysis of Complex Objects Using Dissimilarities.” Pp. 3-19 in *Advances in Knowledge Discovery and Management*, edited by Fabrice Guillet, Gilbert Ritschard, Djamel A. Zighed, and Henri Briand. Berlin, Germany: Springer.

Widmer, Eric and Gilbert Ritschard. 2009. “The De-Standardization of the Life Course: Are Men and Women Equal?” *Advances in Life Course Research* 14:28-39.

---


## Page 40

&lt;page_number&gt;510&lt;/page_number&gt;
Sociological Methods & Research 40(3)

Wu, Lawrence L. 2000. “Some Comments on 'Sequence Analysis and Optimal Matching Methods in Sociology: Review and Prospect.’” *Sociological Methods Research* 29:41-64.

Yujian, Li and Liu Bo. 2007. “A Normalized Levenshtein Distance Metric.” *IEEE Transactions on Pattern Analysis and Machine Intelligence* 29:1091-95.

Zapala, Matthew A. and Nicholas J. Schork. 2006. “Multivariate Regression Analysis of Distance Matrices for Testing Associations Between Gene Expression Patterns and Related Variables.” *Proceedings of the National Academy of Sciences of the United States of America* 103:19430-35.

## Bios

**Matthias Studer** is a PhD student in socioeconomics and a research assistant at the Institute for Demographic and Life Course Studies at the University of Geneva. He holds a master’s degree in economics and a master of advanced studies in sociology. He is currently working on gender and social inequalities at the beginning of academic careers in Switzerland. His research interests include data mining of longitudinal data such as state and event sequences, dissimilarity analysis, and survival trees.

**Gilbert Ritschard** is a full professor of statistics for the social sciences at the University of Geneva. His current research interests are in life-course analysis and the application of exploratory data-mining methods in social sciences. Besides many contributions to edited volumes, he has published recently in *Advanced Life Course Research*, *Studies in Family Planning*, and the *International Journal of Data Mining, Modelling and Management*, and he has coedited the book *Advances in Knowledge Discovery and Management* (Springer-Verlag, 2010). He leads currently a Swiss National Science Foundation research project on methods for mining event histories.

**Alexis Gabadinho** is a scientific collaborator at the Institute for Demographic and Life Course Studies at the University of Geneva. He holds a postgraduate diploma in demography. His current research interests are the application of data-mining methods in social sciences and the development of methods for categorical state sequences analysis, in particular measures of sequence complexity and methods for summarizing sets of sequences.

**Nicolas S. Müller** is currently a PhD student in sociology and a research assistant at the Institute for Demographic and Life Course Studies at the University of Geneva. He holds a master of arts in sociology and a master of science in information systems. His PhD subject concerns the links between life trajectories, socioeconomic factors, and health outcomes. He is interested in the application of data-mining methods in social sciences, especially sequence mining and association rules methods.