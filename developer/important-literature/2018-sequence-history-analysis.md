## Page 1

# Sequence History Analysis (SHA): Estimating the Effect of Past Trajectories on an Upcoming Event

Florence Rossignon, Matthias Studer, Jacques-Antoine Gauthier, and Jean-Marie Le Goff

&lt;img&gt;Check for updates&lt;/img&gt;

## 1 Introduction

In many research questions framed within the life-course paradigm, the estimation of the effect of a previous trajectory on an upcoming event is of central interest. While this paradigm recognizes that structural constraints influence choices and outcomes, it also acknowledges the significant impact of past individual trajectories. Many previous studies addressed such kinds of issues. For instance, Madero-Cabib et al. (2015) modeled the influence of past occupational trajectories on the timing of retirement. Studer (2012) studied the effect of past working and financing conditions on the chances of obtaining a PhD among teaching assistants at the University of Geneva. Lundevaller et al. (2018) investigated how different work and family trajectories during early adulthood affected mortality risks during late adulthood in Sweden in the nineteenth century. Finally, Eerola and Helske (2016) studied how trajectories of partnership formation and parenthood predict depression scores (see also the first case study in Eerola 2018, in this bundle). In all these examples, the past processes under study consist of categorical states unfolding over time, such as family or occupational statuses. These kinds of research questions might also

F. Rossignon ()
Swiss Federal Statistical Office, Neuchâtel, Switzerland
e-mail: florence.rossignon@bfs.admin.ch

M. Studer
NCCR LIVES and Geneva School of Social Sciences, University of Geneva, Geneva, Switzerland

J.-A. Gauthier · J.-M. Le Goff
NCCR LIVES and University of Lausanne, Lausanne, Switzerland
e-mail: jacques-antoine.gauthier@unil.ch; jean-marie.legoff@unil.ch

© The Author(s) 2018
G. Ritschard, M. Studer (eds.), *Sequence Analysis and Related Approaches*, Life Course Research and Social Policies 10,
https://doi.org/10.1007/978-3-319-95420-2_6
&lt;page_number&gt;83&lt;/page_number&gt;

---


## Page 2

&lt;page_number&gt;84&lt;/page_number&gt;
F. Rossignon et al.

emerge when studying linked-life domains, another core principle of the life-course paradigm. For instance, one may be interested in estimating the effect of past family trajectories on the chances of obtaining a promotion among female managers.

From a methodological point of view, some event history models have tackled this issue by including a few summary indicators of the past trajectory (e.g., time spent in a given state, or a dummy variable indicating whether a specific event has already occurred or not) (Blossfeld et al. 2007). This approach allows estimation of the effect of these past trajectory indicators on the chances of experiencing the event under study. However, this process is often limited as it might fail to identify the key dimensions of the previous trajectory affecting the event. First, these key dimensions might depend on the trajectories themselves. In that case, it becomes spurious to decide a priori which relevant past trajectory indicators should be included in the model. Second, trajectories represent complex objects with many different dimensions. It might therefore be difficult to identify the most relevant ones. For instance, life-course scholars stress the importance of three sub-dimensions, each requiring several indicators to be included in the analysis. These indicators refer to the timing, the ordering, and the duration of states and of transitions (Scott and Alwin 1998). Finally, there might be many interaction effects between these sub-dimensions, making the selection of relevant indicators of past trajectories even more difficult.

In this article, we develop an innovative method combining Sequence Analysis and Event History Analysis which we call Sequence History Analysis (SHA). Its aim is to tackle the aforementioned methodological challenges and the method works in two steps. We start by identifying typical past trajectories of individuals over time by using Sequence Analysis. As trajectories are considered as a progression over time, where events and life stages accumulate, individuals are likely to move from one cluster to another over time. We then estimate the effect of these typical past trajectories on the event under study using discrete-time models. SHA is presented in the first part of this paper.

In the second part of this article, we use the proposed methodological approach in an original study of the effect of past childhood co-residence structures on the chances of leaving the parental home in Switzerland. In western countries where nuclear families and neolocality prevail, the departure from the parental home is a crucial step and an indicator of the transition to adulthood. It is often a prerequisite to achieving other family life transitions, such as co-residency and becoming a parent (Mulder 2009; Schizzerotto and Lucchini 2004). Furthermore, the departure from the parental home has significant consequences for important policy areas, such as the demand for housing (Ermisch and Di Salvo 1997) and the risk of poverty among young people (Iacovou and Aassve 2007). In this context, identifying the determinants of the early departure from the parental home of young adults is of prime interest. Among these determinants, many sociological theories stress the importance of family configuration, as well as the whole individual trajectory preceding home-leaving. This is a key concern in Switzerland where the

---


## Page 3

<header>Sequence History Analysis (SHA)</header>
&lt;page_number&gt;85&lt;/page_number&gt;

number of divorces has experienced a strong increase over the past 40 years (Swiss Federal Statistical Office 2016), with a divorce rate that reached 52.6% in 2005. A significant number of studies showed the impact of lone- and step-parenthood on early departure (Holdsworth 2000; Bernhardt et al. 2005). Consequently, some studies focused on the co-residence structure in which a young adult lived at a specific moment of his/her life, often at the time of the youth's final home-leaving (Mitchell et al. 1989; Chiuri and Del Boca 2010). However, few studies looked back at the effect of the whole co-residence trajectory. This is mostly due to the lack of detailed life history records of co-residence structures during childhood (Aquilino 1991; Goldscheider and Goldscheider 1998; Blaauboer and Mulder 2010) and to the lack of a proper methodological framework to estimate the influence of early co-residence trajectories on the departure from the parental home.

The structure of the article is as follows. First, we will present the methodological features of SHA. We will then apply it empirically using social science data. Based on data from the LIVES Cohort Study (Elcheroth and Antal 2013), our analyses showed that it is not only the occurrence of an event such as parental divorce that increases the risk of leaving home, but also the order in which changes to the preceding family co-residence structure occurred. Two features have a significant influence on leaving home: the co-residence structure itself and the arrival or departure of siblings from the parental home.

## 1.1 Sequence History Analysis: A Combination of Sequence Analysis and Event History Analysis

Several methods are available to estimate the effects of a set of covariates on the hazard rate of a given event. This approach uses a discrete-time representation of the data: the so-called person-period file (Allison 2014). In this format, one observation is generated for each individual $i$ at each time point $t$. Since the time $t$ is assumed to be observed on a discrete scale, a finite set of observations is generated. The time ranges from the start of observation (typically 0) until the end of the observation period of the $i$th individual.

Before going into details, let us present a small example that will help us to clarify the presentation of a person-period file. In this example, (Table 1), we are interested in estimating the effect of past cohabitation trajectories on the chances of leaving the parental home among a cohort of young adults. The individual 72 is a woman. She left home after 6 time periods. She also has the following cohabitation trajectory: BP-BS-BS-BS-LS-LS where BP stands for biparental household, BS for biparental household and siblings, and LS for lone-parent household and siblings. The corresponding person-period file therefore reads as follows:

<table>
  <thead>
    <tr>
      <th>Individual</th>
      <th>Time</th>
      <th>Household Type</th>
      <th>Leaving Home</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>72</td>
      <td>0</td>
      <td>BP</td>
      <td>0</td>
    </tr>
    <tr>
      <td>72</td>
      <td>1</td>
      <td>BS</td>
      <td>0</td>
    </tr>
    <tr>
      <td>72</td>
      <td>2</td>
      <td>BS</td>
      <td>0</td>
    </tr>
    <tr>
      <td>72</td>
      <td>3</td>
      <td>BS</td>
      <td>0</td>
    </tr>
    <tr>
      <td>72</td>
      <td>4</td>
      <td>LS</td>
      <td>0</td>
    </tr>
    <tr>
      <td>72</td>
      <td>5</td>
      <td>LS</td>
      <td>0</td>
    </tr>
    <tr>
      <td>72</td>
      <td>6</td>
      <td>LS</td>
      <td>1</td>
    </tr>
  </tbody>
</table>

---


## Page 4

&lt;page_number&gt;86&lt;/page_number&gt;
F. Rossignon et al.

<table>
  <caption>Table 1 Example of a person-period file</caption>
  <thead>
    <tr>
      <th>ID</th>
      <th>Time</th>
      <th>Departure from the parental home</th>
      <th>Cohabitation status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>72 (Woman)</td>
      <td>15</td>
      <td>0</td>
      <td>BP</td>
    </tr>
    <tr>
      <td>72 (Woman)</td>
      <td>16</td>
      <td>0</td>
      <td>BS</td>
    </tr>
    <tr>
      <td>72 (Woman)</td>
      <td>17</td>
      <td>0</td>
      <td>BS</td>
    </tr>
    <tr>
      <td>72 (Woman)</td>
      <td>18</td>
      <td>0</td>
      <td>BS</td>
    </tr>
    <tr>
      <td>72 (Woman)</td>
      <td>19</td>
      <td>0</td>
      <td>LS</td>
    </tr>
    <tr>
      <td>72 (Woman)</td>
      <td>20</td>
      <td>1</td>
      <td>LS</td>
    </tr>
  </tbody>
</table>

## 1.2 Sequence History Analysis: Operationalizing Previous Trajectories

There are two different interpretations of the aim of Sequence Analysis. It may be seen as a way to identify ideal-typical trajectories. It can also be considered as an effective means to reduce the complexity of trajectories into a few main types of sequences. Both approaches are interesting in our context, because one might typically expect to observe many different individual past trajectories. Sequence Analysis can therefore be used to operationalize the concept of past trajectories by reducing this complexity or as a way to identify ideal-typical past trajectories.

Since the 1990s-2000s, the research trend in Sequence Analysis has been structured around a core program including a limited number of methodological options (Gauthier et al. 2014). Generally, Sequence Analysis works in three steps. First, trajectories are coded as sequences of states. Second, the distances between each pair of sequences are computed and gathered into a distance matrix. Finally, a cluster analysis is conducted on this matrix. It gathers together similar sequences while separating dissimilar sequences. The result is a categorical covariate that can be used in subsequent analyses. Let us briefly discuss these three steps in our case.

In the first step, we rebuild the past trajectory at each time point, i.e., for each observation of each individual i at time t. Taking our previous example a step further, Table 2 presents two ways of modeling past family trajectories. First, rebuilding the past trajectory at each time point for each individual in our person-period file is done by considering, at each time point t, the trajectory leading to the current position. As such, the length of the trajectory logically increases by one for each additional time unit. Sometimes, we are only interested in the previous trajectory, excluding the present state. Thus, the last column reconstructs for each individual i at each time point t, past trajectory until t − 1. These past trajectories can therefore be interpreted as past trajectories until all possible present times. There are, thereby, t trajectories of varying lengths for each individual. Indeed, since the duration from the starting time is not the same at each time point, the past trajectories considered grow over time.

In a second step, we need to choose a distance measure to conduct Sequences Analysis. This measure defines which criteria should be taken into account to

---


## Page 5

Sequence History Analysis (SHA)
&lt;page_number&gt;87&lt;/page_number&gt;

Table 2 Two different ways of reconstructing past trajectories
<table>
  <thead>
    <tr>
      <th>ID</th>
      <th>Time</th>
      <th>Departure from the parental home</th>
      <th>Past trajectory</th>
      <th>Past trajectory excluding present</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>72</td>
      <td>15</td>
      <td>0</td>
      <td>BP/15</td>
      <td>BP/14</td>
    </tr>
    <tr>
      <td>72</td>
      <td>16</td>
      <td>0</td>
      <td>BP/15-BS/1</td>
      <td>BP/15</td>
    </tr>
    <tr>
      <td>72</td>
      <td>17</td>
      <td>0</td>
      <td>BP/15-BS/2</td>
      <td>BP/15-BS/1</td>
    </tr>
    <tr>
      <td>72</td>
      <td>18</td>
      <td>0</td>
      <td>BP/15-BS/3</td>
      <td>BP/15-BS/2</td>
    </tr>
    <tr>
      <td>72</td>
      <td>19</td>
      <td>0</td>
      <td>BP/15-BS/3-LS/1</td>
      <td>BP/15-BS/3</td>
    </tr>
    <tr>
      <td>72</td>
      <td>20</td>
      <td>1</td>
      <td>BP/15-BS/3-LS/2</td>
      <td>BP/15-BS/3-LS/1</td>
    </tr>
  </tbody>
</table>

Table 3 Creation of a typology of past trajectories
<table>
  <thead>
    <tr>
      <th>ID</th>
      <th>Time</th>
      <th>Departure from the parental home</th>
      <th>Past trajectory excluding present</th>
      <th>Typology</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>72</td>
      <td>15</td>
      <td>0</td>
      <td>BP/14</td>
      <td>Biparental household</td>
    </tr>
    <tr>
      <td>72</td>
      <td>16</td>
      <td>0</td>
      <td>BP/15</td>
      <td>Biparental household</td>
    </tr>
    <tr>
      <td>72</td>
      <td>17</td>
      <td>0</td>
      <td>BP/15-BS/1</td>
      <td>Early arrival of siblings</td>
    </tr>
    <tr>
      <td>72</td>
      <td>18</td>
      <td>0</td>
      <td>BP/15-BS/2</td>
      <td>Early arrival of siblings</td>
    </tr>
    <tr>
      <td>72</td>
      <td>19</td>
      <td>0</td>
      <td>BP/15-BS/3</td>
      <td>Early arrival of siblings</td>
    </tr>
    <tr>
      <td>72</td>
      <td>20</td>
      <td>1</td>
      <td>BP/15-BS/3-LS/1</td>
      <td>From biparental to lone-parent household (with siblings)</td>
    </tr>
  </tbody>
</table>

compare two trajectories. According to Studer and Ritschard (2016), the choice of a dissimilarity measure should be based on its sensitivity to timing, sequencing, or duration. For instance, if age is thought to be an important property of the past trajectory, one should emphasize timing. This would be interesting if age at parental divorce is believed to be of key importance. Conversely, if we want to focus on the path, i.e., the states through which an individual goes, a distance measure sensitive to sequencing should be chosen. Finally, if the time spent in each state is important, a distance measure sensitive to duration should be used.

In a third step, after having computed the distances between sequences, a typology is built using cluster analysis. This step results in a categorical covariate in our person-period file. Taking back our previous example and assuming that the cluster analysis identified three groups: (1) “Biparental household”, (2) “Arrival of siblings”, and (3) “From biparental to lone-parent household (with siblings),” our person-period file would be read as shown in Table 3. It can be noted that a given individual can belong to different clusters over time. In other words, the type of past trajectory an individual belongs to may change over time. It stems from the fact that our unit of analysis is one person-period, not one individual. In the next step, we will use this information to estimate the effect of a past trajectory on the event under study.

---


## Page 6

&lt;page_number&gt;88&lt;/page_number&gt;
F. Rossignon et al.

In these last two steps, we analyze sequences of different lengths.¹ Sequence Analysis should not be used to analyze sequences of different length when this difference results from incomplete or censored data, because Sequence Analysis implicitly assumes that the processes under study are fully observed.² However, in our case, the differences in sequence length do not result from incomplete data. They result from meaningful differences in the length of the process leading to the current situation. However, as the length of the previous trajectory and age are often closely related, we strongly recommend to always control for age.

### 1.3 Event History Analysis: Estimating the Effect of Typical Past Trajectories on the Event Under Study

Event history analysis is an suitable tool to understand how events are produced and how they are conditioned by other explanatory variables, which may or may not vary over time (Allison 2014). As such, once the typology of previous trajectories is created, we propose to estimate its effect on a given event using a discrete-time event history model. More precisely, at each time of observation since the starting time, the trajectory is introduced as a time-varying covariate. Consequently, applying Sequence History Analysis to our previous example, we could estimate the chance of obtaining a promotion according to the type of previous family trajectories.

Two factors should be taken into consideration when specifying the model. First, our typology of past trajectories might be linked to their length. If this is the case, we recommend adding the length of the previous trajectory or a transformation of it to the model. Consequently, the effect of the typology will no longer be related to the length of the trajectories, which could lead to misleading interpretations.³

Second, two interpretations of the past trajectories' effect can be made. It could be related to the ability of the typology to summarize the main information of the current situation. For instance, the effect of the past family trajectory on the chances of receiving a promotion could be related to the characteristics of the current family situation, such as being married or having a child at time $t$. It could also be linked to the individual history, such as the age when the individual got married or if he/she was married before having a child or not. We can distinguish these two situations by adding simple indicators of the current situation to the model. If the effect of the past trajectory types remains significant, one may conclude that “individual history

¹The length of the previous trajectories typically depends on age.
²By clustering incomplete sequences, we often end up with one (or several) clusters of incomplete trajectories, which cannot be interpreted. If this is not the case, we implicitly predict the end of the sequence, which might also be problematic.
³From a general point of view, even though this is not specific to the proposed methodology, it is generally recommended to add some timing information to the model as the hazard rate is usually not constant over time.

---


## Page 7

Sequence History Analysis (SHA)
&lt;page_number&gt;89&lt;/page_number&gt;

matters” for the issue under study. Aside from these two kinds of information, the usual control variables should be added to the model. The latter are research-specific and we therefore do not discuss them in more detail.

To sum up, we propose a methodological approach to estimate the effect of a previous trajectory on an upcoming event. This methodology functions in three steps: (1) building previous trajectories in a person-period file, (2) running Sequence Analysis, and (3) estimating the effect of typical past trajectories on an upcoming event using a discrete-time model. After having presented this methodology, we now turn to its application. Therefore, we provide an empirical example displaying the effect of childhood co-residence trajectories on the risk of leaving home. The analyses will be based on life history calendar data.

## 2 Empirical Application: Childhood Co-residence Trajectories and Leaving Home

In this section, we apply Sequence History Analysis to assess the influence of childhood co-residence trajectories on the probability of leaving the parental home in Switzerland. Previous research has demonstrated the multiple effects of previous co-residence trajectories on the departure from the parental home (Mitchell et al. 1989; Aquilino 1991; Sandefur et al. 2008; Blaauboer and Mulder 2010). There are also some reasons to believe that the number of siblings living in the same household is likely to affect the probability of young adults leaving the parental home (Mitchell et al. 1989; Aquilino 1991; Gierveld et al. 1991; Avery et al. 1992; Buck and Scott 1993).

First and foremost, growing up with two biological parents—which is still the most common form of living arrangements in Western Europe—is linked with closer family bonds and longer stays in the parental home (Mitchell et al. 1989; Aquilino 1991; Mitchell 1994; Goldscheider and Goldscheider 1998).

Second, several studies showed that children of divorced parents tend to leave the parental home earlier than those of intact families (Goldscheider and Goldscheider 1998; Cherlin et al. 1995; Juang et al. 1999; Holdsworth 2000; Bernhardt et al. 2005; Zorlu and Gaalen 2016). As noted by several authors, this effect might be more related to low family socio-economic background than to the absence of one of the parental figures (Bianchi 1987; Mitchell et al. 1989; McLanahan and Carlson 2004; Kiernan 2006, for instance). Aquilino (1991) showed that young adults who grew up in a single-parent household from birth do not have a higher hazard of leaving home than those who grew up in an intact family. Therefore, the stability of co-residence structure could also have an impact on the timing of leaving home.

Third, children from step-parent families tend to leave home earlier than young adults from intact families (Mitchell et al. 1989; Aquilino 1991; Kiernan 1992; Goldscheider and Goldscheider 1998). Among various explanations, Goldscheider and Goldscheider (1998) stress the difficulty of welcoming a new parental figure,

---


## Page 8

&lt;page_number&gt;90&lt;/page_number&gt;
F. Rossignon et al.

step-siblings, and/or half-siblings into one’s home. Other studies have shown that severe conflicts and disagreements within step-families play a significant role in early nest-leaving (Gaehler and Bernhardt 2000; Gossens 2001).

Fourth, there might be some circumstances in which both intact and non-intact families may no longer be able to maintain their households. In such situations, both children and parents might seek shelter in someone else’s household, in most cases in the houses of grandparents (Aquilino 1991). This type of family arrangement is often referred to as “extended family.” Therefore, as having to move back in with relatives is usually the result of financial difficulties, such situations might push children to get a job and establish an independent household earlier.

There is some evidence that having siblings might also influence the departure from the parental home. Individuals with many siblings were found to have a higher likelihood of leaving home (Mitchell et al. 1989; Aquilino 1991; Gierveld et al. 1991; Avery et al. 1992; Buck and Scott 1993). This may be explained by the fact that individuals who grow up with a large number of siblings have a higher risk of feeling “overcrowded” in their parental home and of suffering from a lack of physical space for privacy. First-born children have a higher risk of leaving home at an earlier age than any other children, except if they are an only child (Bianchi 1987). Indeed, Holdsworth (2000) has shown that an only child will tend to stay longer at home in order to take care of their parents.

## 3 Data

We use data from the LIVES Cohort Study (FORS and NCCR LIVES 2015), a panel survey whose first wave was conducted from mid-October 2013 to the end of June 2014 (Elcheroth and Antal 2013). The sample includes 1691 respondents, of whom 415 were Swiss and 1276 were from a foreign background. The sample is composed of people aged 15–24 on January 1st 2013 and who began in a Swiss school before the age of 10. Second-generation immigrants are over-represented in the sample and particular attention is paid to offspring of low- or middle-skilled migrants who mainly hail from Southern Europe or from the Balkan Peninsula. The sampling design of the survey is quite innovative in the sense that it combines a stratified random sample with two iterations of controlled network sampling, with random selection within the personal networks (Brändle 2017, for more details).

The life history calendar allows for the collection of detailed information regarding the co-residence trajectories of each respondent. This information was recoded into five statuses—combining information on parents and siblings—namely living with: (a) both parents (without siblings), (b) both parents and sibling(s), (c) one parent (without siblings), (d) one parent and sibling(s), and (e) other relatives. Unfortunately, living with a step-parent could not be distinguished from the “one parent” situation, since this information was not available in the survey.

The timing of the departure from the parental home was operationalized as the first episode in which an individual does not live with his/her parents anymore.

---


## Page 9

Sequence History Analysis (SHA)
&lt;page_number&gt;91&lt;/page_number&gt;

We assumed that individuals are at risk of leaving home from the age of 15. Consequently, we identified 147 events (i.e., departures from the parental home) for 1637 individuals.<sup>4</sup>

### 3.1 Control Variables

Several control variables were introduced into the model, such as age (logarithm), sex, ethnic origin (based on the mother’s country of birth), place of residence at 14 years old, labor market integration (apprenticeship included)<sup>5</sup> and family socio-economic background.<sup>6</sup> This last is a key control variable since it has been argued that the effect of living with a single parent is linked to differences in economic conditions. However, the large number of missing values for family socio-economic background (70%) forced us to run a model without it. Finally, two additional control variables were included: the occurrence of parental disruption and the presence of siblings. These variables were included to verify that the effect of the previous trajectory, as measured through our method, is not only related to the current situation of the household.

### 4 Analysis

Sequence History Analysis works in three steps. We start by recoding all past trajectories in a person-period file. We then conduct a Sequence Analysis on these recoded past trajectories. Finally, using a discrete-time model, we estimate the effect of these past trajectories on the probability of leaving home. Let us present these steps in more detail using our empirical example.

---
<sup>4</sup>This means that 55 individuals had missing data in at least one of the variables.
<sup>5</sup>The Swiss education system is largely an apprenticeship-based system of education (Thomsin et al. 2004). Almost two thirds of every cohort of students attend a vocational and training program (VET) (Swiss Federal Statistical Office 2015) In principle, an apprenticeship contract is signed between the apprentices and an approved firm in which the former will spend about two thirds of their time following a practical vocational training. The rest of the time is spent in a vocational school.
<sup>6</sup>In most cases, the occupation of the father when the respondent was 15 was taken as the benchmark to define the family’s socio-economic background. When this information was unavailable, the occupation of the mother when the respondent was 15 was used.

---


## Page 10

&lt;page_number&gt;92&lt;/page_number&gt;
F. Rossignon et al.

## 4.1 Sequence Analysis: Operationalizing Previous Co-residence Trajectories

Here, we consider previous co-residence trajectories for each individual $i$ at each time $t$, from birth until the current age of each individual at the time of the survey. Since we are interested in the hazard of leaving home after the age of 15, the past trajectories have a starting length of 14 for all individuals. The final length of the past trajectory corresponds to the occurrence of the departure from the parental home minus one time unit (cf. Table 2, past trajectories excluding present state) or to the end of the observation period.

We then ran Sequence Analysis on these past trajectories excluding the present state to identify ideal-types of past trajectories. All past trajectories were included in the analyses at the same time. The use of Sequence Analysis involves choosing an appropriate distance measure and a clustering algorithm. We are interested in estimating the effect of a previous family history on the departure from the parental home. This previous history is strongly linked to the order of the stages through which an individual goes. We have therefore chosen a distance sensitive to differences in sequencing. When sequencing is of central interest, Studer and Ritschard (2016) suggest using optimal matching on sequences of distinct successive states (DSS). In our case, we used a constant substitution cost of 2 and an indel cost of 1. The DSS is obtained by considering the succession of states without considering the duration of each state. For instance, the sequence “S-S-M-M-M” is recorded “S-M.” We therefore focus only on sequencing (the information about timing and duration is not used).

We then clustered the sequences in the following way. Two specific groups were created “manually” to match a precise definition: two whole trajectories spent with both parents either with or without siblings. Each resulting group represents respectively 40.5% and 4.1% of the sample. Although small, this latter group (i.e., only child with both parents) is relevant as being an only child is expected to have a significant influence on the risk of leaving the parental home. We then used the PAM (“Partitioning Around Medoids”) algorithm to cluster the remaining past trajectories. This algorithm aims to obtain the best partitioning for a data set into a predefined number $k$ of groups (Kaufman and Rousseeuw 2005; Studer 2013). Based on the best average silhouette width, we kept six groups. The result is a final typology of eight groups (the two manually-constructed groups and the six clusters on the remaining sequences).

All statistical analyses conducted in this article use the R Software and environment (R Core Team 2016), along with the TraMineR package (Gabadinho et al. 2011) for sequence analysis and the WeightedCluster package (Studer 2013) for cluster analysis. The final typology of co-residence trajectories is presented in Fig. 1. When we observe these ideal-types of past trajectories, we see that the percentages presented are based on the person-period trajectories and that individuals can switch between clusters over time. For instance, we expect that some individuals will start by being in the cluster “Both parents and siblings” before going to the cluster “Late departure of siblings.”

---


## Page 11

Sequence History Analysis (SHA)
&lt;page_number&gt;93&lt;/page_number&gt;

&lt;img&gt;
Both parents & siblings (40.5%)
Both parents (4.1%)
Both parents to one parent, with siblings (10.4%)
Both parent to one parent, without siblings (4.5%)
Early arrival of siblings (28.7%)
Early arrival of siblings & parental separation (6.2%)
Late departure of siblings (3.1%)
One parent to both parent, with siblings (2.5%)
&lt;/img&gt;

Fig. 1 Trajectories of past co-residence structures

• Both parents and siblings (40.5%)—Trajectories spent entirely with both parents and siblings. As this cluster represents the most common trajectory, it was used as the reference category in the regression models.
• Early arrival of siblings (28.7%)—Trajectories of oldest children who experienced the arrival of younger siblings during their early childhood.
• Both parents to one parent (with siblings) (10.4%)—Trajectories characterized by a transition from a biparental to a lone-parent household, in both cases in the presence of siblings.
• Early arrival of siblings and parental separation (6.2%)—Trajectories of older siblings who experienced the arrival of younger siblings during their adolescence and a subsequent parental disruption.
• Both parents to one parent (without siblings) (4.5%)—Trajectories characterized by a parental disruption without siblings.

---


## Page 12

&lt;page_number&gt;94&lt;/page_number&gt;
F. Rossignon et al.

*   Both parents (4.1%)—Trajectories spent entirely with both parents, but without siblings.
*   Late departure of siblings (3.1%)—Trajectories characterized by the departure of siblings. These trajectories are probably those of younger children.
*   One parent to both parents (with siblings) (2.5%)—Trajectories initiated by a co-residency with one parent only and siblings before the second parent joins the household later. This might be a typically common trajectory of a migrant family. Fathers first migrate alone, leaving their wives and children behind. After a few years, mothers and children will also migrate, reuniting the family. Consequently, children will live their first years in a “lone-parent household” before moving to a biparental household.

## 4.2 Event History Analysis: Estimating the Effect of Typical Past Trajectories on the Event Under Study

After having identified the previous typology of trajectories, we estimate their effect on the risk of leaving home using a discrete-time model (cf. Table 4). We do it by running a logistic regression on the person-period file. Individuals with missing data were not included in the models (4% in models 1,2, & 3 and 70% in model 4).

Four models were estimated. The first model includes the past trajectories and the control variables. In the second model, aggregated indicators of parental divorce and presence of sibling(s) were included to assess whether the effect of past trajectories remains significant in the presence of these aggregated indicators. The third model is computed without the past trajectories to estimate its statistical power. Finally, the last model is composed of the past trajectories, the control variables, and the family socio-economic background factor.

A first sign of the overall importance of the past trajectory covariate can be asserted by looking at the Bayesian Information Criterion (BIC).<sup>7</sup> We decided to use this criterion because it is the most conservative and penalizes complexity more than the Akaike Information Criterion (AIC). According to the BIC, the first model is the most parsimonious. The BIC value of the second model is also very close to that of the first model. Both models include the ideal-typical past trajectories. From a statistical point of view, there is therefore an added value to including the past ideal-typical trajectories in the model.

<sup>7</sup>BIC = −2 ln(L) + ln(N) * k, where L is the likelihood, −2 ln(L) is equal to the deviance, ln is the logarithm, and k represents the number of parameters (i.e., coefficients). Raftery (1995) argues that the N can be estimated in three different manners when it is used for event history models: the number of observations (person-period), the number of individuals, or the number of events. According to the recommendations made in this article, we used the last option which is the least conservative. This option is also coherent with the calculation of the BIC for survival continuous-time models (i.e., Cox models) in which N represents the number of observed events.

---


## Page 13

Sequence History Analysis (SHA)
&lt;page_number&gt;95&lt;/page_number&gt;

<table>
  <caption>Table 4 Logit models predicting probability of first home-leaving</caption>
  <thead>
    <tr>
      <th></th>
      <th>Model 1</th>
      <th>Model 2</th>
      <th>Model 3</th>
      <th>Model 4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Intercept</td>
      <td>-9.79 (0.67) ***</td>
      <td>-10.79 (0.77) ***</td>
      <td>-9.20 (0.66) ***</td>
      <td>-10.76 (1.10) ***</td>
    </tr>
    <tr>
      <td>Co-residence configurations:</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Both parents & siblings (ref.)</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Both parents</td>
      <td>0.09 (0.56)</td>
      <td>0.96 (0.65)</td>
      <td></td>
      <td>-0.05 (0.83)</td>
    </tr>
    <tr>
      <td>Late departure of siblings</td>
      <td>1.05 (0.35) **</td>
      <td>1.91 (0.48) ***</td>
      <td></td>
      <td>0.92 (0.49) +</td>
    </tr>
    <tr>
      <td>Early arrival of siblings</td>
      <td>0.34 (0.25)</td>
      <td>0.40 (0.26)</td>
      <td></td>
      <td>0.75 (0.35) *</td>
    </tr>
    <tr>
      <td>Both parents to one parent (without siblings)</td>
      <td>1.62 (0.34) ***</td>
      <td>2.06 (0.45) ***</td>
      <td></td>
      <td>2.59 (0.53) ***</td>
    </tr>
    <tr>
      <td>Early arrival of siblings & parental separation</td>
      <td>0.66 (0.35) +</td>
      <td>0.66 (0.48)</td>
      <td></td>
      <td>0.32 (0.60)</td>
    </tr>
    <tr>
      <td>One parent to both parents (with siblings)</td>
      <td>0.32 (0.58)</td>
      <td>0.37 (0.59)</td>
      <td></td>
      <td>0.71 (1.13)</td>
    </tr>
    <tr>
      <td>Both parents to one parent (with siblings)</td>
      <td>0.80 (0.28) **</td>
      <td>1.01 (0.30) ***</td>
      <td></td>
      <td>1.01 (0.40) *</td>
    </tr>
    <tr>
      <td>Age (ln)</td>
      <td>3.10 (0.28) ***</td>
      <td>3.14 (0.29) ***</td>
      <td>3.12 (0.28) ***</td>
      <td>2.96 (0.42) ***</td>
    </tr>
    <tr>
      <td>Women</td>
      <td>0.52 (0.18) **</td>
      <td>0.54 (0.18) **</td>
      <td>0.44 (0.18) *</td>
      <td>0.83 (0.27) **</td>
    </tr>
    <tr>
      <td>Ethnic origin: Switzerland (ref.)</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Eastern Europe</td>
      <td>-0.99 (0.26) ***</td>
      <td>-1.02 (0.26) ***</td>
      <td>-1.10 (0.26) ***</td>
      <td>-0.75 (0.43) +</td>
    </tr>
    <tr>
      <td>South-Western Europe</td>
      <td>-0.88 (0.29) **</td>
      <td>-0.88 (0.27) **</td>
      <td>-0.93 (0.28) ***</td>
      <td>-0.98 (0.42) *</td>
    </tr>
    <tr>
      <td>North-Western Europe & North America</td>
      <td>0.69 (0.33) *</td>
      <td>0.74 (0.33) *</td>
      <td>0.61 (0.33) +</td>
      <td>0.56 (0.51)</td>
    </tr>
    <tr>
      <td>Other continents</td>
      <td>-0.29 (0.31)</td>
      <td>-0.24 (0.30)</td>
      <td>-0.15 (0.30)</td>
      <td>-0.11 (0.44)</td>
    </tr>
    <tr>
      <td>Labor market integration</td>
      <td>0.52 (0.22) *</td>
      <td>0.56 (0.22) *</td>
      <td>0.51 (0.22) *</td>
      <td>0.89 (0.31) **</td>
    </tr>
    <tr>
      <td>Place of residence: Large population centers (ref.)</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Periurban & metropolitan centers</td>
      <td>0.22 (0.34)</td>
      <td>0.19 (0.35)</td>
      <td>0.11 (0.34)</td>
      <td>0.23 (0.50)</td>
    </tr>
    <tr>
      <td>Touristic municipalities</td>
      <td>0.48 (0.51)</td>
      <td>0.39 (0.51)</td>
      <td>0.46 (0.50)</td>
      <td>0.58 (0.65)</td>
    </tr>
    <tr>
      <td>Middle- and small-sized population centers</td>
      <td>0.17 (0.22)</td>
      <td>0.09 (0.23)</td>
      <td>0.13 (0.22)</td>
      <td>0.62 (0.33) +</td>
    </tr>
    <tr>
      <td>Periurban & commuting municipalities</td>
      <td>0.26 (0.36)</td>
      <td>0.35 (0.36)</td>
      <td>0.07 (0.35)</td>
      <td>0.44 (0.56)</td>
    </tr>
    <tr>
      <td>Outlying municipalities</td>
      <td>0.31 (0.28)</td>
      <td>0.24 (0.28)</td>
      <td>0.19 (0.28)</td>
      <td>0.56 (0.39)</td>
    </tr>
    <tr>
      <td>Underrepresented places of birth</td>
      <td>-0.19 (0.28)</td>
      <td>-0.14 (0.28)</td>
      <td>-0.29 (0.26)</td>
      <td>-0.14 (0.46)</td>
    </tr>
    <tr>
      <td>Divorce</td>
      <td></td>
      <td>0.09 (0.37)</td>
      <td>0.54 (0.23) *</td>
      <td></td>
    </tr>
    <tr>
      <td>Siblings</td>
      <td></td>
      <td>0.89 (0.33) **</td>
      <td>-0.03 (0.22)</td>
      <td></td>
    </tr>
    <tr>
      <td>Family socioeconomic status Qualified manual professions (ref.)</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Top management</td>
      <td></td>
      <td></td>
      <td></td>
      <td>-0.35 (0.81)</td>
    </tr>
    <tr>
      <td>Academic professions & senior management</td>
      <td></td>
      <td></td>
      <td></td>
      <td>1.11 (0.47) *</td>
    </tr>
    <tr>
      <td>Liberal professions</td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.47 (0.49)</td>
    </tr>
    <tr>
      <td>Other self-employed</td>
      <td></td>
      <td></td>
      <td></td>
      <td>-0.14 (0.60)</td>
    </tr>
    <tr>
      <td>Intermediate professions</td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.27 (0.52)</td>
    </tr>
    <tr>
      <td>Skilled non-manual professions</td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.88 (0.38) *</td>
    </tr>
    <tr>
      <td>Unskilled non-manual & manual professions</td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.48 (0.50)</td>
    </tr>
    <tr>
      <td>Nb obs.</td>
      <td>8700</td>
      <td>8700</td>
      <td>8700</td>
      <td>3456</td>
    </tr>
    <tr>
      <td>Nb ind.</td>
      <td>1624</td>
      <td>1624</td>
      <td>1624</td>
      <td>506</td>
    </tr>
    <tr>
      <td>Nb events</td>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>77</td>
    </tr>
    <tr>
      <td>Deviance</td>
      <td>1096.1</td>
      <td>1088</td>
      <td>1118.2</td>
      <td>550.23</td>
    </tr>
    <tr>
      <td>BIC</td>
      <td>1200.2</td>
      <td>1201.9</td>
      <td>1212.4</td>
      <td>724.57</td>
    </tr>
  </tbody>
</table>

+p < 0.1, * p < 0.05, **p < 0.01, ***p < 0.001

---


## Page 14

&lt;page_number&gt;96&lt;/page_number&gt;
F. Rossignon et al.

In all models, most coefficients of past co-residence trajectories are significant and go in the same direction. While the differences between the trajectories “both parents” and “both parents with siblings” are not significant, we observe a significant effect concerning the changes in the presence of siblings. Staying in the parental home after the departure of siblings increases the probability of leaving home. The “early arrival of siblings” also increases the hazard rate of leaving home. Besides, living with only one parent—with or without siblings—is associated with a higher risk of leaving home. There is no significant difference between the categories “early arrival of siblings and parental separation” and those who grew up with two parents. The same applies for young individuals who started by living in a lone parent household with siblings before moving to a bi-parental household.

We can see in Table 4 that the types of past co-residence trajectories are more informative than the aggregated indicators of parental divorce and of the presence of siblings. Indeed, in the second model that includes covariates related to these two aspects, the effects of childhood co-residence trajectories remain significant. These results show that behind the effect of having siblings or of having experienced a parental divorce, the past trajectory, i.e., the personal history, does matter.

The effect of control variables confirms our hypotheses. The hazard rate of leaving the parental home rises with increasing age. The departure from the parental home is also significantly influenced by the ethnic origin. Second-generation immigrants from Eastern or South-Western countries are less likely to leave home than Swiss natives. Conversely, having a Northern-Western European or a North-American background increases the risk of leaving home. Obtaining a first job significantly increases the likelihood of leaving home. Women are more likely to leave the parental home than men. However, we did not find a significant effect of the place of residence. Residents of middle and small centers have a higher probability of leaving home than inhabitants of large population centers, but this effect is only significant at the 0.1 level. Lastly, respondents whose parents had an academic profession or a senior management position when they were adolescents leave home more often than those whose parents had a qualified and manual profession. Children of skilled and non-manual workers are also more likely to leave the parental home.

## 5 Discussion

The occurrence of divorce does not play a strong role in the departure from the parental home, nor does that of having siblings. Conversely, childhood co-residence patterns influence the ways in which young adults leave the parental home. More precisely, the occurrence, the timing, and the sequencing of the events have a specific effect on home-leaving. For instance, two features have a significant and strong impact on the departure from the parental home: the lone parenthood configuration and the arrival and departure of siblings.

---


## Page 15

Sequence History Analysis (SHA)
&lt;page_number&gt;97&lt;/page_number&gt;

Having spent some years in a lone-parent household has a positive impact on the risk of leaving home. It does not seem to matter much if it occurred in the presence of siblings or not, as both situations lead to an increase in the likelihood of leaving home. In addition, these effects remained significant when controlling for the family socio-economic background. In other words, the negative effect of lone parenthood is not only explained by lower socio-economic background.

Having siblings matters when it comes to leaving the parental home. However, behind that simple fact, our method showed that birth order and arrivals or departures of siblings matter more. Moreover, when the family socio-economic background is taken into account, being an only child significantly increases the odds of leaving the parental home. Additionally, the departure from the parental home of siblings (most probably older siblings) encourages the remaining siblings to leave home. This could be interpreted as an imitation of the first-to-go individuals’ behavior.

In this study, we measured a link between childhood co-residence structures and departure from the parental home. However, the underlying mechanism linking these two concepts was not explained. For example, some longitudinal information regarding the relational quality in households was not available and could therefore not be included in the model. Moreover, the respondents were quite young. Consequently, only a small proportion of them had left the parental home at the end of the observation period. Hence, the observed effects could be mainly related to differences among early home-leavers.

## 6 Conclusion

The aim of this paper was two-fold. First, we proposed a methodological framework to estimate the effect of an unfolding trajectory on an upcoming event. We then applied the proposed approach to an original study of the effect of past co-residence trajectories on the departure from the parental home in Switzerland.

The results obtained with the combination of Sequence Analysis and Event History analysis provided results that would not have been obtained if each method had been used separately. However, the combination of Sequence Analysis and Event History Analysis was not an easy task as these two methods are based on very different approaches to life-course data. Sequence Analysis is based on a holistic approach of life course trajectories. The overall trajectory of each respondent is examined and compared with that of the other individuals. Consequently, Sequence Analysis aims to investigate the progression of individuals over their life course. Conversely, the focus in Event History Analysis is rather the investigation of the probabilistic distribution of life course events over time according to individual characteristics (Tuma and Hannan 1984; Mayer and Tuma 1990; Courgeau and Lelièvre 1992). The aim of Sequence History Analysis is thus to resolve this conflict between the two approaches. The proposed framework may have a much broader field of application. For instance, it could allow us to study how previous

---


## Page 16

&lt;page_number&gt;98&lt;/page_number&gt;
F. Rossignon et al.

professional trajectories are linked with the risk of dying at each age. We believe that Sequence History Analysis is a very promising tool. This method allows the combining of two traditions of investigating longitudinal data in life-course research: the holistic approach of Sequence Analysis and the processual approach of Event History Analysis.

Further work is needed to develop this approach more fully, to address its weaknesses and build on its strengths. For example, the proposed framework does not allow the drawing of causal interpretations of the results. We cannot state that parental divorce “causes” early departure from the parental home. In this respect, quasi-experimental designs, such as propensity-score matching, could be an interesting lead to follow. It would also be interesting to know to what extent the individuals have experienced relocation in their past. It might make young adults more likely to move out of the parental home and less hesitant about setting up their own independent households. Consequently, if possible, further analysis should take this factor into account.

**Acknowledgements** This paper benefited from the support of the Swiss National Centre of Competence in Research LIVES—Overcoming Vulnerability: Life Course Perspectives, which is financed by the Swiss National Science Foundation (Grant number: 51NF40-160590).

**References**

Allison, P. D. (2014). *Event history and survival analysis* (Vol. 46). Los Angeles: Sage Publications.

Aquilino, W. S. (1991). Family structure and home-leaving: A further specification of the relationship. *Journal of Marriage and the Family*, 53(4), 999–1010.

Avery, R., Goldscheider, F., & Speare, A. (1992). Feathered nest/gilded cage: Parental income and leaving home in the transition to adulthood. *Demography*, 29(3), 375–388.

Bernhardt, E., Gähler, M., & Goldscheider, F. (2005). Childhood family structure and routes out of the parental home in Sweden. *Acta Sociologica*, 48(2), 99–115.

Bianchi, S. M. (1987). Living at home: Young adults’ living arrangements in the 1980s. In *Annual Meeting of the American Sociological Association*, Chicago.

Blaauboer, M., & Mulder, C. H. (2010). Gender differences in the impact of family background on leaving the parental home. *Journal of Housing and the Built Environment*, 25(1), 53–71.

Blossfeld, H.-P., Golsch, K., & Rohwer, G. (2007). *Event history analysis with Stata*. New York: Psychology Press.

Brändle, K. (2017). The geography of social links among a young cohort in Switzerland. *LIVES Working Papers* (58), 1–33.

Buck, N., & Scott, J. (1993). She’s leaving home: But why? An analysis of young people leaving the parental home. *Journal of Marriage and the Family*, 55(4), 863–874.

Cherlin, A. J., Kiernan, K. E., & Chase-Lansdale, P. L. (1995). Parental divorce in childhood and demographic outcomes in young adulthood. *Demography*, 32(3), 299–318.

Chiuri, M. C., & Del Boca, D. (2010). Home-leaving decisions of daughters and sons. *Review of Economics of the Household*, 8(3), 393–408.

Courgeau, D., & Lelièvre, E. (1992). Analyse des données biographiques en démographie. In L. Coutrot & C. Dubar (Eds.), *Cheminements professionnels et mobilités sociales* (pp. 59–70). Paris: La Documentation Française.

---


## Page 17

Sequence History Analysis (SHA)
&lt;page_number&gt;99&lt;/page_number&gt;

Eerola, M. (2018). Case studies of combining sequence analysis and modelling. In G. Ritschard & M. Studer (Eds.), *Sequence analysis and related approaches: Innovative methods and applications*. Cham: Springer (this volume).

Eerola, M., & Helske, S. (2016). Statistical analysis of life history calendar data. *Statistical Methods in Medical Research*, 25(2), 571–597.

Elcheroth, G., & Antal, E. (2013). Echantillon de cohorte LIVES. Première vague. Technical report, University of Lausanne, Lausanne.

Ermisch, J., & Di Salvo, P. (1997). The economic determinants of young people’s household formation. *Economica*, 64(256), 627–644.

FORS and NCCR LIVES. (2015). LIVES Cohort Panel, Wave 1. Data available at URL: http://forscenter.ch/fr/our-surveys/swiss-household-panel/datasupport/telecharger-les-donnees-4/cohort-w1/

Gabadinho, A., Ritschard, G., Mueller, N. S., & Studer, M. (2011). Analyzing and visualizing state sequences in R with TraMineR. *Journal of Statistical Software*, 40(4), 1–37.

Gaehler, M., & Bernhardt, E. (2000). The impact of parental divorce, family reconstitution, and family conflict on nest-leaving in Sweden (pp. 6–8). Rostock: Max Planck Institut für Bevölkerungsforschung.

Gauthier, J.-A., Bühlmann, F., & Blanchard, P. (2014). Introduction: Sequence analysis in 2014. In Ph. Blanchard, F. Bühlmann, & J.-A. Gauthier (Eds.), *Advances in Sequence Analysis: Theory, Method, Applications* (pp. 1–19). New York: Springer.

Gierveld, J. D. J., Liefbroer, A. C., & Beekink, E. (1991). The effect of parental resources on patterns of leaving home among young adults in the Netherlands. *European Sociological Review*, 7(1), 55–71.

Goldscheider, F. K., & Goldscheider, C. (1998). The effects of childhood family structure on leaving and returning home. *Journal of Marriage and the Family*, 60(3), 745–756.

Gossens, L. (2001). Transition to adulthood: Developmental factors. In M. Corijn & E. Klijzing (Eds.), *Transition to Adulthood* (pp. 27–42). The Netherlands: Springer.

Holdsworth, C. (2000). Leaving home in Britain and Spain. *European Sociological Review*, 16(2), 201–222.

Iacovou, M., & Aassve, A. (2007). *Youth poverty in Europe*. New York: Joseph Rowntree Foundation.

Juang, L. P., Silbereisen, R. K., & Wiesner, M. (1999). Predictors of leaving home in young adults raised in Germany: A replication of a 1991 study. *Journal of Marriage and the Family*, 61(2), 505–515.

Kaufman, L., & Rousseeuw, P. J. (2005). *Finding groups in data: An introduction to cluster analysis*. Hoboken: Wiley.

Kiernan, K. (1992). The impact of family disruption in childhood on transitions made in young adult life. *Population Studies*, 46(2), 213–234.

Kiernan, K. (2006). Non-residential fatherhood and child involvement: Evidence from the Millen-nium Cohort study. *Journal of Social Policy*, 35(4), 651–669.

Lundevaller, E. H., Vikström, L., & Haage, H. (2018). Modelling mortality using life trajectories of disabled and non-disabled individuals in 19th-century Sweden. In G. Ritschard & M. Studer (Eds.), *Sequence analysis and related approaches: Innovative methods and applications*. Cham: Springer (this volume).

Madero-Cabib, I., Gauthier, J.-A., & Le Goff, J.-M. (2015). The influence of interlocked employment-family trajectories on retirement timing. *Work, Aging and Retirement*, 2(1), 38–53.

Mayer, K. U., & Tuma, N. B. (1990). *Event history analysis in life course research*. Madison: The University of Wisconsin Press.

McLanahan, S., & Carlson, M. S. (2004). Fathers in fragile families. In M.E. Lamb (Ed.), *The role of the father in child development* (Vol. 4, pp. 368–396). Hoboken: Wiley.

Mitchell, B. A. (1994). Family structure and leaving the nest: A social resource perspective. *Sociological Perspectives*, 37(4), 651–671.

---


## Page 18

&lt;page_number&gt;100&lt;/page_number&gt;
F. Rossignon et al.

Mitchell, B. A., Wister, A. V., & Burch, T. K. (1989). The family environment and leaving the parental home. *Journal of Marriage and the Family*, 51(3), 605–613.

Mulder, C. H. (2009). Leaving the parental home in young adulthood. In A. Furlong (Ed.), *Handbook of youth and young adulthood: New perspectives and agendas* (pp. 203–210). New York: Routledge.

R Core Team. (2016). *R: A language and environment for statistical computing*. Vienna: R Foundation for Statistical Computing.

Raftery, A. E. (1995). Bayesian model selection in social research. *Sociological Methodology*, 25, 111–164.

Sandefur, G. D., Eggerling-Boeck, J., & Park, H. (2008). Off to a good start? Postsecondary education and early adult life. In R.A. Settersten Jr., F.F. Furstenberg Jr. & R.G. Rumbaut (Eds.), *On the frontier of adulthood: Theory, research and public policy* (pp. 292–319). Chicago: University of Chicago Press.

Schizzerotto, A., & Lucchini, M. (2004). Transitions to adulthood. In R. Berthoud & M. Iacovou (Eds.), *Social Europe: Living standards and welfare states* (pp. 46–68). Cheltenham: Edward Elgar.

Scott, J., & Alwin, D. (1998). Retrospective versus prospective measurement of life histories in longitudinal research. In J.Z. Giele & G.H. Elder Jr. (Eds.), *Methods of life course research: Qualitative and quantitative approaches* (pp. 98–127). California: Sage Publications.

Studer, M. (2012). *Étude des inégalités de genre en début de carrière académique à l'aide de méthodes innovatrices d'analyse de données séquentielles*. Ph.D. thesis, University of Geneva, Geneva.

Studer, M. (2013). *Weighted Cluster library manual: A practical guide to creating typologies of trajectories in the social sciences with R*. *LIVES Working Papers*(24) (pp. 1–32).

Studer, M., & Ritschard, G. (2016). What matters in differences between life trajectories: A comparative review of sequence dissimilarity measures. *Journal of the Royal Statistical Society A*, 179(2), 481–511.

Swiss Federal Statistical Office. (2015). Secondary education II – Synoptic table. Retrieved from http://www.bfs.admin.ch/bfs/portal/fr/index/themen/15/04/00/blank/uebersicht.html, 2015-09-15.

Swiss Federal Statistical Office. (2016). Divorces and divortiality since 1876. Retrieved from: https://www.bfs.admin.ch/bfs/en/home/statistics/population/marriages-partnerships-divorces/divortiality.assetdetail.325774.html, 2017-12-04.

Thomsin, L., Le Goff, J.-M., & Sauvain-Dugerdil, C. (2004). Genre et étapes du passage à la vie adulte en Suisse. *Espace Populations Sociétés. Space Populations Societies*, 11(1), 81–96.

Tuma, N. B., & Hannan, M. T. (1984). *Social dynamics models and methods*. Orlando: Academic Press.

Zorlu, A., & Gaalen, R. (2016). Leaving home and destination of early nest leavers: Ethnicity, spaces and prices. *European Journal of Population*, 32(2), 1–25.

**Open Access** This chapter is licensed under the terms of the Creative Commons Attribution 4.0 International License (http://creativecommons.org/licenses/by/4.0/), which permits use, sharing, adaptation, distribution and reproduction in any medium or format, as long as you give appropriate credit to the original author(s) and the source, provide a link to the Creative Commons license and indicate if changes were made.

The images or other third party material in this chapter are included in the chapter's Creative Commons license, unless indicated otherwise in a credit line to the material. If material is not included in the chapter's Creative Commons license and your intended use is not permitted by statutory regulation or exceeds the permitted use, you will need to obtain permission directly from the copyright holder.

&lt;img&gt;CC BY&lt;/img&gt;