## Page 1

ARTICLE IN PRESS

# Scientific Data
## Article in Press

https://doi.org/10.1038/s41597-026-06682-w

# DEPRESS: Dataset on Emotions, Performance, Responses, Environment, and Satisfaction during COVID-19

Received: 22 August 2025
Accepted: 21 January 2026

Xingtong Guo, Angela C. Incollingo Rodriguez, Chao Wang, Elke A. Rundensteiner & Shichao Liu

Cite this article as: Guo, X., Incollingo Rodriguez, A.C., Wang, C. *et al.* DEPRESS: Dataset on Emotions, Performance, Responses, Environment, and Satisfaction during COVID-19. *Sci Data* (2026). https://doi.org/10.1038/s41597-026-06682-w

We are providing an unedited version of this manuscript to give early access to its findings. Before final publication, the manuscript will undergo further editing. Please note there may be errors present which affect the content, and all legal disclaimers apply.

If this paper is publishing under a Transparent Peer Review model then Peer Review reports will publish with the final article.

&lt;watermark&gt;ARTICLE IN PRESS&lt;/watermark&gt;

© The Author(s) 2026. **Open Access** This article is licensed under a Creative Commons Attribution 4.0 International License, which permits use, sharing, adaptation, distribution and reproduction in any medium or format, as long as you give appropriate credit to the original author(s) and the source, provide a link to the Creative Commons licence, and indicate if changes were made. The images or other third party material in this article are included in the article's Creative Commons licence, unless indicated otherwise in a credit line to the material. If material is not included in the article's Creative Commons licence and your intended use is not permitted by statutory regulation or exceeds the permitted use, you will need to obtain permission directly from the copyright holder. To view a copy of this licence, visit http://creativecommons.org/licenses/by/4.0/.

---


## Page 2

ARTICLE IN PRESS

# DEPRESS: Dataset on Emotions, Performance, Responses, Environment, and Satisfaction during COVID-19

Xingtong Guo<sup>a</sup>, Angela C Incollingo Rodriguez<sup>b</sup>, Chao Wang<sup>a, c</sup>, Elke A. Rundensteiner<sup>d, e</sup>, Shichao Liu<sup>a*</sup>

&lt;watermark&gt;ARTICLE IN PRESS&lt;/watermark&gt;
a. Civil, Environmental, and Architectural Engineering, Worcester Polytechnic Institute, 100 Institute Road, Worcester, MA, 01609, USA
b. Psychological and Cognitive Sciences, Worcester Polytechnic Institute, 100 Institute Road, Worcester, MA, 01609, USA
c. Psychotic Disorders Division, McLean Hospital, Harvard Medical School, Longwood Ave, Boston, MA 02115, USA
d. Computer Science, Worcester Polytechnic Institute, 100 Institute Road, Worcester, MA, 01609, USA
e. Data Science, Worcester Polytechnic Institute, 100 Institute Road, Worcester, MA, 01609, USA

*Corresponding email: sliu8@wpi.edu

COVID-19 posed a significant threat to the mental health of the population in general and college students in particular, severely disrupting their daily routines due to protective measures and lockdown policies. The abrupt transition from in-person to online learning further introduced uncertainty regarding academic performance. To comprehensively assess the impacts of the pandemic on college students, this study collected longitudinal data from June 2020 to June 2021, involving 180 undergraduate students at Worcester Polytechnic Institute. The dataset includes demographic and socioeconomic status information of participants, measures of mental health outcomes, online student engagement, computer and Internet performance, daily activity diary, general indoor environment satisfaction, Fitbit data, sensor measured indoor environment quality, facial expression, and GPA. To our best knowledge, this dataset is also the first dataset that covers multimodal assessment of mental health outcomes, online learning, and potential influencing variables during COVID-19. Data was gathered through online surveys, video recordings, IoT indoor environmental sensors, and Fitbit wristbands.

## Background & Summary

In May 2020, the World Health Organization (WHO) officially declared COVID-19 as a pandemic [1], prompting governments worldwide to implement measures to protect their citizens. Quarantine emerged as one of the most widely adopted policies, significantly disrupting daily routines. In response, most universities suspended in-person classes and transitioned to online instruction. While online learning is not

---


## Page 3

ARTICLE IN PRESS

new and its effectiveness has been explored in previous studies [2], it was typically considered a supplementary tool in education before the pandemic. Meanwhile, due to the quarantine, students spend more time in their apartment or house where the indoor environment was not specifically designed for learning. How the learning performance may be affected by the bedroom indoor environment has not been well studied.

During the pandemic, mental health emerged as one of the severe issues that threatened students' learning and well-being. Mental health problem had been identified already as a critical problem prior to the pandemic with around one in five young adults having been diagnosed with a mental health issue [3]. This situation was exacerbated due to the pandemic and the restrictive policies it brought, which led to young adults facing increased uncertainty in academic success, future career prospects, and social life [4].

The aim of this study was thus to comprehensively understand how college students' mental health and learning were affected by the pandemic over time as pandemic-related measures were rolled out and increasingly impacted daily life. This paper presents a longitudinal dataset, DEPRESS (Dataset on Emotions, Performance, Responses, Environment, and Satisfaction of Students), which covers mental health states, indoor environment quality, learning performance, and daily routine. DEPRESS was collected from June 2020 to June 2021 containing a rich variety of modalities of 180 undergraduate students from Worcester Polytechnic Institute (WPI). We recorded the demographic and socioeconomic status information of the participants; measures of mental health outcomes; online student engagement (OSE); computer and Internet performance; daily activity diary; general indoor environment satisfaction; Fitbit data; sensor-measured indicators of indoor environmental quality (IEQ), including CO₂, particulate matter (PM_{2.5}), lighting, noise, and Volatile Organic Compounds (VOCs); general indoor environmental satisfaction; facial expression data during online courses; and academic transcripts to calculate Grade Point Average (GPA).

This study enrolled 180 participants. Among them, 174 completed daily activity diary surveys, 163 responded to mental health assessments, OSE, and computer and internet performance surveys, and 126 provided synced Fitbit data. Indoor environmental quality (IEQ) data were collected using AWAIR sensors from the bedrooms of 42 participants. This longitudinal dataset enables us to examine how college students adapted to the sudden pandemic outbreaks and how their mental health and learning performance evolved during such events.

To the best of our knowledge, this is the first dataset to capture students' satisfaction with the physical environment while learning outside the classroom. Its uniqueness also lies in the inclusion of IEQ measurements collected via IoT sensors. Moreover, the dataset enables the analysis of students' emotional states during online courses through extracted facial expressions, as well as investigations into how IEQ influences emotional states and mental health.

**Methods**

**Data collection.** This study utilized longitudinal data gathered from undergraduate students at Worcester Polytechnic Institute (WPI), Massachusetts, from June 2020 to May 2021, spanning three semesters and thus composed of three different cohorts: 1) Summer Semester (June–August 2020): All students attended classes online while residing at home. 2) Fall Semester (Late August–December 2020): Students could opt for on-campus living, which included routine COVID testing. Instruction was predominantly online, with some courses offered in a hybrid format. 3) Spring Semester (February–May 2021): Most students lived on campus and participated in routine COVID testing. Most classes remained online, supplemented by some
&lt;watermark&gt;APFCEN 5RFSs&lt;/watermark&gt;

---


## Page 4

ARTICLE IN PRESS

hybrid course offerings. To reduce the burden on participants and maintain a consistent data collection duration across individuals, each participant joined only one cohort. This design reduces individual workload, minimizes dropout, and helps preserve the reliability of the longitudinal patterns observed across the study. Figure 1 provides the major events that happened within the survey span based on the academic calendar of WPI.

&lt;img&gt;Figure 1. Timeline of major events during the study period and daily new COVID-19 case number in Massachusetts [5]. More than 68.6% of the participants were in Massachusetts during the study&lt;/img&gt;

During the study, participants were asked to fill out a series of surveys. As for the mental health outcomes survey, participants completed the Positive and Negative Affect Schedule (PANAS, scale from 20 to 100) [6], the four-item Perceived Stress Scale (PSS, score scale from 0 to 4) [7], the ten-item Center for Epidemiological Studies-Depression Scale (CES-D, score scale from 0 to 30) [8] with a clinical cut-off of 10 [9], indicating the presence of depressive symptoms to a degree that may require professional help. The State-Trait Anxiety Inventory [10] (STAI, score scale from 20 to 80), and the nineteen-item online student engagement (OSE, scale from 0 to 76) [11].

To understand the change of students' daily routines during the pandemic, we adapted the daily activity survey from Richard et al [12]. Students reported the duration of time spent on various activities, including school-related (e.g., attending online classes, reading course materials), social (e.g., phone or in-person conversations), and entertainment (e.g., listening to music, watching TV). We also collected self-reported information on students' demographic features, socioeconomic status, and satisfaction with internet speed and computer hardware such as central processing unit (CPU) and speaker. We required students to complete the demographic features and socioeconomic status surveys before officially participating in the study; however, we still allowed them to proceed with the study even if they had not filled it out.

In addition to the survey data, participants were asked to wear a Fitbit wristband (Inspire 2, USA) to collect sleep patterns, heart rate, calories burned, and daily activity metrics including walking distance, and step count. Some participants also agreed to install AWAIR sensors (OMNI 1, USA) in their bedrooms to monitor IEQ. Additionally, a subset of participants consented to recording only their own facial expression videos while attending online courses. The number of participants contributing to each type of measurement is summarized in Table 1.

---


## Page 5

ARTICLE IN PRESS

Because different mental health outcomes fluctuate at different temporal scales, the data were collected with varying frequencies. Depression and anxiety tend to change more gradually, so these measures were administered monthly [13]. In contrast, positive and negative affect schedule, stress can vary more rapidly; therefore, the corresponding surveys was conducted weekly [14].

<table>
  <thead>
    <tr>
      <th>Measure type</th>
      <th>Measurement description</th>
      <th>2020 Summer (N, %)<sup>1</sup></th>
      <th>2020 Fall (N, %)<sup>1</sup></th>
      <th>2021 Spring (N, %)<sup>1</sup></th>
      <th>Frequency</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="8">Survey-based</td>
      <td>Demographic and socioeconomic status</td>
      <td>36<br>(75.0%)</td>
      <td>83<br>(85.5%)</td>
      <td>65<br>(64.6%)</td>
      <td>Once</td>
    </tr>
    <tr>
      <td>PSS, PANAS</td>
      <td>24<br>(83.0%)</td>
      <td>77<br>(68.0%)</td>
      <td>62<br>(75.3%)</td>
      <td>Weekly</td>
    </tr>
    <tr>
      <td>CES-D, STAI</td>
      <td>24<br>(97.4%)</td>
      <td>77<br>(88.7%)</td>
      <td>62<br>(94.4%)</td>
      <td>Monthly</td>
    </tr>
    <tr>
      <td>OSE, computer and internet performance</td>
      <td>24<br>(83.0%)</td>
      <td>77<br>(68.0%)</td>
      <td>62<br>(75.3%)</td>
      <td>Weekly</td>
    </tr>
    <tr>
      <td>Daily activity diary</td>
      <td>25<br>(55.1%)</td>
      <td>73<br>(51.8%)</td>
      <td>54<br>(76.7%)</td>
      <td>Daily</td>
    </tr>
    <tr>
      <td>General indoor environmental satisfaction</td>
      <td>24<br>(76.7%)</td>
      <td>77<br>(83.6%)</td>
      <td>62<br>(68.3%)</td>
      <td>Weekly</td>
    </tr>
    <tr>
      <td rowspan="2">Sensor-based</td>
      <td>Fitbit</td>
      <td>14 (NA)</td>
      <td>64 (NA)</td>
      <td>48 (NA)</td>
      <td>1-min</td>
    </tr>
    <tr>
      <td>IEQ (by AWAIR)</td>
      <td>11 (NA)</td>
      <td>15 (NA)</td>
      <td>16 (NA)</td>
      <td>15-min</td>
    </tr>
    <tr>
      <td>Video-based</td>
      <td>Facial expression</td>
      <td>0 (NA)</td>
      <td>39<br>(100%)</td>
      <td>45<br>(100%)</td>
      <td>At least once</td>
    </tr>
    <tr>
      <td>Academic records</td>
      <td>Transcript</td>
      <td>36<br>(38.9%)</td>
      <td>83<br>(50.6%)</td>
      <td>65<br>(35.4%)</td>
      <td>Once</td>
    </tr>
  </tbody>
</table>

&lt;watermark&gt;ARTICLE IN PRESS&lt;/watermark&gt;

**Table 1. Summary of collected data sources and the number of participants contributing to each data type.**

Note 1: N indicates the number of participants contributing at least one valid observation in the corresponding semester; % denotes the completion rate. Completion rate was not applicable (NA) for continuous sensor-based data, since 100% data coverage was neither expected nor feasible by design. For example, Fitbit wear time varied naturally across participants, and gaps in data primarily reflect real-world usage constraints rather than missingness relative to a fixed target.

**Participants and recruitment.** Undergraduate students who had taken at least one online course during the pandemic were eligible. All participants provided signed consent before completing any surveys or measurements and were subsequently assigned a unique ID to replace their real name. Consenting participants were compensated based on task adherence. To enroll participants, we posted the study on the Student Government Association's Facebook account, sent emails to undergraduates, and asked faculty to advertise the study in their classes. In the end, we enrolled 182 students across Summer 2020 (n=36), Fall 2020 (n=83), and Spring 2021 (n=65) cohorts. At the start of the study, participants were invited to complete the demographic and socioeconomic questionnaire; completion was optional, and they can skip any questions they feel uncomfortable with. Detailed demographic features of the participants are summarized in Figure 2 (n = 140, missing: 40). Fitbit wristbands and AWAIR sensors were picked up or shipped to the participants before their participation.

---


## Page 6

ARTICLE IN PRESS

&lt;img&gt;A series of pie charts illustrating demographic features and social economic status of participants (n=140). The charts include:
- Sex: Female 68.8%, Male 30.5%, Prefer not to answer 0.7%.
- School year: 1st year 25.0%, 2nd year 25.7%, 3rd year 21.4%, 4th year 27.9%.
- Ethnicity: White 56.4%, Asia 26.4%, Hispanic/Latino 10.7%, Black or Africa American 2.9%, Others 3.6%.
- No. of people living in the house: 1-3 34.3%, 4-6 63.6%, 6+ 0.01%, Prefer not to answer 0.01%.
- Annual income ($): <10k 1.4%, 10-49k 27.9%, 50-99k 10.7%, 100-149k 32.1%, >149k 26.4%, Prefer not to answer & Really do not know 1.4%.
- No. of bedroom: 3-4 82.1%, 1-2 11.4%, 5+ 5.7%, Prefer not to answer 0.7%.
- No. of possessed car: 0-2 36.4%, 3-4 51.4%, 5+ 11.4%, Prefer not to answer 0.7%.
- Employment: Private company & Business 74.1%, Government 15.1%, Individual & self-employment 15.1%, Prefer not to answer 2.2%.
- Home ownership: Own with mortgage 52.9%, Own without mortgage 32.1%, Rent 7.9%, Not sure 7.1%.&lt;/img&gt;

**Figure 2. Summary of demographic features and social economic status of participants (n=140)**

**Ethics approval.** This study was conducted with the approval by the Institutional Review Board (20-0656) of Worcester Polytechnic Institute. All participants provided informed consent prior to participation. Participants received a $30 VISA gift card at the end of their cohort and were allowed to keep the Fitbit as compensation. Participation was voluntary, and refusal to participate did not result in any penalty.

The dataset released in this paper is fully anonymized, with all direct identifiers removed and only indirect, non-identifiable variables retained. In accordance with IRB guidance, the anonymized dataset does not contain information that can be used to identify individual participants, and therefore its public sharing is permitted under the approved protocol.

**Data pre-processing.** For the PANAS survey, each response option was translated from 1-5, with 1 meaning “very slightly or not at all” and 5 meaning “extremely.” The subject’s positive affect score was determined by summing the response scores of the items 1, 3, 5, 9, 10, 12, 14, 16, 17 and 19, while the negative affect score corresponds to summing the response scores of the 2, 4, 6, 7, 8, 11, 13, 15, 18 and 20 [6].

For the PSS-4 survey, responses to its questions were coded based on five options. For questions one and four, this coding was as follows: “Never” = 0; “Almost Never” = 1; “Sometimes” = 2; “Fairly Often” = 3;

---


## Page 7

ARTICLE IN PRESS

and “Very Often” = 4. For questions two and three, this scoring was reversed, with responses coded as: “Never” = 4; “Almost Never” = 3; “Sometimes” = 2; “Fairly Often” = 1; and “Very Often” = 0. Finally, the average value of the coded responses across all four questions was calculated to determine the PSS-4 score [15].

For the CES-D survey, responses were coded as follows: “Rarely or None of the Time” = 0; “Some or a Little of the Time” = 1; “Occasionally or a Moderate Amount of Time” = 2; and “Most or All of the Time” = 3. For items five through eight, this scoring was reversed (e.g., “Most or All of the Time” became 0 instead of 3 for these specific items, with other responses similarly adjusted). Finally, the sum of these coded values across all ten items was calculated to determine the total score [16].

For the STAI state anxiety survey, responses were coded as follows: “Not at all” = 1, “Somewhat” = 2, “Moderately so” = 3, and “Very much so” = 4. This coding was applied to items 3, 4, 6, 7, 9, 12, 13, 14, 17, and 18. For items 1, 2, 5, 8, 10, 11, 15, 16, 19, and 20, the coding was reversed. Finally, the sum of all items was calculated [10].

For the OSE survey, responses were coded as follows: “Not at all characteristics of me” = 0; “Not really characteristic of me” =1; “Moderately characteristic of me” =2; “Characteristic of me” = 3; “Very characteristic of me” = 4.

For the general environmental satisfaction survey, thermal sensations were coded from -3(cold) to 3 (hot) and satisfaction related questions were coded as: -3 (very dissatisfied) to 3 (very satisfied). Interference or enhancement of indoor environment factors with learning were coded as: “Significantly interferes” = -3; “Interferes” = -2; “Somewhat interferes” = -1; “Neither interferes nor enhances” = 0; “Somewhat enhances” = 1; “Enhances” = 2; “Significantly enhances” = 3. The amount of outdoor air and daylight (natural light) were coded as: “Far too little” = -3; “Moderately too little” = -2; “Slightly too little” = -1; “Neither too much nor too little” = 0; “Slightly too much” = 1; “Moderately too much” = 2; “Far too much” = 3. Reported learning enhancement: “Decreased 20% or more” = -3; “Decreased 10%” = -2; “Decreased 5%” = -1; “Neither increased nor decreased” = 0; “Increased 5%” = 1; “Increased 10%” = 2; “Increased 20% or more” = 3. Students’ GPAs were coded with 3 for A, 2 for B and 1 for C that was the lowest possible grade in this study. There are no pluses and minuses grades for each class.

For the computer and internet performance and survey, the satisfaction-related questions were coded as: -3 (very dissatisfied) to 3(very satisfied). The download and upload speed were filled in by the participants.

For the daily activity time diary survey, responses were coded as follows: “0 min” = 0, “15 min” =1, “30 min” = 2, “45 min” =3, “more than 45 min” = 4. All detailed coding schemes for the survey responses are summarized in Table 2.

<table>
<thead>
<tr>
<th>Survey</th>
<th>Items</th>
<th>Response (coding)</th>
</tr>
</thead>
<tbody>
<tr>
<td>PANAS</td>
<td>All items</td>
<td>Very slightly or not at all (1)<br>A little (2)<br>Moderately (3)<br>Quite a bit (4)<br>Extremely (5)</td>
</tr>
<tr>
<td>PSS-4</td>
<td>Q1, Q4<br>(Reverse for Q2, Q3)</td>
<td>Never (0)<br>Almost never (1)<br>Sometimes (2)</td>
</tr>
</tbody>
</table>

---


## Page 8

ARTICLE IN PRESS

<table>
  <tr>
    <td>CES-D</td>
    <td>All items</td>
    <td>Fairly often (3)<br>Very often (4)<br>Rarely or None of the Time (0)<br>Some or a Little of the Time (1)<br>Occasionally or a Moderate Amount of Time (2)<br>Most or All of the Time (3)</td>
  </tr>
  <tr>
    <td>STAI</td>
    <td>Q3, 4, 6, 7, 9, 12, 13, 14, 17, 18<br>(Reverse for Q1, 2, 5, 8, 10, 11, 15, 16, 19, 20)</td>
    <td>Not at all (1)<br>Somewhat (2)<br>Moderately so (3)<br>Very much so (4)</td>
  </tr>
  <tr>
    <td>OSE</td>
    <td>All items</td>
    <td>Not at all characteristics of me (0)<br>Not really characteristic of me (1)<br>Moderately characteristic of me (2)<br>Characteristic of me (3)<br>Very characteristic of me (4)</td>
  </tr>
  <tr>
    <td>Thermal sensations</td>
    <td>All items</td>
    <td>Cold (-3)<br>Cool (-2)<br>Slightly cool (-1)<br>Neutral (0)<br>Slightly warm (1)<br>Warm (2)<br>Hot (3)</td>
  </tr>
  <tr>
    <td>Satisfaction related survey</td>
    <td>All items</td>
    <td>very dissatisfied (-3)<br>Dissatisfied (-2)<br>Slightly dissatisfied (-1)<br>Neutral (0)<br>Slightly satisfied (1)<br>Satisfied (2)<br>very satisfied (3)</td>
  </tr>
  <tr>
    <td>Interference or enhancement of indoor environment factors</td>
    <td>All items</td>
    <td>Significantly interferes (-3)<br>Interferes (-2)<br>Somewhat interferes (-1)<br>Neither interferes nor enhances (0)<br>Somewhat enhances (1)<br>Enhances (2)<br>Significantly enhances (3)</td>
  </tr>
  <tr>
    <td>The amount of outdoor air and daylight</td>
    <td>All items</td>
    <td>Far too little (-3)<br>Moderately too little (-2)<br>Slightly too little (-1)<br>Neither too much nor too little (0)<br>Slightly too much (1)<br>Moderately too much (2)<br>Far too much (3)</td>
  </tr>
  <tr>
    <td>Reported learning enhancement</td>
    <td>All items</td>
    <td>Decreased 20% or more (-3)<br>Decreased 10% (-2)<br>Decreased 5% (-1)<br>Neither increased nor decreased (0)<br>Increased 5% (1)<br>Increased 10% (2)</td>
  </tr>
</table>

---


## Page 9

ARTICLE IN PRESS

<table>
  <tr>
    <td></td>
    <td></td>
    <td>Increased 20% or more (3)</td>
  </tr>
  <tr>
    <td>GPA</td>
    <td>All items</td>
    <td>A (1); B (2); C (3)</td>
  </tr>
  <tr>
    <td>Daily activity time diary</td>
    <td>All items</td>
    <td>0 min (0)<br>15 min (1)<br>30 min (2)<br>45 min (3)<br>More than 45 min (4)</td>
  </tr>
</table>

**Table 2. Numerical coding for corresponding survey measures.**

Fitbit and AWIR data were downloaded directly from the official platforms and are provided in their original, raw formats with no cleaning, transformation, or post-processing.

Facial features were extracted from video recordings of students' (only the participants') facial expressions during online courses using OpenFace 2.2.0 [17]. OpenFace analyzes individual video frames to detect facial landmarks (like eyes, lips, nose, cheeks) and then calculates Facial Action Units (FAUs), which are specific muscle movements corresponding to facial expressions (e.g., an outer brow raise). When OpenFace processes a video and successfully extracts features, it generates a row for each frame with the positions for facial landmarks and action units as columns. However, not all frames were successfully detected, and the failed frames were indicated in the final output data. The detailed video data processing procedure is shown in Figure 3. For this dataset, only FAUs presence and intensity values were retained. No images, video frames, audio, or facial landmark coordinates were stored or shared.

&lt;watermark&gt;FIN PRESENT&lt;/watermark&gt;

```mermaid
graph TD
    subgraph Data Pre-processing
        A[Raw video data] --> B[Anonymization: rename files using participants IDs]
        A --> C[Data cleaning: remove duplicate submissions]
        B --> D[Cleaned video data]
        C --> D
    end

    subgraph Feature Extraction
        E[Cleaned video data] --> F[OpenFace]
        F --> G[Extracted feature (csv): calculate Facial Action Units]
    end

    D --> H[Feature Extraction]
    H --> G
```

**Figure 3. Workflow of the facial expression extraction process. Data was anonymized by renaming files with participant IDs and cleaned by removing duplicate submissions. OpenFace was then used to extract FAUs.**

## Data Records

The files have been deposited in the Harvard Dataverse [18] and are publicly accessible at: https://doi.org/10.7910/DVN/SJ8ILQ .

---


## Page 10

ARTICLE IN PRESS

**Questionnaire.** In this study, all questions were implemented in Qualtrics. The detailed questions in each section have been summarized in the dataset repository.

**Main dataset.** The dataset provided consists of four folders, capturing all the data collected during the study. Each folder summarizes data on the same topics. All files include information on ID, date, and the responses to the questions. Detailed content and structure of the dataset is listed in Figure 4.

<mermaid>
graph TD
    A[Dataset/]
    B[Mental_Health_Outcomes/]
    C[CES-D_STAI.csv]
    D[PANAS_PSS.csv]
    E[IEQ/]
    F[Bedroom_IEQ.csv]
    G[General_Environmental_Satisfaction_Survey.csv]
    H[Learning_Performance/]
    I[Facial expression/]
    J[OSE_Computer_Internet.csv]
    K[Grades.csv]
    L[Daily_Routine/]
    M[Activity_Time_Diary.csv]
    N[Fitbit_Data/]
    O[Consent form and questionnaires.doc]
    P[Demographic and socioeconomic status.csv]
    Q[Readme.doc]

    A --> B
    A --> E
    A --> H
    A --> L

    B --> C
    B --> D

    E --> F
    E --> G

    H --> I
    H --> J
    H --> K

    L --> M
    L --> N

    N --> O
    N --> P
    N --> Q
</mermaid>

**Figure 4.** Dataset directory tree. **Mental_Health_Outcomes:** stress, positive and negative affect schedule, depression, and anxiety survey responses; **IEQ:** indoor environmental quality survey data and AWAIR measured data; **Learning_Performance:** extracted facial expression features, OSE survey responses, and GPA data; **Daily_Routine:** daily activity time diary records and raw Fitbit data. Additional supporting documents include the consent form, demographic and socioeconomic data, and the dataset readme file.

**Technical Validation**

**Questionnaire data.** To ensure the quality and reliability of the questionnaire data, several validation steps were conducted. Internal consistency of the mental health outcome and OSE surveys were assessed using the Cronbach’s alpha test [19], a standard measure of scale reliability. PSS and the CES-D both demonstrated strong reliability, each with a Cronbach’s alpha of 0.87. The STAI yielded an even higher alpha of 0.95, indicating excellent internal consistency. Similarly, the OSE achieved an alpha of 0.90.

Figure 5 shows the distribution of self-reported daily activity diary, which captures participants’ engagement in school-related, social, and entertainment activities. The responses ranged from 0 min to more than 45 min, reflecting time reporting. The boxplots reveal a wide spread of responses across different activity types, indicating variability in participants' daily routines. Entertainment activities such as "Watching TV" and "Listening to music" show relatively high median values and wide interquartile ranges, whereas cognitively demanding activities like "Reading school-related material not on the Internet" exhibit
&lt;watermark&gt;Article&lt;/watermark&gt;

---


## Page 11

ARTICLE IN PRESS

lower engagement and higher skewness. The detailed time spent diaries of each participant across the cohorts are provided in the dataset.

&lt;img&gt;Figure 5. Boxplot of time of daily activity diaries&lt;/img&gt;

**Figure 5.** *Boxplot of time of daily activity diaries*

Figure 6 shows the distribution of participants' self-reported perceptions of environmental satisfaction and its interference or enhancement with learning. These items were rated on a Likert scale from -3 to +3 as described in Section Data pre-processing. The data show a relatively balanced distribution across categories, with most items exhibiting a reasonable spread.
&lt;watermark&gt;ARTICLE&lt;/watermark&gt;

---


## Page 12

ARTICLE IN PRESS

&lt;img&gt;Boxplot of general indoor environment satisfaction survey data.&lt;/img&gt;

**Figure 6.** Boxplot of general indoor environment satisfaction survey data. **Thermal sensation** was rated from -3 (cold) to 3 (hot); **Satisfaction variables** were rated from -3 (very dissatisfied) to 3 (very satisfied); **Interference with learning** was coded from -3 (significantly interferes) to 3 (significantly enhances), with 0 indicating no effect; **Outdoor air and daylight amounts** were rated from -3 (far too little) to 3 (far too much); **Reported learning outcomes** ranged from -3 (decreased by 20%) to 3 (increased by 20%).

**Sensor calibration.** AWAIR sensors and Fitbits used in this study were brand newly purchased for the study and thus were freshly inspected and calibrated by the manufacturers. Details about the AWAIR sensor’s resolution accuracy are provided in Table 3. The accuracy of Fitbit data is discussed in a review [20], according to which Fitbit devices are reasonably accurate for step counts but less reliable for energy-expenditure measures. Studies also indicate that Fitbit's heart rate accuracy at rest is high and comparable to chest straps, Polar monitors, and electrocardiograms (ECG) [21,22].

<table>
<thead>
<tr>
<th>Monitored parameter</th>
<th>Device</th>
<th>Resolution</th>
<th>Measurement accuracy</th>
</tr>
</thead>
<tbody>
<tr>
<td>Temperature</td>
<td>AWAIR</td>
<td>0.015°C</td>
<td>±0.2°C</td>
</tr>
<tr>
<td>CO<sub>2</sub></td>
<td>AWAIR</td>
<td>1ppm</td>
<td>±75ppm</td>
</tr>
<tr>
<td>Relative Humidity</td>
<td>AWAIR</td>
<td>0.01% RH</td>
<td>±2%</td>
</tr>
<tr>
<td>TVOCs</td>
<td>AWAIR</td>
<td>1ppb</td>
<td>±15%</td>
</tr>
<tr>
<td>PM<sub>2.5</sub></td>
<td>AWAIR</td>
<td>1 μg/m³</td>
<td>±15 μg/m³</td>
</tr>
</tbody>
</table>

**Table 3.** Accuracy of AWAIR sensor data.

**Quality Control Prior to Feature Extraction.** Raw video recordings were used only for feature extraction and are not included in the released dataset to protect participant privacy. A random subset of videos (20%) was manually reviewed to assess image clarity, lighting consistency, and audio quality. This pre-checking step was necessary to ensure that the extracted facial-expression features were reliable and not distorted by poor video quality.

---


## Page 13

ARTICLE IN PRESS

**FAUs Data.** To quantitatively assess the video tracking quality, we calculated the frame-level facial feature extraction success rate for each video. Across all videos, the average percentage of successfully detected frames was 87.32%, indicating a generally high tracking quality suitable for downstream analyses.

**Data availability**

All these files have been deposited in the Harvard Dataverse [18] and are publicly accessible at:
https://doi.org/10.7910/DVN/SJ8ILQ.

**Code availability**

No custom code was developed for this dataset. The facial expression features were extracted using OpenFace, an open-source facial behavior analysis toolkit available online [17].

**Strengths and limitations**

This study collected a comprehensive set of multimodal data that integrates Fitbit physiological signals, facial-expressions, indoor environmental monitoring, and self-reported survey responses. Such multimodal data collection aligns with established practices in health and behavioral research, where combining physiological, behavioral, and environmental parameters is recognized for producing richer and more reliable insights than unimodal approaches.

Despite its strengths, this study has several limitations. First, the summer cohort had a smaller sample size than other two cohorts because fewer students enroll in summer courses, which may also limit the robustness of semester-to-semester comparisons. Second, data were collected only during the COVID-19 pandemic; the lack of post-pandemic data restricts our ability to compare changes in students’ mental health status and online learning performance over time. Third, this longitudinal dataset consists of three separate cohorts, and the participants differ across cohorts. This inconsistency may affect the continuity of the time-series data analyses. Finally, because survey questions were not mandatory in order to reduce participant burden and survey dropout, missing data were observed in the demographic and socioeconomic surveys and in academic transcripts. These missing data may not be randomly distributed across demographic or socioeconomic groups or mental health conditions. This limitation should be considered when interpreting the findings and highlights the need for future studies to adopt strategies that better capture demographic information from psychologically vulnerable populations.

**Acknowledgements**

This research was supported by U.S. National Science Foundation (#2028224 and #1931077). Any opinions, findings, conclusions or recommendations expressed in this work are those of the authors and do not necessarily reflect the views of the National Science Foundation. We also thank Soroush Farzin, Steven Van Dessel, and Jacob Whitehill for their excellent support in the project.

**Ethics declarations**

**Competing interests**

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

---


## Page 14

ARTICLE IN PRESS

Reference

[1] World Health Organization, WHO Director-General’s opening remarks at the media briefing on COVID-19 - 11 March 2020, (2020). https://www.who.int/director-general/speeches/detail/who-director-general-s-opening-remarks-at-the-media-briefing-on-covid-19---11-march-2020 (accessed June 23, 2025).
[2] M.D.B. Castro, G.M. Tumibay, A literature review: efficacy of online learning courses for higher education institution using meta-analysis, Educ. Inf. Technol. 26 (2021) 1367–1385. https://doi.org/10.1007/s10639-019-10027-z.
[3] S. Griggs, Hope and Mental Health in Young Adult College Students: An Integrative Review, J. Psychosoc. Nurs. Ment. Health Serv. 55(2) (2017) 28–35.
[4] A. Aristovnik, D. Keržič, D. Ravšelj, N. Tomaževič, L. Umek, Impacts of the COVID-19 Pandemic on Life of Higher Education Students: A Global Perspective, Sustain. Switz. 12 (2020) 1–34.
[5] COVID-19 reporting | Mass.gov, (n.d.). https://www.mass.gov/info-details/covid-19-reporting (accessed July 10, 2025).
[6] D. Watson, L.A. Clark, A. Tellegen, Development and validation of brief measures of positive and negative affect: The PANAS scales, J. Pers. Soc. Psychol. 54 (1988) 1063–1070. https://doi.org/10.1037/0022-3514.54.6.1063.
[7] R. Cohen, S., Kamarck, T. and Mermelstein, Perceived stress scale, Meas. Stress Guide Health Soc. Sci. 10(2) (n.d.) 1–2.
[8] R. LS., The CES-D Scale: A Self-Report Depression Scale for Research in the General Population, Appl. Psychol. Meas. 1(3) (1977) 385–401.
[9] M.E. Andren, B.W. Carter, A.J. Malmgren, L.D. Patrick, Screening for Depression in Well Older Adults: Evaluation of a Short Form of the CES-D, Am. J. Prev. Med. 10 (1994) 77–84.
[10]A. Spielberger, Charles D., Gonzalez-Reigosa, Fernando Martinez-Urrutia, L.F. Natalicio, D.S. Natalicio, State-Trait Anxiety Inventory, Int. J. Psychol. 5 (1791) 3–4. https://doi.org/10.4135/9781483365817.n1316.
[11]M.D. Dixson, Measuring student engagement in the online course: the Online Student Engagement scale (OSE), Online Learn. J. 19 (2015).
[12]R. Emanuel, Adams ,Jim, Baker ,Kim, Daufin ,E. K., Ellington ,Coke, Fitts ,Elizabeth, Himsel ,Jonathan, Holladay ,Linda, D. and Okeowo, How College Students Spend Their Time Communicating, Int. J. List. 22 (2008) 13–28. https://doi.org/10.1080/10904010701802139.
[13]B.W. Osweiler, T. Rammaha, H.S. Szlyk, N.A. Dell, J. Macon, N.C. Jacobson, C. Burley, M. Goodman, P.A. Cavazos-Rehg, A.T. Ramsey, Co-designing a mobile application to reduce self-stigma for people with opioid use disorder during pregnancy and the postpartum period, Front. Psychiatry 16 (2025). https://doi.org/10.3389/fpsyt.2025.1607652.
[14]L.D. Feld, A. Shusterman, Into the pressure cooker: Student stress in college preparatory high schools, J. Adolesc. 41 (2015) 31–42. https://doi.org/10.1016/j.adolescence.2015.02.003.
[15]S. Cohen, T. Kamarck, R. Mermelstein, A Global Measure of Perceived Stress, J. Health Soc. Behav. 24 (1983) 385–396. https://doi.org/10.2307/2136404.
[16]W.C. Miller, H.A. Anton, A.F. Townson, Measurement properties of the CESD scale among individuals with spinal cord injury, Spinal Cord 46 (2008) 287–292. https://doi.org/10.1038/sj.sc.3102127.
[17]T. Baltrušaitis, P. Robinson, L.-P. Morency, OpenFace: An open source facial behavior analysis toolkit, in: 2016 IEEE Winter Conf. Appl. Comput. Vis. WACV, 2016: pp. 1–10. https://doi.org/10.1109/WACV.2016.7477553.
[18]Liu S., Guo X., DEPRESS: Dataset on Emotions, Performance, Responses, Environment, Stress, and Satisfaction of Students During COVID-19 Online Education, (n.d.). https://doi.org/10.7910/DVN/SJ8ILQ.
[19]M. Tavakol, R. Dennick, Making sense of Cronbach’s alpha, Int. J. Med. Educ. 2 (2011) 53–55. https://doi.org/10.5116/ijme.4dfb.8dfd.
&lt;watermark&gt;A2RT JCfE1N PKFSS&lt;/watermark&gt;

---


## Page 15

ARTICLE IN PRESS

[20] L.M. Feehan, J. Geldman, E.C. Sayre, C. Park, A.M. Ezzat, J.Y. Yoo, C.B. Hamilton, L.C. Li, Accuracy of Fitbit Devices: Systematic Review and Narrative Syntheses of Quantitative Data, JMIR MHealth UHealth 6 (2018) e10527. https://doi.org/10.2196/10527.
[21] G. Singh, Wearable IoT (w-IoT) artificial intelligence (AI) solution for sustainable smart-healthcare, Int. J. Inf. Manag. Data Insights 5 (2025) 100291. https://doi.org/10.1016/j.jjimei.2024.100291.
[22] J.M. Radin, N.E. Wineinger, E.J. Topol, S.R. Steinhubl, Harnessing wearable device data to improve state-level real-time surveillance of influenza-like illness in the USA: a population-based study, Lancet Digit. Health 2 (2020) e85–e93. https://doi.org/10.1016/S2589-7500(19)30222-5.

&lt;watermark&gt;ARTICLE IN PRESS&lt;/watermark&gt;